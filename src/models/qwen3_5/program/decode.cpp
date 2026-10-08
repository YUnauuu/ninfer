#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "core/nvtx.h"
#include "core/device.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// decode.cpp —— 解码：把一批 decode-ready 的序列推进一步，并把结果挂成待结算事务
//
// 与 prefill.cpp 是同一套骨架，分两段：
//   · execution:: 段是**一步执行**的原语：把这一轮的 ingress 拷上设备、走一遍目标前向、采样、把 egress
//     拷回来。它只回答"这一步怎么算"，不碰账本、不碰生命周期。capture 与 eager 走的是同一份函数体，
//     区别只在录不录图。
//   · detail:: 段是 ProgramImpl 的解码流程：校验成员 → 按（精确 B，前沿区间）选图 → 逐行装 ingress →
//     执行并同步 → 落账、把这一轮挂成 Pending，等外部裁决。
//
// 三个解码后端的骨架是同构的（普通 / MTP / DFlash），差别只在"一轮产出几个 token"：
//   · 普通轮：一轮一枚，采样结果在本文件里落账；
//   · MTP / DFlash 轮：一轮 1..K+1 枚（草案 + 修正/奖励），本文件只负责算出来并读回主机，"接受几枚"
//     是 Frontend 的裁决，由 commit / abort_pending 落到账本（见 transactions/commit.cpp）。
//
// 贯穿全文件的位置约束（与 prefill.cpp 是同一条不变式的前后两半）：
//   · 进入一轮时每一行必须 decode-ready：ledger_frontier == execution_frontier + 1，账本里多出来的
//     那一格是"已经采样、设备还没算过"的下一个输入（锚点）。执行前沿只按"设备真正算完"推进，
//     绝不被采样结果提前拉动。
//   · 账本、前缀身份、前缀摘要三者长度始终一致；设备落地之后才允许改前沿。
//   · Pending 是独占状态：同一行在结算之前不允许再下发下一轮。
//   · 图按"精确 B + 前沿区间"选取（见 DecodeGraphFamily），页面 ID、位置、采样配置都是图里的数据，
//     不是图的键——换 profile 要动图，不是纯查询。
// ============================================================================

// ============================================================================
// 一步执行的原语（execution::）
//
// 这一段只回答"这一轮怎么算"。ordinary_batch_body 是普通解码唯一的设备侧动作序列，capture 与 eager
// 共用它——两者必须逐条一致，否则图里录的与线上跑的会是两套东西。
// ============================================================================

namespace ninfer::models::qwen3_5::execution {
namespace {

// 普通解码一轮的完整设备侧动作，固定几步：拷 ingress 上设备 → 按本轮 B 切出各张张量的精确前缀（帧是
// 按最大并发准备的，只有前 batch_size 行有效）→ 目标模型前向 → 把最后一列 hidden 按行散写进各序列的
// 续跑隐藏槽 → 采样 → 把 egress 拷回钉住内存。
//
// 之所以包成"返回 lambda"而不是普通函数：capture 与 eager 是同一份函数体，谁调用它只决定录不录图。
// 行里同时带状态镜像的读槽与写槽（fork 未结算时两者不同，见 state_selectors），所以前向按行"读一份、
// 写一份"；页面行号、位置、采样配置都从这里进设备，不构成图的键。
auto ordinary_batch_body(OrdinaryBatchContext& state, std::int32_t batch_size,
                         ops::CausalAttentionExecutionEnvelope envelope) {
    return [&state, batch_size, envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency)) {
            throw std::logic_error("ordinary decode batch state is incomplete");
        }

        qwen3_5::OrdinaryDecodeState& ordinary = state.frame;
        CUDA_CHECK(cudaMemcpyAsync(ordinary.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_5::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                         {}, state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);

        Tensor tokens             = ordinary.tokens.slice(0, 0, batch_size);
        Tensor cache_positions    = ordinary.cache_positions.slice(0, 0, batch_size);
        Tensor rope_positions     = ordinary.rope_positions.slice(0, 0, batch_size);
        Tensor kv_rows            = ordinary.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = ordinary.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = ordinary.state_destination_slots.slice(0, 0, batch_size);
        Tensor hidden             = ordinary.hidden.slice(1, 0, batch_size);
        Tensor logits             = ordinary.logits.slice(1, 0, batch_size);
        Tensor sampled            = ordinary.sampled_tokens.slice(0, 0, batch_size);

        card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows, state_sources,
                                   state_destinations, envelope, hidden, logits);
        ops::scatter(hidden, state_destinations, state.continuation_hidden_store,
                     state.execution.device.stream);
        ops::sample(logits, sampled,
                    dimension(state.execution.parameters.model.resources().public_token_count),
                    ordinary.sampling, cache_positions, ops::kSamplePurposeDecode,
                    state.execution.work, state.execution.device.stream);
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, ordinary.egress.data,
                                   sizeof(qwen3_5::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

// 建图入口：把上面那份函数体录进一条图定义。传进来的包封必须是这张 profile 的稳定区间——图是按那段
// 区间录的，重放时不能拿本轮的实际值去撞它。
void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    capture_graph(state, definition, body);
}

// 执行入口：executable 非空就直接重放图，否则以 eager 方式跑同一份函数体。图与请求无关，只是被开关
// 按形状选中的私有资源。
void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::CausalAttentionExecutionEnvelope envelope,
                           DecodeGraphExecutable* executable) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::models::qwen3_5::execution

// ============================================================================
// ProgramImpl 的解码流程（detail::）
//
// 这一段才是"解码轮"本身：选图 → 装 ingress → 执行 → 落账 → 挂 Pending。执行原语（上一段）只是被它
// 调用的工具。发起与结算是分开的：本文件只做到"把这一轮变成待结算事务"，接受与否由外部裁决之后
// 才落地——普通行在 resolve_non_speculative_pending（本文件末尾），投机行走 commit / abort_pending。
// ============================================================================

namespace ninfer::models::qwen3_5::detail {

namespace {

// 解码图家族的三件小事：按（精确 B，前沿区间）找 profile、按拓扑类找已实例化的图、把某个 profile 装到
// 它的拓扑上。它们回答"这一轮用哪张图"，与请求无关；profile 覆盖的前沿区间是连续的，区间对不上就说明
// 建图阶段漏了形状，直接抛而不是临时补。装在拓扑上的 executable 是 Program 私有资源：同一拓扑类内换
// profile（区间变了，图定义就不同）要重装一次，所以 install 是改状态，不是查询。
DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label);

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label);

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label);

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label) {
    const auto it = std::find_if(
        family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& profile) {
            return profile.batch_size == batch_size && profile.min_execution_frontier <= frontier &&
                   frontier <= profile.max_execution_frontier;
        });
    if (it == family.profiles.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label) {
    const auto it = std::find_if(family.topologies.begin(), family.topologies.end(),
                                 [topology_class](const DecodeGraphTopology& topology) {
                                     return topology.topology_class == topology_class;
                                 });
    if (it == family.topologies.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph topology is unavailable");
    }
    return *it;
}

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label) {
    DecodeGraphTopology& topology   = select_graph_topology(family, profile.topology_class, label);
    const std::size_t profile_index = static_cast<std::size_t>(&profile - family.profiles.data());
    if (topology.installed_profile != profile_index) {
        topology.executable.update(profile.definition);
        topology.installed_profile = profile_index;
    }
    return topology.executable;
}

} // namespace

// 把一条 lane 的采样参数装上设备：主机侧留一份（每轮随 ingress 再发一次，采样配置是"随轮走的数据"），
// 设备侧写进这条 lane 的配置槽。惩罚项（presence / frequency）要逐 token 计数，计数缓冲挂在 lane 对应的
// token_counts 行上；不启用惩罚时指针置空并在本次写入中一并覆盖——槽位里绝不能残留上一轮的旧指针。
void ProgramImpl::install_sampling(SequenceState& sequence, RequestControl& request,
                                   const ops::SamplingConfig& config) {
    Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(sequence.lane), 1)
                        .view({dimension(parameters.model.resources().public_token_count)});
    request.sampling_host     = config;
    request.speculative_stats = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = speculative_backend != SpeculativeBackend::None,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
    };
    const bool penalties = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F;
    if (penalties) { CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream)); }
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &request.sampling_host,
                               sizeof(request.sampling_host), cudaMemcpyHostToDevice,
                               device.stream));
}

// 记下序列的"末尾 hidden"（目标模型算完最后一个 token 的那一列隐藏态）：零后缀复用与检查点捕获都靠它，
// 所以只能在设备确实算完之后调用。形状校验很严（BF16、[hidden, 1]），因为这份 hidden 会在别处被当成
// 目标模型的隐藏态使用，来路不明的张量不能被记成它。
void ProgramImpl::copy_tail(SequenceState& sequence, const Tensor& source) {
    if (source.dtype != DType::BF16 ||
        source.ne[0] != dimension(parameters.model.config().text.hidden_size) ||
        source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(sequence.tail_hidden.data, source.data, sequence.tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    sequence.tail_hidden_valid = true;
}

// 把回合帧里刚采出的那一枚 token 取回主机钉住内存——调用方只是"要看一眼这个数"（MTP 桥、强制续写、
// 预填收尾都这样），没必要等整轮结算。
void ProgramImpl::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

// 工作区的使用记账：启动时按阶段规划过工作区，这里记下实际用到的逻辑峰值，用于核对"规划够不够"。
// 只记账，不做任何分配（分配在 work.alloc 那一侧，用完由 work.reset 整段归还）。
void ProgramImpl::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

// DFlash 后端的"上下文补齐"：把已提交 target 的特征按位置补进 draft 自己的上下文 KV。
//
// 位置关系是这样的：draft 只能对"target 已经算过、特征已经落下"的位置做条件；target 推进到 F 之后，
// 序列的 dflash_context_frontier 允许落后于 F——差出来的那段特征就挂在 dflash->pending_features 上，
// 属于"已提交、还没被消费"。下一次提案之前（以及终结、发布检查点之前）必须先把这段补上，本函数就是
// 那次补齐：按行给出起始位置与长度，现场把待补特征拼成规整张量（prepare_ragged_prefix），再走一次
// DFlash 的上下文追加执行。用完 workspace 立刻归还，特征不跨轮保留。
//
// 三道前置检查分别防三类错误：成员（lane 不重、counts 落在 [1, K+1]，因为一次结算最多推进 K+1 个位置）、
// 存储（补齐后的范围不能超出该序列保留的 target 存储）、身份（KV 必须已绑定且状态槽位可解析）。
void ProgramImpl::enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                                std::span<const std::uint32_t> starts,
                                                std::span<const std::uint32_t> counts) {
    if (!is_masked_draft_backend(speculative_backend) || !dflash || !io.dflash_decode ||
        lanes.empty() || lanes.size() > max_concurrency || starts.size() != lanes.size() ||
        counts.size() != lanes.size()) {
        throw std::logic_error("DFlash context append has invalid membership");
    }

    std::uint32_t minimum_count = draft_window + 1U;
    std::uint32_t maximum_count = 0;
    *dflash_host_ingress        = {};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || counts[row] == 0 || counts[row] > draft_window + 1U ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("DFlash context append contains an invalid row");
        }
        SequenceState& sequence   = active_sequence(lane);
        const std::uint32_t start = starts[row];
        const std::uint64_t end64 = static_cast<std::uint64_t>(start) + counts[row];
        const std::uint32_t end   = static_cast<std::uint32_t>(end64);
        if (!sequence.kv || text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            (backend_kv_cache() && (!sequence.kv->backend ||
                                    backend_kv_addresses->bound_row(*sequence.kv->backend) < 0)) ||
            end64 > capacity) {
            throw std::logic_error("DFlash context append is outside retained target storage");
        }
        dflash_host_ingress->context_frontiers[row] =
            checked_i32(start, "DFlash append context frontier");
        dflash_host_ingress->execution_frontiers[row] =
            checked_i32(end, "DFlash append target frontier");
        dflash_host_ingress->dflash_kv_table_rows[row] =
            sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
        dflash_host_ingress->active_lanes[row]            = static_cast<std::int32_t>(lane);
        const StateImageSelectors selectors               = state_selectors(sequence);
        dflash_host_ingress->state_source_slots[row]      = selectors.source;
        dflash_host_ingress->state_destination_slots[row] = selectors.destination;
        // 上下文补齐只写 draft 自己的缓存；Main KV 的覆盖范围归 target 执行所有，其中也包含"已写、还没
        // 裁决"的那一段后缀——它要活到这一轮结算为止。DFlash2 只有固定的环形状态；DFlash 还要在这里
        // 把它那份 Full 后端 KV 的映射撑到 end。
        if (sequence.kv->backend) {
            backend_kv_addresses->ensure_mapped_to_tokens(*sequence.kv->backend, end,
                                                          device.stream);
        }
        minimum_count = std::min(minimum_count, counts[row]);
        maximum_count = std::max(maximum_count, counts[row]);
    }

    qwen3_5::DFlashDecodeState& frame = *io.dflash_decode;
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, dflash_host_ingress,
                               sizeof(qwen3_5::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                               device.stream));
    const auto batch                = static_cast<std::int32_t>(lanes.size());
    Tensor active_lane_tensor       = frame.active_lanes.slice(0, 0, batch);
    Tensor state_destination_tensor = frame.state_destination_slots.slice(0, 0, batch);
    Tensor device_starts            = frame.context_frontiers.slice(0, 0, batch);
    Tensor device_ends              = frame.execution_frontiers.slice(0, 0, batch);
    Tensor table_rows               = frame.dflash_kv_table_rows.slice(0, 0, batch);
    Tensor positions                = frame.append_positions.slice(1, 0, batch);
    Tensor device_counts            = frame.append_counts.slice(0, 0, batch);

    work.reset();
    Tensor features =
        work.alloc(DType::BF16, {dimension(parameters.draft->feature_projection.weight.k),
                                 static_cast<std::int32_t>(draft_window + 1U), batch});
    ops::prepare_ragged_prefix(dflash->pending_features, active_lane_tensor, device_starts,
                               device_ends, features, positions, device_counts, device.stream);

    execution::DFlashAppendContext state{{device, parameters, work, state_images->linear(),
                                          replay_records ? &*replay_records : nullptr, io,
                                          prefill_hidden, prefill_chunk, proposal_head},
                                         *dflash};
    mark_workspace_usage(workspace_plan.dflash_context);
    execution::dflash_append_context(state, features, positions, device_counts,
                                     state_destination_tensor, table_rows,
                                     {minimum_count, maximum_count});
}

// 设备事实入账前的最后一道校验：发放出来的 token 必须落在公开词表内。越界意味着设备侧算出了不可能
// 的值——这是必须炸掉的不变量错误，不能降级成"这条请求失败"。
void ProgramImpl::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= dimension(parameters.model.resources().public_token_count)) {
            throw std::runtime_error("target returned a token outside the public token domain");
        }
    }
}

// ============================================================================
// 三种后端的一轮解码（decode_ordinary / mtp / dflash）与分派（decode_raw）
//
// 三者的骨架相同：成员校验 → 选图 → 逐行装 ingress → 执行 → 同步 → 读回 egress → 落账并把行挂成
// Pending。区别只在落账的深度：普通轮把这一枚 token 当场写进账本，投机轮只登记产出与统计，账本等结算
// 再写。落账一律发生在同步之后，所以那里读到的都是设备确实写完的事实；失败路径也一致：同步收口、整批
// lane 作废再抛——已经发出去的 kernel 不试图挽回（设备可能已经写了 KV，没有"不带走一片云彩"的回滚）。
// 图与 eager 用同一条路径，只是包封取值不同：eager 用本轮实际前沿，图必须用 profile 的稳定区间。
// ============================================================================

// 普通后端的一轮紧凑解码：B 条 decode-ready 的序列共享一次模型前向，各出一枚 token。
//
// 顺序是刻意的，可以当模板看：
//   1) 成员校验：lane 不重复、生命周期为 Active、预算未用尽、KV 在册，且行形态满足"多一格"不变式
//      （ledger_frontier == execution_frontier + 1，账本、前缀身份、摘要三者等长）。这里任何一条不
//      成立都是逻辑错误——能被选进本轮的行早该被上游筛过。
//   2) 选图：有图时用 profile 的稳定区间当包封（图就是按那段录的），无图时用本轮实际前沿。图重放的
//      是形状与包封，页面行号、位置、采样配置都是重放时填进去的数据。
//   3) 逐行装 ingress，并把每条序列的 KV 映射撑到本轮需要的位置（输入 token 也要落进 KV，所以要
//      frontier + 1）。状态读写槽位随行给出，本轮前向按它"读一份、写一份"。
//   4) 一次同步收口：本轮唯一的等待点。
//   5) 落账：采样回的 token 逐一校验后追加进账本（连同前缀身份与摘要），KV 有效长度推进到设备确实写过
//      的位置，行挂成 Pending（PendingKind::Ordinary）。注意执行前沿与账本前沿到这里都**没有**动，
//      要等外部接受之后才推进（见 resolve_non_speculative_pending）——设备先写、前沿后动，是这一整套
//      轮次事务的共同节奏。多写的 KV 要么随后续前沿被用上，要么随整行作废一起消失。
runtime::BatchedGeneratedRound
ProgramImpl::decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                                   std::span<const runtime::RoundBudget> budgets,
                                   runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeOrdinaryRound, nvtx::Category::Decode,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::None) {
        throw std::logic_error("ordinary batch execution requires the ordinary backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("ordinary batch membership is invalid");
    }

    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("ordinary batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("ordinary batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto start = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeOrdinarySubmit, nvtx::Category::Decode,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable = nullptr;
        ops::CausalAttentionExecutionEnvelope envelope{maximum_frontier + 1, maximum_frontier + 1};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(ordinary_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "ordinary batch");
            executable = &install_graph_profile(ordinary_graphs, profile, "ordinary batch");
            envelope   = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence            = active_sequence(lanes[row]);
            const RequestControl& request      = requests[lanes[row]];
            const std::uint32_t frontier       = sequence.execution_frontier;
            ordinary_host_ingress->tokens[row] = sequence.ledger.back();
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(frontier, "ordinary batch position");
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "ordinary batch RoPE position") + sequence.rope_delta;
            ordinary_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            const StateImageSelectors selectors                 = state_selectors(sequence);
            ordinary_host_ingress->state_source_slots[row]      = selectors.source;
            ordinary_host_ingress->state_destination_slots[row] = selectors.destination;
            ordinary_host_ingress->sampling[row]                = request.sampling_host;
            ensure_sequence_kv_mapped(sequence, frontier + 1, 0);
        }

        execution::OrdinaryBatchContext schedule_state{
            {device, parameters, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
            decoder->text_kv,
            *io.ordinary,
            *ordinary_host_ingress,
            *ordinary_host_egress,
            state_images->continuation_hidden_store()};

        mark_workspace_usage(workspace_plan.ordinary_round);
        execution::ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                         envelope, executable);
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = active_sequence(lanes[row]);
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            const TokenId token        = ordinary_host_egress->sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            sequence.text_kv_valid = base_E + 1;
            commit_sequence_kv(sequence, sequence.text_kv_valid, 0);
            sequence.tail_hidden_valid = true;
            sequence.ledger.push_back(token);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
            sequence.prefix_digests.append_generated(std::span<const TokenId>(&token, 1),
                                                     sequence.rope_delta);
            request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                                 .base_E        = base_E,
                                                 .base_S        = base_S,
                                                 .prompt_tokens = 0,
                                                 .produced      = 1};
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens =
                std::span<const TokenId>(ordinary_host_egress->sampled_tokens.data(), lanes.size()),
            .timing = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

// MTP 后端的一轮：目标模型在一次前向里验证 [anchor, 若干草案]，逐行给出接受数量，并顺手起草下一轮。
//
// 与普通轮的结构性差别：
//   · 行的形状是固定的 K+1 列，逐行有效长度（extent）可以不同——受四样东西约束：draft 已经起草了多少
//     （mtp_draft_count）、窗口上限、剩余预算、容量余量。不足的列用 anchor 补齐，所以形状不表达语义，
//     语义由有效列数与基前沿共同界定。
//   · 产出不是"一枚 token"，而是每行的 licensed_tokens（count 枚，其中草案接受数 + 1 枚修正/奖励）、
//     接受数统计，以及下一轮要用的草案；本文件把它们读回主机，**不接受任何 token**——账本由结算写。
//   · 两种 KV 都要前瞻映射：文本 KV 要盖住本轮验证会写到的最远处（frontier + extent + 1），MTP 自己的
//     KV 还要再让出下一轮草案的位置（frontier + extent + draft_window）。这里撑开的是"映射"——地址
//     空间与页是两件事，页由之前的增长额度预留保证。
//   · 行必须 decode-ready，且 mtp_kv_valid == execution_frontier：起草 KV 的进度代表着"草案是在哪个
//     前沿上作出的"，落后于目标前沿的草案不能拿来验证。
// 统计（fallback_steps / accepted_per_position）只服务观测：extent 为 0 的行退化成普通一步，单独记一笔。
runtime::BatchedGeneratedRound
ProgramImpl::decode_mtp_batch(std::span<const std::uint32_t> lanes,
                              std::span<const runtime::RoundBudget> budgets,
                              runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeMtpRound, nvtx::Category::Mtp,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode ||
        decoder->mtp_cache() == nullptr) {
        throw std::logic_error("MTP batch execution requires the MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("MTP batch membership is invalid");
    }

    const std::uint32_t width      = draft_window + 1;
    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("MTP batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            backend_kv_addresses->bound_row(*sequence.kv->backend) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.mtp_kv_valid != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier ||
            sequence.mtp_draft_count > draft_window) {
            throw std::logic_error("MTP batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto started = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeMtpSubmit, nvtx::Category::Mtp,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable = nullptr;
        execution::MtpCausalAttentionEnvelopes envelopes =
            mtp_causal_attention_envelopes(maximum_frontier, draft_window, capacity);
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(mtp_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "MTP batch");
            executable = &install_graph_profile(mtp_graphs, profile, "MTP batch");
            envelopes = mtp_causal_attention_envelopes(profile.max_execution_frontier, draft_window,
                                                       capacity);
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = active_sequence(lanes[row]);
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1
                                                    : 0;
            const std::uint32_t extent =
                std::min({sequence.mtp_draft_count, draft_window, max_by_budget,
                          capacity - sequence.execution_frontier - 1});
            mtp_host_ingress->anchors[row]        = sequence.ledger.back();
            mtp_host_ingress->base_frontiers[row] = checked_i32(frontier, "MTP batch frontier");
            mtp_host_ingress->remaining_budgets[row] =
                checked_i32(budgets[row].generated_tokens_remaining, "MTP batch remaining budget");
            mtp_host_ingress->current_extents[row]      = static_cast<std::int32_t>(extent);
            mtp_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1);
            for (std::uint32_t j = 0; j < draft_window; ++j) {
                mtp_host_ingress->current_drafts[row * draft_window + j] =
                    j < extent ? sequence.mtp_drafts[j] : sequence.ledger.back();
            }
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::uint32_t position = frontier + std::min(j, extent);
                mtp_host_ingress->target_rope_positions[row * width + j] =
                    checked_i32(position, "MTP batch RoPE position") + sequence.rope_delta;
            }
            mtp_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            mtp_host_ingress->mtp_kv_table_rows[row] =
                backend_kv_addresses->bound_row(*sequence.kv->backend);
            const StateImageSelectors selectors            = state_selectors(sequence);
            mtp_host_ingress->state_source_slots[row]      = selectors.source;
            mtp_host_ingress->state_destination_slots[row] = selectors.destination;
            mtp_host_ingress->rope_deltas[row]             = sequence.rope_delta;
            mtp_host_ingress->sampling[row]                = request.sampling_host;
            ensure_sequence_kv_mapped(sequence, frontier + extent + 1,
                                      std::min(capacity, frontier + extent + draft_window));
        }

        execution::MtpBatchContext schedule_state{{device, parameters, work, state_images->linear(),
                                                   replay_records ? &*replay_records : nullptr, io,
                                                   prefill_hidden, prefill_chunk, proposal_head},
                                                  decoder->text_kv,
                                                  *decoder->mtp_cache(),
                                                  *io.mtp_decode,
                                                  *mtp_host_ingress,
                                                  *mtp_host_egress,
                                                  state_images->continuation_hidden_store()};

        mark_workspace_usage(workspace_plan.mtp_round);
        execution::mtp_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                    draft_window, envelopes, executable);
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = active_sequence(lanes[row]);
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = mtp_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = mtp_host_egress->accepted_drafts[row];
            const std::int32_t next_i     = mtp_host_egress->next_extents[row];
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || next_i < 0 ||
                next_i > static_cast<std::int32_t>(draft_window) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("MTP batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(mtp_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            const std::uint32_t pcur =
                static_cast<std::uint32_t>(mtp_host_ingress->current_extents[row]);
            if (pcur == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += pcur;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            request.pending = PendingCandidate{
                .kind          = PendingKind::Speculative,
                .base_E        = base_E,
                .base_S        = base_S,
                .prompt_tokens = 0,
                .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(mtp_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(mtp_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width,
            .timing     = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

// DFlash 后端的一轮：**同一轮里**先起草、再验证、再发放——草案不跨轮，所以没有"上一轮草案"这种状态，
// 行里带的是待补上下文的位置，而不是草案 token。
//
// 与 MTP 轮的两处结构性差别：
//   · 上下文允许落后：draft 只对"target 特征已经落下"的位置做条件，所以 context_frontier 可以落后于
//     执行前沿（但至多落后一个窗口——待补特征只保留这么长）。本轮提案会把这段补齐到基准前沿，因此轮末
//     把 dflash_context_frontier 记在 base_E：本轮新算出的那些位置的特征要等结算时才补
//     （见 enqueue_dflash_context_append 与 commit）。
//   · 包封分成两组，互不通用：draft 侧一组（local / full / append，描述草案模型自己的注意力与上下文
//     追加），target 验证一组（每行最多 extent + 1 列的列包封）——两侧的前沿推进本来就不是一回事。
// 行必须 decode-ready，且三种 KV 的进度要自洽：文本 KV 已到前沿、draft 上下文不超前、待补长度不出窗。
runtime::BatchedGeneratedRound
ProgramImpl::decode_dflash_batch(std::span<const std::uint32_t> lanes,
                                 std::span<const runtime::RoundBudget> budgets,
                                 runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeDFlashRound, nvtx::Category::DFlash,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (!is_masked_draft_backend(speculative_backend) || !io.dflash_decode || !dflash) {
        throw std::logic_error("DFlash batch execution requires the DFlash backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("DFlash batch membership is invalid");
    }

    const std::uint32_t width           = draft_window + 1U;
    std::uint32_t maximum_frontier      = 0;
    std::uint32_t maximum_target_tokens = 1;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("DFlash batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            (backend_kv_cache() && (!sequence.kv->backend ||
                                    backend_kv_addresses->bound_row(*sequence.kv->backend) < 0)) ||
            sequence.execution_frontier >= capacity ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            sequence.dflash_context_frontier > sequence.execution_frontier ||
            sequence.execution_frontier - sequence.dflash_context_frontier > width ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("DFlash batch row is not decode-ready");
        }
        const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                ? budgets[row].generated_tokens_remaining - 1U
                                                : 0U;
        const std::uint32_t extent =
            std::min({draft_window, max_by_budget, capacity - sequence.execution_frontier - 1U});
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
        maximum_target_tokens =
            std::max(maximum_target_tokens, sequence.execution_frontier + extent + 1U);
    }

    const auto started = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeDFlashSubmit, nvtx::Category::DFlash,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable    = nullptr;
        execution::DFlashEnvelopes envelopes = dflash_envelopes(0, maximum_frontier, draft_window);
        ops::CausalAttentionExecutionEnvelope target_envelope{1, maximum_target_tokens};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(dflash_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "DFlash batch");
            executable      = &install_graph_profile(dflash_graphs, profile, "DFlash batch");
            envelopes       = dflash_envelopes(profile.min_execution_frontier,
                                               profile.max_execution_frontier, draft_window);
            target_envelope = {
                1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                       capacity, static_cast<std::uint64_t>(profile.max_execution_frontier) +
                                     draft_window + 1ULL))};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = active_sequence(lanes[row]);
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1U
                                                    : 0U;
            const std::uint32_t extent =
                std::min({draft_window, max_by_budget, capacity - frontier - 1U});
            dflash_host_ingress->anchors[row] = sequence.ledger.back();
            dflash_host_ingress->execution_frontiers[row] =
                checked_i32(frontier, "DFlash batch frontier");
            dflash_host_ingress->context_frontiers[row] =
                checked_i32(sequence.dflash_context_frontier, "DFlash context frontier");
            dflash_host_ingress->proposal_valid_columns[row] = static_cast<std::int32_t>(width);
            dflash_host_ingress->proposal_extents[row]       = static_cast<std::int32_t>(extent);
            dflash_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1U);
            for (std::uint32_t column = 0; column < width; ++column) {
                const std::uint32_t position = frontier + std::min(column, extent);
                dflash_host_ingress->target_rope_positions[row * width + column] =
                    checked_i32(position, "DFlash target RoPE position") + sequence.rope_delta;
            }
            dflash_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            dflash_host_ingress->dflash_kv_table_rows[row] =
                sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
            dflash_host_ingress->active_lanes[row]       = static_cast<std::int32_t>(sequence.lane);
            const StateImageSelectors selectors          = state_selectors(sequence);
            dflash_host_ingress->state_source_slots[row] = selectors.source;
            dflash_host_ingress->state_destination_slots[row] = selectors.destination;
            dflash_host_ingress->sampling[row]                = request.sampling_host;
            ensure_sequence_kv_mapped(sequence, frontier + extent + 1U,
                                      backend_kv_cache() ? frontier : 0U);
        }

        execution::DFlashBatchContext schedule_state{
            {device, parameters, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
            decoder->text_kv,
            *dflash,
            *io.dflash_decode,
            *dflash_host_ingress,
            *dflash_host_egress,
            state_images->continuation_hidden_store()};

        mark_workspace_usage(workspace_plan.dflash_round);
        execution::dflash_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                       draft_window, envelopes, target_envelope, executable);
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = active_sequence(lanes[row]);
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = dflash_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = dflash_host_egress->accepted_drafts[row];
            const std::uint32_t extent =
                static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]);
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || accepted_i > static_cast<std::int32_t>(extent) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("DFlash batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(dflash_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            if (extent == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += extent;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            sequence.dflash_context_frontier = base_E;
            request.pending                  = PendingCandidate{
                                 .kind          = PendingKind::Speculative,
                                 .base_E        = base_E,
                                 .base_S        = base_S,
                                 .prompt_tokens = 0,
                                 .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(dflash_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(dflash_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width,
            .timing     = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

// 三种后端的唯一分派点。后端在启动时选定，不是运行期策略，所以这只是一次朴素的三岔，不是选择器；
// 三条支路返回同一种结果（这一轮的 token 与计时），由上层统一包成待结算票据。
runtime::BatchedGeneratedRound
ProgramImpl::decode_raw(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed_timing) {
    if (speculative_backend == SpeculativeBackend::None) {
        return decode_ordinary_batch(lanes, budgets, failed_timing);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        return decode_mtp_batch(lanes, budgets, failed_timing);
    }
    return decode_dflash_batch(lanes, budgets, failed_timing);
}

// ============================================================================
// 非投机行的结算（resolve_non_speculative_pending）
//
// 普通轮与"预填收尾的 Begin 轮"都只有一枚 token，接受与否没有中间地带：要么原样接受，要么整行取消
// （由调用方另行处理）。所以这里是单行结算里最简单的一种，与投机行的 commit / abort 并列。
// ============================================================================

// 把外部的裁决落到账本上，一次结算三件事，顺序不能换：
//   1) 前缀身份与摘要按"被接受的那一格"补齐——这是设备事实写成逻辑账本的那一步；
//   2) 前沿推进：执行前沿随"设备确实算过"推进，账本前沿停在其下一格（那条"多一格"不变式）；
//   3) 收尾：KV 的映射按有效长度收口、按需结算状态 fork、终结时清掉草案计数，生命周期落回 Active
//      或 Finishable。
//
// prefix_execution_split_after 是 Frontend 给出的"这一段内可以安全断开执行"的边界，本函数只把它转交给
// 前缀身份的提交，不解释它的语义。
runtime::ExecutionTiming ProgramImpl::resolve_non_speculative_pending(
    SequenceState& sequence, RequestControl& request, std::uint32_t accepted_tokens, bool terminal,
    std::optional<std::uint32_t> prefix_execution_split_after,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (request.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("pending resolution requires a pending generated round");
    }
    if ((request.pending.kind != PendingKind::Begin &&
         request.pending.kind != PendingKind::Ordinary) ||
        request.pending.produced != 1 || accepted_tokens != 1) {
        throw std::logic_error("non-speculative pending round must commit its single token");
    }

    const std::uint32_t base_ledger_frontier = request.pending.kind == PendingKind::Begin
                                                   ? request.pending.prompt_tokens
                                                   : request.pending.base_S;
    commit_generated_prefix_identity(
        sequence, base_ledger_frontier,
        std::span<const TokenId>(sequence.ledger).subspan(base_ledger_frontier, accepted_tokens),
        prefix_execution_split_after);

    // 两种轮次的前沿推进方式不同：Begin 的那一枚 token 没让 target 算过，所以执行前沿停在 prompt 上、
    // 账本照旧多出一格；Ordinary 的那一枚确实算过了，执行前沿与账本前沿一起前进一格，前缀重建工作的
    // 记账（advance_rebuild_work）也随之推进。
    switch (request.pending.kind) {
    case PendingKind::Begin:
        sequence.execution_frontier = request.pending.prompt_tokens;
        sequence.ledger_frontier    = request.pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
        advance_rebuild_work(sequence, request.pending.base_E + request.pending.produced,
                             prefill_chunk);
        sequence.execution_frontier = request.pending.base_E + request.pending.produced;
        sequence.ledger_frontier    = request.pending.base_S + request.pending.produced;
        break;
    case PendingKind::Speculative:
    case PendingKind::None:
        throw std::logic_error("non-speculative pending round has an invalid kind");
    }
    if (sequence.ledger_frontier != sequence.execution_frontier + 1 ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier ||
        sequence.prefix_digests.size() != sequence.ledger_frontier) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    // Begin 只发布了采样结果，并没有让 target 真正算过这一枚 token。所以"精确命中"的 fork 在这里仍然是
    // 读源不可变、写目的没写过的一对；要等第一次真正会写状态的解码 commit 才把它合上。后缀预填则早在
    // 提交的那个预填前沿上就合上了。
    if (request.pending.kind == PendingKind::Begin && terminal && sequence.state.fork_pending) {
        const StateImageSelectors selectors = state_selectors(sequence);
        timing.resume_submit();
        state_images->copy_slot(selectors.source, selectors.destination, device.stream);
        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        settle_state_fork(sequence);
    } else if (request.pending.kind == PendingKind::Ordinary) {
        settle_state_fork(sequence);
    }
    trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    if (terminal) { sequence.mtp_draft_count = 0; }
    request.lifecycle = terminal ? Lifecycle::Finishable : Lifecycle::Active;
    request.pending   = {};
    return timing.finish();
}


} // namespace ninfer::models::qwen3_5::detail
