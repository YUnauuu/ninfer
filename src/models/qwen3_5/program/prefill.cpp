#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/execution/linear.h"
#include "core/device.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

// ============================================================================
// prefill.cpp —— 预填充：把 prompt 算成设备上的事实，并采样出第一个生成 token
//
// 全文件分两段，职责完全不同：
//   · execution:: 段是**一步执行**的原语：给槽位、前沿与采样配置，跑一段 prefill chunk（文本或多模态）、
//     打一次 MTP 桥、或从单点 hidden 直接采样。它们不碰事务、不碰生命周期，算完就返回。
//   · detail:: 段是 ProgramImpl 的预填流程：先把物化事务留下的预留装成一条活着的 lane（start_sequence），
//     再逐块推进（advance_prefill），最后把采样到的 Begin token 挂成未结算事务，等外部裁决（resolve_*）。
//
// 两条贯穿全文件的位置约束：
//   · 预填要同时维护好几个"位置"，含义各不相同，不能互相推：
//       base   —— 本次复用了多少 token（复用前缀的长度，也叫复用前沿）；
//       cursor —— 设备上真正算到了 prompt 的第几个 token；
//       text_kv_valid / mtp_kv_valid / dflash_context_frontier —— 三种 KV 各自推进到了哪。
//     cursor 只增不减；三种 KV 推进度跟在它后面（可以落后，绝不允许超过）。
//   · 账本（ledger / prefix_identity / prefix_digests）在预填结束时恰好长 prompt_tokens + 1：多出来的
//     那一格是刚采样出、设备还没算过的 Begin token。此时执行前沿仍停在 prompt_tokens。
// ============================================================================

// ============================================================================
// 一步执行的原语（execution::）
//
// 这一段只回答"这一步怎么算"：填一张 TextContext 卡片、跑一段 chunk、把结果（推进了多少 token、是不是
// 最后一块）交回去。谁在什么前沿上算、能不能算，都是调用方的事。
// ============================================================================

namespace ninfer::models::qwen3_5::execution {
namespace {

// DFlash 后端要把它自己那一路上下文 KV 喂饱：预填每算一块，就把这一块的目标侧特征按位置追加进去。
// 这个函数把"追加"包成执行层要的 sink 回调——回调在算完一块时被调用，此时张量仍有效，workspace 随即
// 复用，所以必须当场消费掉，不能攒着。
//
// 批量维度在这里是塌的：预填只有一条序列，append_counts 只写第 0 行的一个标量；末尾的执行包封
// （min_count / max_count）两处填的是同一个精确值——这一块算了多少 token 是已知的，区间不浮动。
//
// rewrite_checkpoint 参数被显式忽略：DFlash 特征走这条回调，检查点捕获的 hidden 走卡片上另一条通道
// （rewrite_checkpoint_hidden），两者不是一回事。
DFlashFeatureSink make_dflash_prefill_sink(PrefillContext& state) {
    if (!state.execution.io.dflash_decode || state.dflash_host_ingress == nullptr) {
        throw std::logic_error("DFlash prefill controls are unavailable");
    }
    return dflash_feature_sink(
        state, [&state](const Tensor& features, const Tensor& positions, bool rewrite_checkpoint) {
            auto& frame  = *state.execution.io.dflash_decode;
            Tensor count = frame.append_counts.slice(0, 0, 1);
            Tensor lane  = frame.state_destination_slots.slice(0, 0, 1);
            Tensor row   = frame.dflash_kv_table_rows.slice(0, 0, 1);
            ops::set_i32_scalar(count, features.ne[1], state.execution.device.stream);
            const auto exact = static_cast<std::uint32_t>(features.ne[1]);
            dflash_append_context(state, features, positions, count, lane, row, {exact, exact});
            (void)rewrite_checkpoint;
        });
}

} // namespace

// 把一张 TextContext 卡片配置成"预填用"。两处要点：
//   · 状态动作恒为 UpdateInPlace：预填就是把状态从复用点一路算到 prompt 前沿，没有"另写一份、旧的那份
//     留着回滚"的选项——那是解码轮里投机验证才需要的。
//   · proposal_head：执行配置声明用完整提案头时，这里显式把卡片上的提案头清空；否则要求卡片上已经装着
//     一个优化过的提案头（没有就抛）。即"要么明确不要，要么必须真的有"，不接受来路不明的头。
void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t state_source_slot,
                         std::int32_t state_destination_slot, std::uint32_t mtp_proposal_extent) {
    card.set_sampling(sampling);
    card.set_linear_state_slots(state_source_slot, state_destination_slot);
    card.set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
    card.set_mtp_proposal_extent(mtp_proposal_extent);
    if (execution.proposal_head == ProposalHead::Full) {
        card.set_proposal_head(nullptr, nullptr, 0);
        return;
    }
    if (card.proposal_head() == nullptr || card.proposal_head_n() <= 0) {
        throw std::runtime_error("optimized proposal head is unavailable");
    }
}

// 跑一段预填 chunk。每次调用都**新建一张卡片**（TextContext 不缓存复用），把当前状态装上再交出去。
// 装上并传下去的四样东西各自的含义：
//   · 采样配置 / 状态槽位 / MTP 提案范围：由 configure_text_card 统一决定（见上）；
//   · rewrite_checkpoint_hidden：要捕获检查点时，本块最后一个 token 的 hidden 写到哪里；
//   · split_frontier：外部要求"这一段必须提前收尾在某处"（捕获点或改写点），-1 表示不分界。它约束的是
//     卡片内部在哪断开，不是"这块只能算到这里"——后者由调用方用 nominal_length 控制。
//   · DFlash sink：挂着 DFlash 后端时，算出来的特征顺路追加进它的上下文 KV。
// 多模态版本只多一个视觉会话参数，其余完全一致。
PrefillChunkResult prefill_text_chunk(PrefillContext& state, std::span<const TokenId> ids,
                                      std::uint32_t nominal_length,
                                      std::optional<std::uint32_t> split_frontier,
                                      bool finalize_at_end) {
    TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    const std::span<const int> prompt(ids.data(), ids.size());
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end,
                                  sink);
    }
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end);
}

PrefillChunkResult prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                                            VisionPrefillSession& vision,
                                            std::uint32_t nominal_length,
                                            std::optional<std::uint32_t> split_frontier,
                                            bool finalize_at_end) {
    TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision,
                                  finalize_at_end, sink);
    }
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision, finalize_at_end);
}

// 视觉 prompt 上的 MTP 桥接。"桥"补的是这样一件东西：MTP 后端起草下一步需要一个"上一个位置的隐藏态"，
// 而复用一段视觉前缀时这个隐藏态不在文本 KV 里（视觉位置没有走过文本塔），得从视觉编码结果里取。
// 于是两个前提缺一不可：桥必须正好架在复用前沿的前一个位置（position + 1 == text_kv_base），且该位置
// 在视觉 scatter 元数据里确实对应一个视觉 token——对不上就是"要补的隐藏态根本不存在"，直接抛。
// 位置与模态都对上之后，从视觉 chunk 的 embeddings 里按列取出这一格，作为组合输入交给桥接调用，
// 使起草用的输入与当初预填那一格时模型的输入一致。
void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge) {
    if (!state.mtp_kv.valid() || bridge.previous_hidden == nullptr || state.text_kv_base == 0 ||
        bridge.position < 0 ||
        static_cast<std::uint32_t>(bridge.position) + 1 != state.text_kv_base) {
        throw std::logic_error("multimodal MTP bridge does not match the reusable frontier");
    }

    Tensor bridge_token = state.execution.io.mtp->target_input_ids.slice(0, 0, 1);
    const TokenId token = prompt.token_ids[state.text_kv_base];
    CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token), cudaMemcpyHostToDevice,
                               state.execution.device.stream));

    Tensor visual_embedding;
    const Tensor* composed_embedding = nullptr;
    if (prompt.token_types[state.text_kv_base] != 0) {
        const VisionChunk chunk = vision.prepare_chunk(state.text_kv_base, 1);
        if (chunk.control == nullptr) {
            throw std::logic_error("visual MTP bridge has no encoded Vision item");
        }
        const auto& scatter = chunk.control->scatter_indices;
        const auto column   = std::lower_bound(scatter.begin(), scatter.end(),
                                               static_cast<std::int32_t>(state.text_kv_base));
        if (column == scatter.end() || *column != static_cast<std::int32_t>(state.text_kv_base) ||
            static_cast<std::uint8_t>(chunk.control->modality) !=
                prompt.token_types[state.text_kv_base]) {
            throw std::logic_error("visual MTP bridge does not match Vision scatter metadata");
        }
        visual_embedding =
            chunk.embeddings.slice(1, static_cast<std::int32_t>(column - scatter.begin()), 1);
        composed_embedding = &visual_embedding;
    }

    mtp_bridge_and_propose(state, bridge_token, *bridge.previous_hidden, bridge.position,
                           bridge.rope_position, false, composed_embedding);
}

// 零后缀复用专用的采样：prompt 完全命中、没有新 token 要算，于是直接拿序列上留存的 tail hidden 走输出
// 头 → 采样，产出 Begin token。留着这个入口的意义就是"一步都不多算也得能出 token"。
// 前后各一次 work.reset() 是刻意的：输出头投影要用 workspace，用完立刻还回去，不在这里攒临时占用。
// 形状校验很严（BF16、[hidden,1]）：这个 hidden 来自别处（复用来源或上一轮），必须确认它确实是目标模型
// 的隐藏态，而不是别的什么张量。
void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose) {
    if (hidden.dtype != DType::BF16 ||
        hidden.ne[0] != dimension(state.execution.parameters.model.config().text.hidden_size) ||
        hidden.ne[1] != 1 || hidden.ne[2] != 1 || hidden.ne[3] != 1 || hidden.data == nullptr) {
        throw std::invalid_argument("sample_from_hidden requires BF16 [hidden,1]");
    }
    state.execution.work.reset();
    Tensor logits = state.execution.io.logits.slice(1, 0, 1);
    project(hidden, state.execution.parameters.text.output_head, logits, state.execution.work,
            state.execution.device.stream);
    CUDA_CHECK(cudaMemcpyAsync(state.execution.io.pos.data, &absolute_position,
                               sizeof(absolute_position), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    ops::sample(logits, state.execution.io.token,
                dimension(state.execution.parameters.model.resources().public_token_count),
                state.sampling, state.execution.io.pos, purpose, state.execution.work,
                state.execution.device.stream);
    state.execution.work.reset();
}

} // namespace ninfer::models::qwen3_5::execution

// ============================================================================
// ProgramImpl 的预填流程（detail::）
//
// 这一段才是"预填"本身：start_sequence 把一次物化落成活着的序列，advance_prefill 把它逐块推到 prompt
// 前沿并采样出第一个 token，resolve_* 再把外部的裁决落到账本上。执行原语（上一段）只是被它调用的工具。
// ============================================================================

namespace ninfer::models::qwen3_5::detail {

namespace {

// 取某个 token 的三轴 RoPE 位置。注意 positions 是**三个平面首尾相接**的一维数组（第 i 轴从
// i * tokens 处起算），不是 [3, tokens] 那样的二维布局；长度校验 3 * tokens 就是这个意思。
std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token);

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token) {
    const std::size_t tokens = prompt.token_ids.size();
    if (token >= tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("MTP bridge position is outside prepared prompt metadata");
    }
    return {prompt.positions[token], prompt.positions[tokens + token],
            prompt.positions[2 * tokens + token]};
}

} // namespace

// ============================================================================
// start_sequence —— 物化事务的物理发布点（把预留装成一条活着的 lane）
//
// 走进这里时，物化事务已经攒下一堆"已认领但还不算数"的东西：状态镜像的预留与 fork 目的地、主 KV 与
// 后端 KV 的地址空间（可能是新造的，也可能是 COW 出来的）、两种 KV 的页额度、待落地的前缀身份。
// 这个函数把它们一次性装到 lane 上；返回之后，这条序列就有了 KV、状态与账本，可以开始预填。
//
// 按复用路径分派成四种装法，**互不共用**：
//   · Root：全新序列。旧的 KV / 状态一律释放，直接接管事务留的 root 地址空间，账本清空从头写。
//   · Retain（保留来源）：来源（目录里的私有续跑，或共享前缀）要留着继续被别人用，目的地是新的一份。
//     状态按来源副本的落地情况分三种走法（就地指过去 / 只分裂设备副本身份 / 真 fork 并挂 fork_pending）；
//     共享来源还要把引用计数 +1。
//   · PrivateEndpoint / 检查点恢复：来源就是本 lane 已经持有的那一份（resident），接管方式由
//     activate_consumed_state 决定是"整份挪过来（Move）"还是"从检查点 fork"。
//   · 其余路径直接抛——这里不接受任何计划外的复用路径。
//
// 三处容易看漏的地方：
//   · 不保留来源时，目的地地址空间的旧尾部要先**破坏性裁掉**，而且先 preflight 再动手：能不能裁是几何
//     性质（尾部页的列数能否释放、Host 副本能否原子回收），不是"失败了再回滚"的动作。
//   · 前缀 COW 一旦 commit，源地址空间当场释放——这是"新地址空间顶替旧来源"的那一步。
//   · 结尾把物化期攒的 ledger / prefix_identity / prefix_digests **swap** 进来（不是拷贝）：那三个数组
//     本来就是物化期的暂存，换手之后暂存即空。
//
// 失败语义：整段是"要么全成、要么整条拆掉"。catch 里先同步设备，再 best-effort 清 lane，然后原样抛出；
// 所以它不 noexcept，失败后 lane 回到 Empty（清理是尽力而为，不保证资源全部回池）。
// ============================================================================
void ProgramImpl::start_sequence(std::uint32_t lane, SequenceState& sequence,
                                 MaterializationTransaction& transaction) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    RequestControl& request = requests[lane];
    if (!transaction.plan || transaction.plan->impl_ == nullptr || !transaction.prepared ||
        !request.prefill) {
        throw std::invalid_argument("materialization staging is incomplete");
    }
    AdmissionCandidateImpl& request_plan = *transaction.plan->impl_;
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("staged prefill requires a free request lane");
    }
    auto& staged                           = *request.prefill;
    const auto started                     = Clock::now();
    const std::uint32_t prompt_tokens      = staged.prompt_tokens;
    const std::uint32_t base               = staged.base;
    const std::uint32_t initial_mtp_extent = staged.initial_mtp_extent;
    request.lifecycle                      = Lifecycle::Empty;
    try {
        const std::uint32_t state_slots = request_plan.demand.active_entitlement.device.state_slots;
        const bool preserving_source =
            (transaction.has_source || transaction.has_shared_source) &&
            transaction.source_mode == runtime::PrivateSourceMode::Retain;
        const bool text_prefix_fork    = request_plan.text_prefix_fork_required;
        const bool backend_prefix_fork = request_plan.backend_prefix_fork_required;
        // 装法从这里分岔。每个分支开头都是同一件事：把"这一步成立的前提"整组核对一遍（缺一项就抛），
        // 核对的全是"事务当初承诺的东西现在是否还在、形状是否还对得上"。
        if (request_plan.reuse == ReusePath::Root) {
            if (transaction.reserved_state_count != state_slots || state_slots == 0 ||
                !transaction.root_text_address || !transaction.text_activation ||
                transaction.root_backend_address.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0) ||
                transaction.backend_activation.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0)) {
                throw std::logic_error("root materialization reservations are incomplete");
            }
            release_sequence_kv(sequence);
            release_sequence_state(sequence);
            sequence.state = ActiveStateBinding{.read  = transaction.reserved_states[0],
                                                .write = transaction.reserved_states[0]};
            transaction.reserved_states[0] = {};
            if (state_slots == 2) {
                sequence.reserved_state        = transaction.reserved_states[1];
                transaction.reserved_states[1] = {};
            }
            transaction.reserved_state_count = 0;

            SequenceKVBundle bundle{.text = *transaction.root_text_address};
            transaction.root_text_address.reset();
            if (transaction.root_backend_address) {
                bundle.backend = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
            }
            sequence.kv.emplace(bundle);
        } else if (preserving_source) {
            // 保留来源时，来源要么是私有续跑、要么是共享前缀，**恰有其一**：两者同真或同假都是错的
            //（private_source_ready == shared_source_ready 就是这个意思）。
            const bool private_source_ready = transaction.has_source &&
                                              transaction.source_index < continuation_capacity &&
                                              continuation_slots[transaction.source_index].role ==
                                                  ContinuationSlotRole::Catalogued;
            const bool shared_source_ready =
                transaction.has_shared_source &&
                transaction.shared_source_index < shared_prefix_capacity &&
                shared_prefix_slots[transaction.shared_source_index].role ==
                    SharedPrefixSlotRole::Catalogued;
            if (private_source_ready == shared_source_ready ||
                transaction.reserved_state_count != state_slots || state_slots == 0 ||
                !transaction.root_text_address || !transaction.text_prefix_fork ||
                !transaction.prefix_forks_ready ||
                transaction.root_backend_address.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0)) {
                throw std::logic_error("retained materialization is incomplete");
            }
            const StateImageHandle selected =
                private_source_ready
                    ? selected_state(continuation_states[transaction.source_index],
                                     request_plan.reuse, request_plan.selected_checkpoint)
                    : shared_prefix_states[transaction.shared_source_index].state;
            const StateImageHandle current = transaction.reserved_states[0];
            // 来源副本落在哪，决定目的地怎么装：三种情况处理的都是同一件事——"来源那份状态不能被就地
            // 改坏"，区别只在省多少搬运。
            //   · 来源只在 Host：目的地那份活跃副本已经是独立的一份，直接指过去；
            //   · 身份分裂（split_state_identity）：源与目的地共用同一份设备副本，只把身份拆成两个；
            //   · 其余：走 begin_fork，先把读侧锁在来源上、写侧落在目的地，由 fork_pending 记着还没
            //     真正分家，等第一次真的要写状态时才落定。
            if (state_store->residency(selected) == StateReplicaResidency::HostOnly) {
                if (state_store->role(current) != StateImageRole::ActiveMutable) {
                    throw std::logic_error("Host retained Fork destination was not published");
                }
                sequence.state = ActiveStateBinding{.read = current, .write = current};
            } else if (transaction.split_state_identity) {
                if (!private_source_ready ||
                    state_store->residency(selected) != StateReplicaResidency::Both) {
                    throw std::logic_error("StateImage identity split source changed");
                }
                state_store->split_device_replica_identity(selected, current);
                sequence.state = ActiveStateBinding{.read = current, .write = current};
            } else {
                const StateImageSelectors selectors = state_store->begin_fork(selected, current);
                if (is_masked_draft_backend(speculative_backend)) {
                    state_images->copy_dflash_local(selectors.source, selectors.destination,
                                                    device.stream);
                }
                sequence.state = ActiveStateBinding{
                    .read           = selected,
                    .write          = current,
                    .fork_pending   = true,
                    .read_ownership = StateReadOwnership::ExternalOwner,
                };
            }
            transaction.reserved_states[0]   = {};
            transaction.split_state_identity = false;
            if (state_slots == 2) {
                sequence.reserved_state        = transaction.reserved_states[1];
                transaction.reserved_states[1] = {};
            }
            transaction.reserved_state_count = 0;
            sequence.rewrite_state.reset();
            sequence.rewrite_checkpoint = {};

            SequenceKVBundle bundle{.text = *transaction.root_text_address};
            transaction.root_text_address.reset();
            if (transaction.root_backend_address) {
                bundle.backend = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
            }
            sequence.kv.emplace(bundle);
        } else {
            if (request_plan.state_fork_required !=
                transaction.state_fork_destination.has_value()) {
                throw std::logic_error("private materialization StateImage Fork is incomplete");
            }
            if (transaction.reserved_state_count > 1 ||
                (transaction.reserved_state_count != 0 && sequence.reserved_state)) {
                throw std::logic_error("private materialization StateImage reservation is invalid");
            }
            if (transaction.reserved_state_count == 1) {
                sequence.reserved_state          = transaction.reserved_states[0];
                transaction.reserved_states[0]   = {};
                transaction.reserved_state_count = 0;
            }
        }

        // 不保留来源时，目的地的旧尾部必须**破坏性裁掉**：复用只覆盖到某个前沿，前沿之后留着的是上一次
        // 在这个地址空间里算出来的内容，不清干净就不配当新序列的底子。
        // 裁之前先做 preflight，是因为"能不能裁"是几何性质（尾部页的列数是否正好可释放、Host 副本能否
        // 原子回收），而裁本身已经是不可逆的物理动作——先把不可行的情形挡在前面，别做到一半才发现。
        if (!preserving_source) {
            std::array<HostKVPageReplicaRelease, 2> stale_tail_replicas{};
            std::size_t stale_tail_count           = 0;
            const auto preflight_inactive_truncate = [&](KVAddressSpaceStore& addresses,
                                                         LogicalKVPageStore& pages,
                                                         KVAddressSpaceHandle address,
                                                         std::optional<std::uint32_t> frontier) {
                if (!frontier ||
                    (addresses.committed_frontier(address) == *frontier &&
                     addresses.mapped_pages(address) == kv_pages_for_frontier(*frontier))) {
                    return;
                }
                bool releases_tail               = false;
                const std::uint32_t target_pages = kv_pages_for_frontier(*frontier);
                if (target_pages != 0) {
                    const LogicalKVPageHandle tail =
                        addresses.logical_page(address, target_pages - 1U);
                    const std::uint32_t columns =
                        *frontier -
                        (target_pages - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
                    if (columns != pages.committed_columns(tail) && pages.host_resident(tail)) {
                        if (host_kv_extents == nullptr ||
                            stale_tail_count == stale_tail_replicas.size()) {
                            throw std::logic_error("stale Host KV tail replica is not releasable");
                        }
                        stale_tail_replicas[stale_tail_count++] =
                            HostKVPageReplicaRelease{.pages = &pages, .page = tail};
                        releases_tail = true;
                    }
                }
                if (!addresses.can_destructive_truncate_inactive(address, *frontier,
                                                                 releases_tail)) {
                    throw std::logic_error(
                        "selected private KV frontier is not destructively materializable");
                }
            };
            if (!sequence.kv) {
                throw std::logic_error("materialization destination has no KV address space");
            }
            if (!text_prefix_fork) {
                preflight_inactive_truncate(*text_kv_addresses, *text_kv_pages, sequence.kv->text,
                                            transaction.text_activation_frontier);
            }
            if (sequence.kv->backend && !backend_prefix_fork) {
                preflight_inactive_truncate(*backend_kv_addresses, *backend_kv_pages,
                                            *sequence.kv->backend,
                                            transaction.backend_activation_frontier);
            }
            if (stale_tail_count != 0) {
                const std::span<const HostKVPageReplicaRelease> releases(stale_tail_replicas.data(),
                                                                         stale_tail_count);
                if (!host_kv_extents->release_page_replicas(releases)) {
                    throw std::logic_error(
                        "stale Host KV tail replicas cannot be released atomically");
                }
            }
            if (!text_prefix_fork && transaction.text_activation_frontier &&
                (text_kv_addresses->committed_frontier(sequence.kv->text) !=
                     *transaction.text_activation_frontier ||
                 text_kv_addresses->mapped_pages(sequence.kv->text) !=
                     kv_pages_for_frontier(*transaction.text_activation_frontier))) {
                text_kv_addresses->destructive_truncate_inactive(
                    sequence.kv->text, *transaction.text_activation_frontier);
            }
            if (!backend_prefix_fork && transaction.backend_activation_frontier &&
                sequence.kv->backend &&
                (backend_kv_addresses->committed_frontier(*sequence.kv->backend) !=
                     *transaction.backend_activation_frontier ||
                 backend_kv_addresses->mapped_pages(*sequence.kv->backend) !=
                     kv_pages_for_frontier(*transaction.backend_activation_frontier))) {
                backend_kv_addresses->destructive_truncate_inactive(
                    *sequence.kv->backend, *transaction.backend_activation_frontier);
            }
            if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
        }
        if ((text_prefix_fork || backend_prefix_fork) && !transaction.prefix_forks_ready) {
            throw std::logic_error("materialization prefix forks are incomplete");
        }
        // 前缀 COW 的提交点：commit_prefix_fork 把新地址空间接上，源地址空间随即可以释放——不保留来源
        // 时这里当场释放它。后端 KV 同理，只是它可能压根不存在（backend 是 optional）。
        if (text_prefix_fork) {
            text_kv_addresses->commit_prefix_fork(std::move(*transaction.text_prefix_fork),
                                                  device.stream);
            transaction.text_prefix_fork.reset();
            if (!preserving_source) {
                const KVAddressSpaceHandle source_address = sequence.kv->text;
                sequence.kv->text                         = *transaction.root_text_address;
                transaction.root_text_address.reset();
                if (!text_kv_addresses->release(source_address)) {
                    throw std::logic_error("consumed Text KV source remained pinned after COW");
                }
            }
        } else {
            text_kv_addresses->commit_activation(std::move(*transaction.text_activation),
                                                 device.stream);
            transaction.text_activation.reset();
        }
        if (backend_prefix_fork) {
            backend_kv_addresses->commit_prefix_fork(std::move(*transaction.backend_prefix_fork),
                                                     device.stream);
            transaction.backend_prefix_fork.reset();
            if (!preserving_source) {
                const KVAddressSpaceHandle source_address = *sequence.kv->backend;
                sequence.kv->backend                      = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
                if (!backend_kv_addresses->release(source_address)) {
                    throw std::logic_error("consumed Backend KV source remained pinned after COW");
                }
            }
        } else if (transaction.backend_activation) {
            backend_kv_addresses->commit_activation(std::move(*transaction.backend_activation),
                                                    device.stream);
            transaction.backend_activation.reset();
        }
        transaction.prefix_forks_ready = false;
        transaction.text_activation_frontier.reset();
        transaction.backend_activation_frontier.reset();
        transaction.prepared = false;

        const bool preserve_rewrite =
            request_plan.rewrite_disposition == RewriteCheckpointDisposition::RetainExisting;
        // 把一份"被消费掉的检查点"接管成活跃状态，两种方式取决于计划里那份 checkpoint 是要被吃掉
        //（Move：整份挪过来，挪之前必须确认没有别的引用）还是留着（Fork：写侧落到目的地，读侧仍在
        // 检查点上）。fork 时还要算清读侧所有权——引用全在本序列血统之内才算 LineageCheckpoint，否则
        // 是外部还握着，读侧不能被当成"自己人"看待。
        const auto activate_consumed_state = [&](StateImageHandle selected) {
            if (!request_plan.state_fork_required) {
                if (transaction.state_fork_destination ||
                    state_store->checkpoint_references(selected) != 0) {
                    throw std::logic_error("planned StateImage Move is no longer valid");
                }
                state_store->move_checkpoint_to_active(selected);
                sequence.state = ActiveStateBinding{.read = selected, .write = selected};
                return;
            }
            if (!transaction.state_fork_destination ||
                state_store->checkpoint_references(selected) == 0) {
                throw std::logic_error("planned StateImage Fork is no longer valid");
            }
            const StateImageHandle destination = *transaction.state_fork_destination;
            if (transaction.state_restored) {
                if (state_store->role(destination) != StateImageRole::ActiveMutable) {
                    throw std::logic_error("restored StateImage Fork destination is unavailable");
                }
                sequence.state = ActiveStateBinding{.read = destination, .write = destination};
            } else {
                const std::uint32_t references = state_store->checkpoint_references(selected);
                const std::uint32_t lineage_references =
                    owned_checkpoint_references(sequence, selected);
                if (lineage_references > references) {
                    throw std::logic_error("consumed StateImage Fork ownership is inconsistent");
                }
                const StateReadOwnership read_ownership =
                    lineage_references == references ? StateReadOwnership::LineageCheckpoint
                                                     : StateReadOwnership::ExternalOwner;
                const StateImageSelectors selectors =
                    state_store->begin_fork(selected, destination);
                if (is_masked_draft_backend(speculative_backend)) {
                    state_images->copy_dflash_local(selectors.source, selectors.destination,
                                                    device.stream);
                }
                sequence.state = ActiveStateBinding{
                    .read           = selected,
                    .write          = destination,
                    .fork_pending   = true,
                    .read_ownership = read_ownership,
                };
            }
            transaction.state_fork_destination.reset();
        };
        // 第二次按复用路径分岔，这次定的是**逻辑起点**：账本长度、三种 KV 推进度、末尾 hidden 是否可用、
        // 以及共享来源的引用计数。物理装法在上面，逻辑起点在这里，两者必须对得上（对不上就在各自的校验
        // 里抛）。每个分支最后都把 state 视图与 KV 绑定刷新一遍——上面的物理改动到此对执行层可见。
        if (request_plan.reuse == ReusePath::Root) {
            sequence.rewrite_checkpoint = {};
            ordered_reset(sequence);
            sequence.ledger.clear();
            sequence.prefix_digests.clear();
            sequence.text_kv_valid = 0;
            sequence.mtp_kv_valid  = 0;
        } else if (preserving_source) {
            const SequenceState* private_source =
                transaction.has_source ? &continuation_states[transaction.source_index] : nullptr;
            SharedPrefixState* shared_source =
                transaction.has_shared_source
                    ? &shared_prefix_states[transaction.shared_source_index]
                    : nullptr;
            const std::uint32_t source_text_frontier =
                private_source != nullptr ? private_source->text_kv_valid : shared_source->frontier;
            if (!sequence.kv || source_text_frontier < base) {
                throw std::logic_error("retained prefix has incomplete Text KV");
            }
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base       = base == 0 ? 0 : base - 1U;
                const std::uint32_t source_backend = private_source != nullptr
                                                         ? private_source->mtp_kv_valid
                                                         : shared_source->backend_frontier;
                if (!request_plan.prepare_mtp || source_backend < mtp_base) {
                    throw std::logic_error("retained prefix has incomplete MTP KV");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (is_masked_draft_backend(speculative_backend)) {
                const std::uint32_t source_backend = private_source != nullptr
                                                         ? private_source->dflash_context_frontier
                                                         : shared_source->frontier;
                if (source_backend < base) {
                    throw std::logic_error("retained prefix has incomplete DFlash KV");
                }
                sequence.dflash_context_frontier = base;
            }
            sequence.tail_hidden_valid =
                base == prompt_tokens &&
                (private_source != nullptr ? private_source->tail_hidden_valid
                                           : shared_source->tail_hidden_valid);
            if (shared_source != nullptr) {
                if (shared_source->active_references == std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error("shared-prefix active reference overflow");
                }
                ++shared_source->active_references;
                sequence.shared_prefix_references.push_back(transaction.shared_source_index);
            }
            refresh_state_views(sequence);
            bind_sequence_kv(sequence);
        } else if (request_plan.reuse == ReusePath::PrivateEndpoint) {
            if (!state_store->valid(sequence.state.read) ||
                sequence.state.read != sequence.state.write || sequence.state.fork_pending ||
                state_store->role(sequence.state.read) != StateImageRole::CheckpointImmutable) {
                throw std::logic_error("resident endpoint StateImage is not movable");
            }
            if (!preserve_rewrite && sequence.rewrite_state) {
                const StateImageHandle dropped = *sequence.rewrite_state;
                state_store->release_checkpoint_reference(dropped);
                sequence.rewrite_state.reset();
                sequence.rewrite_checkpoint = {};
                if (dropped != sequence.state.read &&
                    state_store->checkpoint_references(dropped) == 0 &&
                    !state_store->release(dropped)) {
                    throw std::logic_error("dropped rewrite StateImage could not be released");
                }
            }
            activate_consumed_state(sequence.state.read);
            if (!sequence.kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            if (sequence.text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than the append frontier");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error("resident MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (is_masked_draft_backend(speculative_backend) &&
                       sequence.dflash_context_frontier != base) {
                throw std::logic_error("resident DFlash context is not at the append frontier");
            }
            bind_sequence_kv(sequence);
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.text_kv_valid = base;
            sequence.ledger.resize(base);
            sequence.prefix_digests.truncate(base);
            reserve_state_entitlement(sequence, state_slots);
            refresh_state_views(sequence);
        } else if (is_rewrite_checkpoint_restore(request_plan.reuse)) {
            if (!sequence.kv || sequence.text_kv_valid < base) {
                throw std::logic_error("resident rewrite checkpoint has no complete KV allocation");
            }
            if (!sequence.rewrite_state || !state_store->valid(*sequence.rewrite_state) ||
                state_store->role(*sequence.rewrite_state) != StateImageRole::CheckpointImmutable ||
                (sequence.endpoint_valid &&
                 (!state_store->valid(sequence.state.read) ||
                  sequence.state.read != sequence.state.write || sequence.state.fork_pending ||
                  state_store->role(sequence.state.read) != StateImageRole::CheckpointImmutable))) {
                throw std::logic_error("resident rewrite StateImage is not movable");
            }
            const StateImageHandle checkpoint = *sequence.rewrite_state;
            if (sequence.endpoint_valid && sequence.state.read == checkpoint) {
                throw std::logic_error("resident endpoint aliases its rewrite StateImage");
            }
            if (sequence.endpoint_valid && !state_store->release(sequence.state.read)) {
                throw std::logic_error("superseded endpoint StateImage could not be released");
            }
            if (!preserve_rewrite) {
                state_store->release_checkpoint_reference(checkpoint);
                sequence.rewrite_state.reset();
                sequence.rewrite_checkpoint = {};
            }
            activate_consumed_state(checkpoint);
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error(
                        "rewrite-checkpoint MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (is_masked_draft_backend(speculative_backend)) {
                if (!dflash || (backend_kv_cache() && !sequence.kv->backend) ||
                    sequence.dflash_context_frontier < base) {
                    throw std::logic_error("planned DFlash rewrite checkpoint is unavailable");
                }
                sequence.dflash_context_frontier = base;
            }
            bind_sequence_kv(sequence);
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.tail_hidden_valid = base == prompt_tokens;
            sequence.ledger.resize(base);
            sequence.prefix_digests.truncate(base);
            reserve_state_entitlement(sequence, state_slots);
            refresh_state_views(sequence);
        } else {
            throw std::logic_error("request plan has an invalid prefix reuse path");
        }

        // 收口：新序列此刻还没有"可发布的端点"（endpoint_valid = false）；两种 KV 先按 base 收齐，再把整
        // 句长度所需的页覆盖铺出来。覆盖是**下界**语义（可能早已映射得更远，只有显式 truncate 才会收回），
        // 决定哪些 token 有效的是 commit_frontier，不是映射宽度。
        sequence.endpoint_valid = false;
        if (!preserving_source) { trim_sequence_kv(sequence, base, backend_kv_valid(sequence)); }
        bind_sequence_kv(sequence);
        const std::uint32_t backend_materialized =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min(capacity,
                           prompt_tokens + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U))
            : speculative_backend == SpeculativeBackend::DFlash ? prompt_tokens
                                                                : 0U;
        ensure_sequence_kv_mapped(sequence, prompt_tokens, backend_materialized);
        // 采样配置与 rope_delta（位置偏移，复用来的 KV 靠它对齐绝对位置）在这里装上：两者一装，这条序列
        // 在数值上的接续关系就定了。
        install_sampling(sequence, request, request_plan.sampling);
        sequence.rope_delta = staged.prompt.rope_delta;
        set_device_i32(io.rope_delta, sequence.rope_delta);

        // 请求侧控制状态清零重来：上一手残留在 RequestControl 里的统计与未结算事务一律不带进来。
        // tail_hidden（末尾 hidden，零后缀复用与检查点靠它）只在整句都复用时才有意义。
        // 物化期攒下的三个数组在这里 swap 进来——它们本来就是物化期的暂存，换手之后暂存即空。
        request.timings              = {};
        request.pending              = {};
        request.publish_continuation = request_plan.summary.publish_continuation;
        sequence.mtp_draft_count     = 0;
        sequence.tail_hidden_valid   = base == prompt_tokens && sequence.tail_hidden_valid;
        sequence.ledger.swap(materialization_ledger_);
        sequence.prefix_identity.swap(materialization_identity_);
        sequence.prefix_digests.swap(materialization_prefix_digests_);
        sequence.rebuild_work       = request_plan.root_rebuild_work;
        sequence.rebuild_tail_begin = request_plan.root_rebuild_tail_begin;

        // DFlash 后端的 ingress 是一小份主机侧描述（活跃 lane、状态源/目的地槽位、后端 KV 的表行），解码
        // 每轮都会重新下发。预填只有一条序列，所以这里只填第 0 行，真正提交给设备是后面推进时的事。
        if (is_masked_draft_backend(speculative_backend)) {
            if (!dflash || !io.dflash_decode || (backend_kv_cache() && !sequence.kv->backend)) {
                throw std::logic_error("DFlash prefill state is incomplete");
            }
            *dflash_host_ingress                       = {};
            dflash_host_ingress->active_lanes[0]       = static_cast<std::int32_t>(sequence.lane);
            const StateImageSelectors selectors        = state_selectors(sequence);
            dflash_host_ingress->state_source_slots[0] = selectors.source;
            dflash_host_ingress->state_destination_slots[0] = selectors.destination;
            dflash_host_ingress->dflash_kv_table_rows[0] =
                sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
            CUDA_CHECK(cudaMemcpyAsync(io.dflash_decode->ingress.data, dflash_host_ingress,
                                       sizeof(qwen3_5::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                       device.stream));
        }

        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        // 全部装完才转 Prefilling —— 这个生命周期就是"这条序列可以被推进"的开关；在中途置上它，外部就
        // 可能看见一条还没装好的序列。
        request.lifecycle = Lifecycle::Prefilling;
    } catch (...) {
        // 失败退场：先把设备上已经发出去的工作同步掉（否则随后释放资源会与在飞的 kernel 打架），再
        // best-effort 清 lane，最后原样抛出。清理只求别崩、别漏，不保证资源全部回池。
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane_best_effort(sequence, request);
        throw;
    }
}

// 两个薄壳入口：把 lane 翻成序列引用，真正的推进与结算在下面。之所以要这层壳，是因为对外的执行入口只
// 拿得到 lane 号（句柄合法性已经在 contract 那一层验过），而内部只谈"哪条序列"。
runtime::PrefillStepResult
ProgramImpl::advance_prefill_raw(std::uint32_t lane, runtime::ExecutionTiming* failed_timing) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    return advance_prefill(active_sequence(lane), requests[lane], failed_timing);
}

// 单行的非投机结算：这一行挂着的必须是 Begin（预填产出的那一个 token），然后接受它。当前树内没有调用
// 者——产品路径一律走由 commit 驱动的批量 resolve_pending_raw，这里保留的是同一件事的单行入口。
runtime::ExecutionTiming ProgramImpl::resolve_prefill_raw(std::uint32_t lane, bool terminal,
                                                          runtime::ExecutionTiming* failed_timing) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (requests[lane].pending.kind != PendingKind::Begin) {
        throw std::logic_error("prefill resolution requires a pending prefill token");
    }
    return resolve_non_speculative_pending(active_sequence(lane), requests[lane], 1, terminal,
                                           std::nullopt, failed_timing);
}

// ============================================================================
// resolve_pending_raw —— 把外部的裁决落进账本（批量的结算点）
//
// 输入是 Runtime 对每一行的裁决（接受几个 token、是否终止、是否取消）与执行切分边界；完成后各行都有新
// 的前沿与去向（Active / Finishable / 已释放）。按"这一批是什么"分三条路：
//   1) 单行且是 Begin：预填那一行。取消 → 严格释放整条；否则接受它唯一那个 token。
//   2) 没有投机后端：逐行独立处理，取消的释放、其余的走 resolve_non_speculative_pending。
//   3) 有投机后端：先批量把设备侧状态重放/回卷到接受范围（replay_fold），再逐行落账。
//
// 硬约束（第 3 条路里逐行复核，其余路径在各自的落地函数里复核）：
//   · 每一行必须仍在它自己记录的出发点上：执行前沿、三个账本数组的长度、三种 KV 推进度，全部要等于
//     pending.base_E / base_S。中途被别人动过就抛——绝不"按现值推断"。
//   · 取消 ⇒ 接受 0 个；非取消 ⇒ 至少接受 1 个；非取消且不终止 ⇒ 必须全收。于是"只收一个前缀"唯一
//     合法的形状是：非取消 + 终止 + 前缀非空。
//   · 部分接受 + 终止时，留存的 hidden 取被接受的那一个（committed - 1），不是产出的最后一个。
//
// 推进分两段，中间隔着一次 device.synchronize()：前段是设备侧动作（状态重放、稀疏统计发布、hidden
// 修正、DFlash 上下文补录），后段是纯主机侧落账。任一段失败 → 整批 lane 走 clear_execution_failure_
// lanes：一行出事整批作废，不允许留下"半批已结算"。
// ============================================================================
runtime::ExecutionTiming ProgramImpl::resolve_pending_raw(
    std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
    std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> cancelled,
    std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (lanes.empty() || lanes.size() > max_concurrency || accepted_tokens.size() != lanes.size() ||
        terminal.size() != lanes.size() || cancelled.size() != lanes.size() ||
        prefix_execution_splits.size() != lanes.size()) {
        throw std::invalid_argument("pending batch resolution has inconsistent membership");
    }

    // ① 预填那一行：Begin 事务只产出一个 token，裁决只有"要"或"取消"两种。
    if (lanes.size() == 1 && lanes.front() < max_concurrency &&
        requests[lanes.front()].pending.kind == PendingKind::Begin) {
        const std::uint32_t lane = lanes.front();
        if (requests[lane].lifecycle != Lifecycle::Pending) {
            throw std::logic_error("prefill pending token no longer matches Program state");
        }
        if (cancelled.front()) {
            if (accepted_tokens.front() != 0 || !terminal.front()) {
                throw std::logic_error("cancelled prefill pending decision is invalid");
            }
            if (!clear_lane_strict(active_sequence(lane), requests[lane])) {
                throw std::logic_error("cancelled prefill lane is not strictly releasable");
            }
        } else {
            timing.pause();
            timing.include(resolve_non_speculative_pending(
                active_sequence(lane), requests[lane], accepted_tokens.front(),
                terminal.front() != 0, prefix_execution_splits.front(), failed_timing));
            timing.resume_post();
        }
        return timing.finish();
    }

    // ② 没有投机后端：每行彼此独立，接受范围就是它自己的产出范围，逐行落地即可。
    if (speculative_backend == SpeculativeBackend::None) {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
                requests[lane].pending.kind != PendingKind::Ordinary) {
                throw std::logic_error("ordinary pending batch no longer matches Program state");
            }
            if (cancelled[row]) {
                if (!clear_lane_strict(active_sequence(lane), requests[lane])) {
                    throw std::logic_error("cancelled decode lane is not strictly releasable");
                }
            } else {
                timing.pause();
                timing.include(resolve_non_speculative_pending(
                    active_sequence(lane), requests[lane], accepted_tokens[row], terminal[row] != 0,
                    prefix_execution_splits[row], failed_timing));
                timing.resume_post();
            }
        }
        return timing.finish();
    }

    if (!replay_fold) {
        throw std::logic_error("speculative pending batch has no ReplaySSM records");
    }

    // ③ 有投机后端：状态必须先按接受范围重放或回卷，一行一次。GdnReplayFoldRow 逐行给出状态槽位与该行
    // 要保留的列数（commit_columns）——取消行是 0，等于把状态整个退回本轮起点。
    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
    std::array<std::int32_t, kMaximumConcurrency> hidden_selectors{};
    bool needs_hidden_correction = false;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
            requests[lane].pending.kind != PendingKind::Speculative) {
            throw std::logic_error("speculative pending batch no longer matches Program state");
        }
        const PendingCandidate& pending = requests[lane].pending;
        const SequenceState& sequence   = active_sequence(lane);
        // "这一行还在出发点上吗"：执行前沿、三个账本数组的长度、三种 KV 推进度，全部要等于本轮开始时
        // 记录的值。少对上一个，就说明有人在中途动过这条序列。
        if (sequence.execution_frontier != pending.base_E ||
            sequence.ledger_frontier != pending.base_S ||
            sequence.ledger.size() != pending.base_S ||
            sequence.prefix_identity.size() != pending.base_S ||
            sequence.prefix_digests.size() != pending.base_S ||
            sequence.text_kv_valid != pending.base_E ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != pending.base_E) ||
            (is_masked_draft_backend(speculative_backend) &&
             sequence.dflash_context_frontier != pending.base_E)) {
            throw std::logic_error("speculative pending row is not at its recorded base");
        }
        // 裁决的形状在这里先验一遍，验过才允许动手改设备状态。
        const std::uint32_t committed = cancelled[row] ? 0U : accepted_tokens[row];
        if ((cancelled[row] && accepted_tokens[row] != 0) ||
            (!cancelled[row] && (committed == 0 || committed > pending.produced ||
                                 (!terminal[row] && committed != pending.produced)))) {
            throw std::logic_error("speculative pending row has an invalid committed prefix");
        }
        const StateImageSelectors selectors = state_selectors(sequence);
        fold_rows[row] =
            ops::GdnReplayFoldRow{.source_state_slot      = selectors.source,
                                  .destination_state_slot = selectors.destination,
                                  .commit_columns         = static_cast<std::int32_t>(committed)};
        // 只有"非取消 + 终止 + 只收前缀"这一种形状需要换 hidden：真正的新前沿是被接受的那一格，留存的
        // hidden 必须跟着换成它，否则下一步或检查点就会拿着一个没被接受的位置的 hidden。
        const bool partial_terminal =
            !cancelled[row] && terminal[row] && committed < pending.produced;
        hidden_selectors[row] =
            static_cast<std::int32_t>(partial_terminal ? committed - 1U : pending.produced - 1U);
        needs_hidden_correction = needs_hidden_correction || partial_terminal;
    }

    const auto tail_started = Clock::now();
    try {
        timing.resume_submit();
        replay_fold->execute(std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()),
                             device.stream);

        // 稀疏接受时统计读的是授权前缀：Frontend 放行了多长，就只把这一段发布进 token_counts——惩罚计数
        // 只该计入真正被采纳的 token，不能把整个草稿窗口都算上。
        if (speculative_backend == SpeculativeBackend::DFlash2) {
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (cancelled[row] || !requests[lanes[row]].sampling_host.token_counts) {
                    continue;
                }
                const auto count = static_cast<std::int32_t>(accepted_tokens[row]);
                Tensor ids =
                    io.dflash_decode->licensed_tokens.slice(1, static_cast<std::int32_t>(row), 1)
                        .slice(0, 0, count)
                        .view({count});
                Tensor counts =
                    token_counts.slice(1, static_cast<std::int32_t>(lanes[row]), 1)
                        .view({dimension(parameters.model.resources().public_token_count)});
                ops::increment_token_counts(ids, counts, device.stream);
            }
        }

        // 需要换 hidden 的行：把选中的那一格隐藏态改成状态槽位上留存的那份（两处 hidden 缓冲的取法按
        // 后端不同，但做的事一样），再按目的地槽位散写回去。
        if (needs_hidden_correction) {
            const auto batch = static_cast<std::int32_t>(lanes.size());
            Tensor selector_tensor;
            Tensor hidden;
            Tensor selected;
            Tensor destinations;
            if (speculative_backend == SpeculativeBackend::Mtp && io.mtp_decode) {
                qwen3_5::MtpDecodeState& frame = *io.mtp_decode;
                selector_tensor                = frame.current_extents.slice(0, 0, batch);
                hidden                         = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.state_destination_slots.slice(0, 0, batch);
            } else if (is_masked_draft_backend(speculative_backend) && io.dflash_decode) {
                qwen3_5::DFlashDecodeState& frame = *io.dflash_decode;
                selector_tensor                   = frame.proposal_extents.slice(0, 0, batch);
                hidden                            = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.state_destination_slots.slice(0, 0, batch);
            } else {
                throw std::logic_error("partial speculative commit has no target frame");
            }
            CUDA_CHECK(cudaMemcpyAsync(selector_tensor.data, hidden_selectors.data(),
                                       lanes.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       device.stream));
            ops::speculative_select_accepted_hidden(hidden, selector_tensor, selected,
                                                    device.stream);
            ops::scatter(selected, destinations, state_images->continuation_hidden_store(),
                         device.stream);
        }

        // DFlash：已经终结的行要把这次接受的那些位置补进它自己的上下文 KV——终结之后不会再有解码轮，
        // 只能在这里补上；没终结的行留在 base 上，等下一轮解码自己往前走。
        if (is_masked_draft_backend(speculative_backend)) {
            std::array<std::uint32_t, kMaximumConcurrency> append_lanes{};
            std::array<std::uint32_t, kMaximumConcurrency> append_starts{};
            std::array<std::uint32_t, kMaximumConcurrency> append_counts{};
            std::size_t append_size = 0;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (!cancelled[row] && terminal[row]) {
                    append_lanes[append_size]  = lanes[row];
                    append_starts[append_size] = requests[lanes[row]].pending.base_E;
                    append_counts[append_size] = accepted_tokens[row];
                    ++append_size;
                }
            }
            if (append_size != 0) {
                enqueue_dflash_context_append(
                    std::span<const std::uint32_t>(append_lanes.data(), append_size),
                    std::span<const std::uint32_t>(append_starts.data(), append_size),
                    std::span<const std::uint32_t>(append_counts.data(), append_size));
            }
        }

        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        work.reset();
    } catch (...) {
        // 设备侧失败：清掉 workspace、整批 lane 一起退场，再抛。已经发出去的 kernel 不试图挽回——同步完
        // 就让整批作废。
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        clear_execution_failure_lanes(lanes);
        throw;
    }

    // 设备侧那一半到此为止，tail_seconds 是它的耗时（最后按行摊进统计）。下面这一段是纯主机侧的落账：
    // device.synchronize() 已经上过，这里读到的都是设备确实写完的事实。
    const double tail_seconds = std::chrono::duration<double>(Clock::now() - tail_started).count();
    const std::uint32_t width = draft_window + 1U;
    try {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = active_sequence(lanes[row]);
            RequestControl& request = requests[lanes[row]];
            if (cancelled[row]) {
                if (!clear_lane_strict(sequence, request)) {
                    throw std::logic_error("cancelled speculative lane is not strictly releasable");
                }
                continue;
            }

            const PendingCandidate pending = request.pending;
            const std::uint32_t committed  = accepted_tokens[row];
            // 落账的前提是"状态与账本对齐"：挂着的 fork 必须先落地，否则账本指向的位置与状态实际写在
            // 哪里会对不上。
            settle_state_fork(sequence);
            const TokenId* token_base =
                speculative_backend == SpeculativeBackend::Mtp
                    ? mtp_host_egress->licensed_tokens.data() + row * width
                    : dflash_host_egress->licensed_tokens.data() + row * width;
            sequence.ledger.insert(sequence.ledger.end(), token_base, token_base + committed);
            commit_generated_prefix_identity(sequence, pending.base_S,
                                             std::span<const TokenId>(token_base, committed),
                                             prefix_execution_splits[row]);
            // 重算账同步推进：这段位置是（部分）重算出来的，重建进度要跟着新的执行前沿走。
            advance_rebuild_work(sequence, pending.base_E + committed, prefill_chunk);
            // 三个前沿一起推进：执行前沿（设备算到哪）、账本前沿（记到哪）、主 KV 推进度。投机轮接受的是
            // **已经算过**的 token（验证过程本身就是目标模型在算），所以账本与执行同步前进——不像非投机
            // 轮那样账本要多出一格"已知但还没算"的 token。
            sequence.execution_frontier = pending.base_E + committed;
            sequence.ledger_frontier    = pending.base_S + committed;
            sequence.text_kv_valid      = sequence.execution_frontier;
            sequence.tail_hidden_valid  = true;

            if (speculative_backend == SpeculativeBackend::Mtp) {
                sequence.mtp_kv_valid = sequence.execution_frontier;
                if (terminal[row]) {
                    sequence.mtp_draft_count = 0;
                } else {
                    const std::int32_t next  = mtp_host_egress->next_extents[row];
                    sequence.mtp_draft_count = static_cast<std::uint32_t>(next);
                    for (std::uint32_t step = 0; step < sequence.mtp_draft_count; ++step) {
                        sequence.mtp_drafts[step] =
                            mtp_host_egress->next_drafts[step * max_concurrency + row];
                    }
                }
            } else {
                sequence.dflash_context_frontier =
                    terminal[row] ? sequence.execution_frontier : pending.base_E;
            }

            // KV 收口：先提交推进到的位置（决定哪些 token 有效），再把越界保留的尾部裁掉。
            commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            if (terminal[row]) {
                request.lifecycle = Lifecycle::Finishable;
            } else {
                request.lifecycle = Lifecycle::Active;
            }
            request.pending = {};
            // 设备侧那半段是按批算的，按行摊进各自的时间统计。
            request.timings.decode_seconds += tail_seconds;
        }
    } catch (...) {
        // 落账阶段失败：已经写好的那几行也不保留——整批走同一条路，不留"半批已结算"。
        clear_execution_failure_lanes(lanes);
        throw;
    }
    return timing.finish();
}

// ============================================================================
// advance_prefill —— 推进一段预填（这个文件的主线）
//
// 三个阶段：可能先给 MTP 打桥（复用的后缀之前要先把 MTP 状态对齐到复用点）、然后按 prefill_chunk 分块
// 把 prompt 算完、最后采样出第一个生成 token 并把这一行挂成未结算事务（PendingKind::Begin）。
//
// 三种推进形态：
//   · 零推进 + 捕获点就在当前 cursor（0-prefill 共享提升）：一个 token 都不算，直接递一张捕获票据返回；
//   · cursor < prompt_tokens：真正跑 chunk，每块推完 cursor 与三种 KV 推进度；碰到捕获前沿就递票据返回
//    （除非这块已经 finalize——最终块的采样还没被结算，票据得等 Begin token 落账之后再由 commit 递）；
//   · cursor == prompt_tokens（零后缀）：不跑 chunk，直接用留存的 tail hidden 采样。
//
// 三个约束：
//   · 同一时刻只允许一张未决的捕获票据（pending_capture_offer）：外部对每个捕获点是**逐个**作决定的，
//     一次捕获的取舍不允许打断其他捕获点的判断。
//   · 切分前沿取"捕获点"与"改写点"里更早的那个：两者都要求 chunk 边界，一个是递票据的位置，一个是
//     分开写状态的位置。
//   · 结束时账本长度必须正好是 prompt_tokens（校验在推入之前），推入 Begin token 后变成 prompt_tokens+1，
//     而执行前沿仍停在 prompt_tokens。
//
// PrefillStepResult.complete 只在"已经采样出 Begin token"时为 true；中途停下的返回都是未完成的一步，
// 调用方得再调一次。
// ============================================================================
runtime::PrefillStepResult ProgramImpl::advance_prefill(SequenceState& sequence,
                                                        RequestControl& request,
                                                        runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
        throw std::logic_error("staged prefill step requires an active concurrent request");
    }

    RequestControl::Prefill& staged = *request.prefill;
    // 上一张捕获票据还没被消化就不能再推进：捕获的取舍是外部逐个作的，这边抢跑到下一个捕获点会把那个
    // 判断顺序打乱。
    if (staged.pending_capture_offer != 0) {
        throw std::logic_error("prefill cannot advance while a capture offer is pending");
    }
    const runtime::BeginSummary summary{.prompt_tokens        = staged.prompt_tokens,
                                        .reused_prompt_tokens = staged.base,
                                        .prefix_reuse_path    = staged.reuse};
    std::uint32_t processed_prompt_tokens = 0;
    const auto started                    = Clock::now();
    try {
        // 零推进的捕获点：cursor 还停在 base 上，而这里恰好是一个捕获候选。一个 token 都不用算——直接把
        // 票据递出去，让外部决定要不要在这里留一份共享前缀（这就是"0-prefill 共享提升"）。递之前先确认
        // 它在形状上确实是共享基础提升，而不是改写点或长锚点——后者本不该在零推进时被递出去。
        if (staged.next_capture < staged.capture_groups.size() &&
            staged.capture_groups[staged.next_capture].frontier == staged.cursor) {
            if (staged.cursor != staged.base ||
                !staged.capture_groups[staged.next_capture].shared ||
                staged.capture_groups[staged.next_capture].rewrite ||
                staged.capture_groups[staged.next_capture].long_anchor) {
                throw std::logic_error("zero-prefill capture is not a shared base promotion");
            }
            // 票据 id 从 1 起（0 表示"没有票据"），回绕时跳过 0。
            if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
            staged.pending_capture_offer = next_capture_offer_id_;
            return runtime::PrefillStepResult{
                .summary = summary,
                .timing  = timing.finish(),
            };
        }
        StateImageSelectors selectors = state_selectors(sequence);
        Tensor rewrite_capture_hidden;
        Tensor* rewrite_capture_hidden_ptr = nullptr;
        // 后面还有捕获点，就先把"hidden 写到哪"准备好：本块最后一个 token 的隐藏态要顺路写进目的地上，
        // 捕获要拿它当检查点的隐藏态。
        if (staged.next_capture < staged.capture_groups.size()) {
            rewrite_capture_hidden = state_images->continuation_hidden_slot(selectors.destination);
            rewrite_capture_hidden_ptr = &rewrite_capture_hidden;
        }
        execution::PrefillContext schedule_state{
            {device, parameters, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
            text_kv_view(sequence),
            mtp_kv_view(sequence),
            decoder->text_kv,
            decoder->mtp_cache(),
            dflash ? &*dflash : nullptr,
            staged.cursor,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            rewrite_capture_hidden_ptr,
            selectors.source,
            selectors.destination,
            staged.initial_mtp_extent,
            dflash_host_ingress};

        // 打桥：把 MTP 状态在**复用点之前**对齐（position 是 base-1 那个位置），对齐之后 mtp_kv_valid 才能
        // 推到 base。视觉 prompt 的桥还要额外从视觉编码里补出那个位置的隐藏态（见 mtp_bridge_multimodal）。
        if (staged.mtp_bridge == MtpBridgeMode::BeforeSuffix) {
            if (staged.cursor != staged.base || staged.base == 0 ||
                staged.cursor >= staged.prompt_tokens) {
                throw std::logic_error("staged MTP bridge is outside the reusable suffix");
            }
            mark_workspace_usage(workspace_plan.mtp_prefill);
            const Tensor& previous_hidden = sequence.tail_hidden;
            const execution::MtpBridgeInput bridge{
                .previous_hidden = &previous_hidden,
                .position        = checked_i32(staged.base - 1, "MTP bridge position"),
                .rope_position   = prompt_rope_position(staged.prompt, staged.base - 1),
            };
            if (staged.vision) {
                execution::mtp_bridge_multimodal(schedule_state, staged.prompt, *staged.vision,
                                                 bridge);
            } else {
                Tensor bridge_token = io.mtp->target_input_ids.slice(0, 0, 1);
                const TokenId token = staged.prompt.token_ids[staged.base];
                CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                           cudaMemcpyHostToDevice, device.stream));
                execution::mtp_bridge_and_propose(schedule_state, bridge_token, previous_hidden,
                                                  bridge.position, bridge.rope_position, false);
            }
            sequence.mtp_kv_valid = staged.base;
            commit_sequence_kv(sequence, sequence.text_kv_valid, sequence.mtp_kv_valid);
            staged.mtp_bridge = MtpBridgeMode::None;
        }

        // 还有 prompt 没算：分块推进，每块最多 prefill_chunk 个 token，直到 prompt 全部算完。
        if (staged.cursor < staged.prompt_tokens) {
            const std::uint32_t nominal =
                std::min(prefill_chunk, staged.prompt_tokens - staged.cursor);
            mark_workspace_usage(staged.prepare_mtp ? workspace_plan.mtp_prefill
                                                    : workspace_plan.text_prefill);
            if (is_masked_draft_backend(speculative_backend)) {
                mark_workspace_usage(workspace_plan.dflash_context);
            }
            std::uint32_t remaining          = nominal;
            std::uint32_t final_chunk_tokens = 0;
            bool finalized                   = false;
            while (remaining != 0) {
                schedule_state.text_kv_base           = staged.cursor;
                selectors                             = state_selectors(sequence);
                schedule_state.state_source_slot      = selectors.source;
                schedule_state.state_destination_slot = selectors.destination;
                if (staged.next_capture < staged.capture_groups.size()) {
                    rewrite_capture_hidden =
                        state_images->continuation_hidden_slot(selectors.destination);
                    schedule_state.rewrite_checkpoint_hidden = &rewrite_capture_hidden;
                } else {
                    schedule_state.rewrite_checkpoint_hidden = nullptr;
                }

                const bool final_candidate = staged.cursor + remaining == staged.prompt_tokens;
                const std::optional<std::uint32_t> capture_frontier =
                    staged.next_capture < staged.capture_groups.size()
                        ? std::optional<std::uint32_t>(
                              staged.capture_groups[staged.next_capture].frontier)
                        : std::nullopt;
                // 切分前沿取两者更早的那个：捕获点要求"跑到这里就得停下来递票据"，改写点要求"状态在这里
                // 分开写"。两者都只能落在 chunk 边界上。
                std::optional<std::uint32_t> split_frontier = capture_frontier;
                const auto rewrite_split                    = std::upper_bound(
                    staged.prompt.identity.rewrite_execution_frontiers.begin(),
                    staged.prompt.identity.rewrite_execution_frontiers.end(), staged.cursor);
                if (rewrite_split != staged.prompt.identity.rewrite_execution_frontiers.end() &&
                    (!split_frontier || *rewrite_split < *split_frontier)) {
                    split_frontier = *rewrite_split;
                }
                execution::PrefillChunkResult result;
                timing.pause();
                if (staged.vision) {
                    if (!workspace_plan.vision) {
                        throw std::logic_error("active Vision prefill lost its workspace plan");
                    }
                    mark_workspace_usage(workspace_plan.vision->capacity_bytes);
                    result = execution::prefill_multimodal_chunk(schedule_state, staged.prompt,
                                                                 *staged.vision, remaining,
                                                                 split_frontier, final_candidate);
                } else {
                    result = execution::prefill_text_chunk(
                        schedule_state, std::span<const TokenId>(staged.prompt.token_ids),
                        remaining, split_frontier, final_candidate);
                }
                timing.include(result.timing);
                timing.resume_post();
                // 实际吃掉的 token 可能少于 nominal（被切分点或视觉块边界提前收尾），调用方必须按它推进
                // 游标；合法区间是 (0, remaining]——0 是没进展，超过 remaining 是越界。
                if (result.processed_tokens == 0 || result.processed_tokens > remaining) {
                    throw std::logic_error("ordinary prefill chunk made invalid progress");
                }
                if (staged.vision) { staged.vision->release_encoded_media_payloads(); }
                staged.cursor += result.processed_tokens;
                processed_prompt_tokens += result.processed_tokens;
                remaining -= result.processed_tokens;
                final_chunk_tokens     = result.processed_tokens;
                // cursor 与三种 KV 推进度一起前进，KV 随即在地址空间上提交一次：这样即使中途停下递票据，
                // 设备上的状态也是自洽的。
                sequence.text_kv_valid = staged.cursor;
                if (staged.prepare_mtp) { sequence.mtp_kv_valid = staged.cursor; }
                if (is_masked_draft_backend(speculative_backend)) {
                    sequence.dflash_context_frontier = staged.cursor;
                }
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));

                // 预填推进到哪，哪里就成为"已经是事实"的位置：如果这是从不可变来源接过来的第一次写，要
                // 在这里先把 fork 合上，之后才谈得上冻结出一份新的改写检查点。
                settle_state_fork(sequence);
                const bool reached_capture = capture_frontier && staged.cursor == *capture_frontier;
                if (reached_capture) {
                    if (result.finalized) {
                        // 块本身已经 finalize（刚采样），票据不能现在递：prompt 前沿上的状态要等 Begin
                        // token 被结算之后才算可发布，那一张票据由 commit 递出去。
                    } else {
                        staged.elapsed_seconds +=
                            std::chrono::duration<double>(Clock::now() - started).count();
                        if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
                        staged.pending_capture_offer = next_capture_offer_id_;
                        return runtime::PrefillStepResult{
                            .summary                 = summary,
                            .processed_prompt_tokens = processed_prompt_tokens,
                            .timing                  = timing.finish(),
                        };
                    }
                }

                finalized = result.finalized;
                if (finalized || remaining == 0) { break; }
            }

            // 没跑完（prompt 还有剩）：返回一个未完成的一步，调用方接着调。
            if (!finalized) {
                if (staged.cursor == staged.prompt_tokens) {
                    throw std::logic_error("staged prefill reached the prompt without sampling");
                }
                staged.elapsed_seconds +=
                    std::chrono::duration<double>(Clock::now() - started).count();
                return runtime::PrefillStepResult{
                    .summary                 = summary,
                    .processed_prompt_tokens = processed_prompt_tokens,
                    .timing                  = timing.finish(),
                };
            }
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("staged prefill sampled before the prompt frontier");
            }
            // 最终块算完并采样之后，把这一块最后一个 token 的 hidden 留下当序列的末尾 hidden——零后缀复用
            // 与检查点都靠它。
            timing.resume_submit();
            copy_tail(sequence, prefill_hidden.slice(
                                    1, static_cast<std::int32_t>(final_chunk_tokens) - 1, 1));
        } else {
            // 零后缀（cursor == prompt_tokens）：prompt 全部复用命中，不跑 chunk，直接拿留存的 tail hidden
            // 走输出头采样。MTP 时还要在命中点之后再打一次桥（AfterExactHit），把 MTP 状态推到 prompt
            // 前沿，顺带按 initial_mtp_extent 决定要不要现在就起草（> 0 才起草）。
            mark_workspace_usage(workspace_plan.ordinary_round);
            if (!sequence.tail_hidden_valid) {
                throw std::logic_error("zero-suffix reuse has no target tail hidden");
            }
            execution::sample_from_hidden(schedule_state, sequence.tail_hidden,
                                          checked_i32(staged.prompt_tokens, "sample position"),
                                          ops::kSamplePurposePrefill);
            set_device_i32(io.rope_pos, checked_i32(staged.prompt_tokens, "rope position") +
                                            sequence.rope_delta);
            if (staged.prepare_mtp) {
                if (staged.mtp_bridge != MtpBridgeMode::AfterExactHit) {
                    throw std::logic_error("zero-suffix MTP reuse has no exact-hit bridge");
                }
                mark_workspace_usage(workspace_plan.mtp_prefill);
                const auto bridge_rope =
                    prompt_rope_position(staged.prompt, staged.prompt_tokens - 1);
                execution::mtp_bridge_and_propose(
                    schedule_state, io.token, sequence.tail_hidden,
                    checked_i32(staged.prompt_tokens - 1, "MTP full-prefix bridge position"),
                    bridge_rope, staged.initial_mtp_extent != 0);
                sequence.mtp_kv_valid = staged.prompt_tokens;
                commit_sequence_kv(sequence, sequence.text_kv_valid, sequence.mtp_kv_valid);
                staged.mtp_bridge = MtpBridgeMode::None;
            }
        }

        copy_round_token();
        std::array<TokenId, qwen3_5::kMtpDecodeMaximumDrafts> initial_drafts{};
        if (staged.prepare_mtp && staged.initial_mtp_extent != 0) {
            CUDA_CHECK(cudaMemcpyAsync(initial_drafts.data(), io.mtp->draft_tokens.data,
                                       staged.initial_mtp_extent * sizeof(TokenId),
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        const double vision_seconds       = staged.vision ? staged.vision->elapsed_seconds() : 0.0;
        const std::uint32_t prompt_tokens = staged.prompt_tokens;

        // 采样结果落账。顺序是刻意排的：先校验 token 落在公开词表内、再确认账本长度正好是 prompt_tokens
        // ——两条都成立才允许推入这个 Begin token，于是账本变成 prompt_tokens + 1 而执行前沿不动。
        // 身份与摘要这里只写"刚采到这枚 token"的**临时**记录：按外部裁决补齐或重写是 commit 阶段的事。
        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (sequence.ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        sequence.ledger.push_back(host_tokens[0]);
        sequence.prefix_identity.append_generated(1, sequence.rope_delta);
        sequence.prefix_digests.append_generated(std::span<const TokenId>(host_tokens, 1),
                                                 sequence.rope_delta);
        sequence.text_kv_valid = prompt_tokens;
        if (staged.prepare_mtp) {
            if (sequence.mtp_kv_valid != prompt_tokens) {
                throw std::logic_error("staged MTP prefill did not reach the prompt frontier");
            }
            sequence.mtp_draft_count = staged.initial_mtp_extent;
            std::copy_n(initial_drafts.begin(), staged.initial_mtp_extent,
                        sequence.mtp_drafts.begin());
        } else if (is_masked_draft_backend(speculative_backend) &&
                   sequence.dflash_context_frontier != prompt_tokens) {
            throw std::logic_error("staged DFlash prefill did not reach the prompt frontier");
        }
        sequence.tail_hidden_valid      = true;
        request.timings.vision_seconds  = vision_seconds;
        request.timings.prefill_seconds = std::max(0.0, staged.elapsed_seconds - vision_seconds);
        staged.prompt.release_all_media_payloads();
        if (staged.vision) { staged.vision->retire_handoff(); }

        // prompt 前沿本身就是一个捕获点时，预填暂存要留着——那张票据还没递出去，得等 commit 用它来递；
        // 否则预填的暂存到此结束。
        const bool prompt_frontier_capture =
            staged.next_capture < staged.capture_groups.size() &&
            staged.capture_groups[staged.next_capture].frontier == prompt_tokens;
        if (!prompt_frontier_capture) { request.prefill.reset(); }
        // 这一行挂成未结算的 Begin 事务：base_E / base_S 都是 0（这是序列的第一步），produced = 1。
        request.pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                             .base_E        = 0,
                                             .base_S        = 0,
                                             .prompt_tokens = prompt_tokens,
                                             .produced      = 1};
        request.lifecycle = Lifecycle::Pending;
        return runtime::PrefillStepResult{
            .summary = summary,
            .round   = runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)},
            .processed_prompt_tokens = processed_prompt_tokens,
            .complete                = true,
            .timing                  = timing.finish(),
        };
    } catch (...) {
        // 失败退场：同样先把在飞的工作同步掉再清 lane（这次只涉及一条 lane），然后原样抛出。
        timing.begin_wait();
        try {
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        const std::uint32_t lane = sequence.lane;
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        throw;
    }
}


} // namespace ninfer::models::qwen3_5::detail
