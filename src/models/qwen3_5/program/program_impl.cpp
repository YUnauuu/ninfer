#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/execution/linear.h"
#include "core/startup.h"
#include "core/device.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// ============================================================================
// ProgramImpl 的"出生、计分、计时、报账"
// 这个 TU 只放四类不依赖具体执行流程的实现：构造 / 析构、离线 CausalScoring、上下文搬运的计时观测、
// 内存统计。解码、预填充、物化、捕获、压力规划都在别的 TU 里（见 program/ 下的子目录），它们共用
// program_impl.h 里的状态定义。看这个文件的意义在于：Program 到底在启动时申请了什么、按什么顺序建立
// 起来，以及对外报的账长什么样。
// ============================================================================

// 契约要求：投机统计要能无异常地移动赋值，否则承载它的容器在搬移时就可能失败。
static_assert(std::is_nothrow_move_assignable_v<SpeculativeStats>);

namespace {

std::uint32_t normalized_private_capacity(const ContextCacheOptions& options);

// 私有续跑容量必须在规划阶段就归一化成"有值且非零"。这里再查一遍属于最后一道闸门：Plan 来自启动期
// 规划，Program 出生后再发现它是默认值就说明上游没算过，此时早失败远好过后面按 0 分配出空账本。
std::uint32_t normalized_private_capacity(const ContextCacheOptions& options) {
    if (!options.max_private_continuations || *options.max_private_continuations == 0) {
        throw std::logic_error("Qwen3.5 context cache private capacity is not normalized");
    }
    return *options.max_private_continuations;
}

} // namespace

// ----------------------------------------------------------------------------
// 构造函数：把启动期定稿的 SequencePlanImpl 落成真实的设备资源
//
// 顺序上分四段：先交叉校验"计划与本次 parameters/features 是同一套"（计划是外部算好后传进来的，
// 这里只做一致性检查，不做规划）；再按 PersistentLayout 一次性切出常驻区、并建立 KV 页存储、地址
// 空间、状态镜像池、Host 侧池子等全部资源；随后建立 round state 与各后端帧并清零设备标量；最后
// （可选）建 CUDA Graph。构造函数结束后，Program 不再申请资源——运行期的一切都在这张预先切好的账上
// 进行，这也是压力规划能靠纯算术判断可行性的前提。
// ----------------------------------------------------------------------------
ProgramImpl::ProgramImpl(const execution::Parameters& parameters_in, const SequencePlanImpl& plan,
                         DeviceContext& device_in, const StartupObserver& startup_observer)
    : parameters(parameters_in), device(device_in), capacity(plan.capacity),
      kv_capacity(plan.kv_capacity), max_concurrency(plan.max_concurrency),
      context_cache(plan.context_cache),
      continuation_capacity(normalized_private_capacity(plan.context_cache)),
      shared_prefix_capacity(plan.context_cache.max_shared_prefixes.value_or(0)),
      prefill_chunk(plan.prefill_chunk), draft_window(plan.draft_window),
      speculative_backend(plan.speculative_backend), kv_storage(plan.kv_storage),
      proposal_head(plan.proposal_head), vision_enabled(plan.features.vision),
      use_cuda_graph(plan.use_cuda_graph), causal_scoring(plan.causal_scoring),
      kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
      continuation_states(continuation_capacity), continuation_slots(continuation_capacity),
      shared_prefix_states(shared_prefix_capacity), shared_prefix_slots(shared_prefix_capacity),
      // 三个钉住内存按"这次跑的是哪种模式"二选一或多选一：计分模式只要 logprob 缓冲，生成模式才要
      // token 回读；解码帧则按后端（普通 / MTP / DFlash）各建一套，互不重叠。
      round_host(plan.causal_scoring ? std::nullopt
                                     : std::make_optional<PinnedHostBuffer>(sizeof(TokenId))),
      score_logprobs_host(plan.causal_scoring ? std::make_optional<PinnedHostBuffer>(
                                                    kCausalScoreTile * sizeof(float))
                                              : std::nullopt),
      ordinary_host(
          !plan.causal_scoring && plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3_5::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::MtpDecodeIngress) +
                                                          sizeof(qwen3_5::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(is_masked_draft_backend(plan.speculative_backend)
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_5::DFlashDecodeIngress) +
                                                             sizeof(qwen3_5::DFlashDecodeEgress))
                      : std::nullopt),
      context_source_ready_(device_in), context_completion_(device_in),
      context_transfer_timers_{CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream)} {
    // 计划里存的是 parameters 的地址而非副本，这里比对地址就确认了"计划是拿这一份 parameters 算出来的"，
    // 再比对 features 是防止同一份 parameters 换了装载选项（视觉/投机等）后被复用。
    if (&parameters != plan.parameters || parameters.model.options() != plan.features) {
        throw std::invalid_argument("Program parameters do not match the frozen sequence plan");
    }
    // 工作区布局也要与 features 对上：一般区不能为空，视觉区与 vision_enabled 同步出现，计分区与
    // causal_scoring 同步出现（计分模式另有 score_hidden 这块常驻中间结果与之配套）。
    if (workspace_plan.general_capacity == 0 ||
        workspace_plan.vision.has_value() != vision_enabled ||
        causal_scoring != plan.persistent.score_hidden.has_value() ||
        causal_scoring != (workspace_plan.causal_score != 0) ||
        (workspace_plan.vision &&
         workspace_plan.vision->general_capacity_bytes != workspace_plan.general_capacity)) {
        throw std::invalid_argument("Qwen3.5 workspace plan does not match startup features");
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    if (!plan.context_cache.max_private_continuations || !plan.context_cache.max_shared_prefixes) {
        throw std::logic_error("Qwen3.5 context cache options are not normalized");
    }
    const std::uint64_t address_capacity64 =
        static_cast<std::uint64_t>(*plan.context_cache.max_private_continuations) +
        *plan.context_cache.max_shared_prefixes;
    if (address_capacity64 == 0 || address_capacity64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 KV address-space capacity exceeds uint32");
    }
    // 地址空间容量比"私有 + 共享"多留一个描述符，给唯一那个进行中的活跃捕获快照用：已发布的私有/共享
    // 地址空间仍以 P + S 为上限，多出来的那一格让"共享目录已满时用一次捕获替换掉其中一项"成为可能
    // ——新快照准备好之前不必先释放旧 checkpoint。
    if (address_capacity64 == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 KV transaction address capacity exceeds uint32");
    }
    const auto address_capacity      = static_cast<std::uint32_t>(address_capacity64 + 1U);
    // 逻辑页容量 = 设备页 + Host 页：Host 侧的 KV 与设备侧的 KV 在**同一套逻辑页编号**里寻址，Host 只是
    // 低层实现不同（页跨距按各自的几何算），所以这里把两边的容量加起来。
    const auto logical_page_capacity = [&](const DeviceKVPagePool& pool) {
        const HostKVPageLayout host_layout = plan_host_kv_page_layout(pool.geometry());
        const std::uint64_t host_pages =
            plan.context_cache.host_kv_capacity_bytes / host_layout.page_stride;
        const std::uint64_t total = static_cast<std::uint64_t>(pool.capacity_pages()) + host_pages;
        if (total > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.5 logical KV page capacity exceeds uint32");
        }
        return static_cast<std::uint32_t>(total);
    };

    // backing 就是上面那块整块预留的常驻区句柄，从这行起把它切成各组件：每个组件只是它的一段视图，
    // 不做二次分配（DecoderState 持有文本 KV 的页池与执行表，随后建立的页存储 / 地址空间都建在同一
    // 批表上）。
    decoder = std::make_unique<qwen3_5::DecoderState>(backing, plan.persistent.decoder);
    text_host_kv_page_stride =
        plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry()).page_stride;
    text_kv_pages = std::make_unique<LogicalKVPageStore>(
        decoder->text_kv.page_pool(), logical_page_capacity(decoder->text_kv.page_pool()));
    text_kv_addresses = std::make_unique<KVAddressSpaceStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(), address_capacity,
        decoder->text_kv.execution_tables().logical_page_capacity());
    // 状态镜像（RNN/GDN 那类需要整份保存恢复的状态）分设备与 Host 两层，但共用一套逻辑编号：
    // state_store 就是这层的统一门面，后面的 fork / 快照 / 降级都通过它寻址，不直接碰两个池子。
    state_images =
        std::make_unique<qwen3_5::StateImageDevicePool>(backing, plan.persistent.state_images);
    if (plan.context_cache.host_state_slots != 0) {
        const std::uint64_t host_state_bytes =
            static_cast<std::uint64_t>(state_images->host_layout().image_bytes) *
            plan.context_cache.host_state_slots;
        StartupPhaseScope host_state_phase(startup_observer, StartupPhase::HostStatePin,
                                           StartupProgressUnit::Bytes, host_state_bytes);
        host_state_images = std::make_unique<qwen3_5::HostStatePool>(
            state_images->host_layout(), plan.context_cache.host_state_slots);
        host_state_phase.complete(host_state_bytes, host_state_bytes);
    }
    const std::uint64_t logical_state_capacity =
        static_cast<std::uint64_t>(state_images->slot_count()) +
        plan.context_cache.host_state_slots;
    if (logical_state_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.5 logical StateImage capacity exceeds uint32");
    }
    state_store = std::make_unique<StateImageStore>(
        *state_images, host_state_images.get(), static_cast<std::uint32_t>(logical_state_capacity));
    // 压力规划的暂存一次性备到位（容量按最坏情况：每个 owner、每个 checkpoint）。压力搜索会被准入
    // 路径反复调用，热路径上不允许再分配内存，所以这里按上界预留下来。
    pressure_private_owner_scratch_.resize(continuation_capacity);
    pressure_shared_owner_scratch_.resize(shared_prefix_capacity);
    pressure_private_drop_scratch_.resize(continuation_capacity);
    const std::size_t pressure_checkpoint_capacity =
        2U + context_cache.max_long_anchors_per_continuation.value_or(0U);
    for (auto& dropped : pressure_private_drop_scratch_) {
        dropped.reserve(pressure_checkpoint_capacity);
    }
    pressure_state_scratch_.reserve(static_cast<std::size_t>(logical_state_capacity));
    // 可选组件都遵循同一条规则：计划里有就必须能建起来，计划里没有就必须留空——任何"半出现"都是计划与
    // 实际能力对不上（replay 记录只服务于投机后端，DFlash 常驻态只服务于 masked draft）。
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
        replay_fold.emplace(*replay_records, state_images->linear().all_layers_view());
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None) ||
        replay_fold.has_value() != replay_records.has_value()) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (plan.persistent.dflash) {
        CyclicKVCache* local = state_images->dflash_local();
        if (local == nullptr) {
            throw std::logic_error("DFlash StateImage has no local fixed state");
        }
        dflash.emplace(backing, *plan.persistent.dflash, *local);
    }
    if (dflash.has_value() != plan.features.masked_draft()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }
    // 后端 KV（MTP / DFlash 的上下文缓存）与文本 KV 走完全平行的另一套页存储 + 地址空间，只是共用同一
    // 份地址空间容量。
    if (qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
        backend_host_kv_page_stride =
            plan_host_kv_page_layout(backend->page_pool().geometry()).page_stride;
        backend_kv_pages = std::make_unique<LogicalKVPageStore>(
            backend->page_pool(), logical_page_capacity(backend->page_pool()));
        backend_kv_addresses = std::make_unique<KVAddressSpaceStore>(
            *backend_kv_pages, backend->execution_tables(), address_capacity,
            backend->execution_tables().logical_page_capacity());
    }
    pressure_text_page_scratch_.resize(text_kv_pages->capacity());
    pressure_text_selected_pages_.reserve(text_kv_pages->capacity());
    if (backend_kv_pages) {
        pressure_backend_page_scratch_.resize(backend_kv_pages->capacity());
        pressure_backend_selected_pages_.reserve(backend_kv_pages->capacity());
    }
    // Host KV 是页锁定内存里的一块池子，按"最坏情况下页跨距最小的那种布局"预留区间容量：布局不同的
    // 后端（文本 / 后端 KV）共用同一个 arena，所以 extent 容量要按最小跨距算才不会超卖。
    if (plan.context_cache.host_kv_capacity_bytes != 0) {
        std::vector<HostKVPageLayout> layouts;
        layouts.push_back(plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry()));
        if (const qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
            HostKVPageLayout backend_layout =
                plan_host_kv_page_layout(backend->page_pool().geometry());
            if (backend_layout != layouts.front()) { layouts.push_back(std::move(backend_layout)); }
        }
        StartupPhaseScope host_kv_phase(
            startup_observer, StartupPhase::HostKvPin, StartupProgressUnit::Bytes,
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes));
        host_kv_arena = std::make_unique<HostKVArena>(
            plan.context_cache.host_kv_capacity_bytes,
            std::span<const HostKVPageLayout>(layouts.data(), layouts.size()));
        host_kv_phase.complete(
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes),
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes));
        std::size_t minimum_stride = layouts.front().page_stride;
        for (const HostKVPageLayout& layout : layouts) {
            minimum_stride = std::min(minimum_stride, layout.page_stride);
        }
        const std::size_t extent_capacity =
            plan.context_cache.host_kv_capacity_bytes / minimum_stride;
        if (extent_capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.5 Host KV extent capacity exceeds uint32");
        }
        if (extent_capacity != 0) {
            host_kv_extents = std::make_unique<HostKVExtentStore>(
                *host_kv_arena, static_cast<std::uint32_t>(extent_capacity));
        }
    }

    // round state 是"一轮计算"的输入输出帧集合。它按后端裁剪：只有被选中的后端对应的帧才该存在，
    // 下面这组检查就是在确认帧的取舍与计划一致——帧错了，图里的地址也就全错了。
    io = qwen3_5::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() !=
        (!causal_scoring && speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != is_masked_draft_backend(speculative_backend)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    // 常驻张量同样是 backing 的视图（bind 只取偏移与形状，不额外分配）。这三个是运行期反复写的设备
    // 标量/配置块，只有在对应模式开启时才存在。
    prefill_hidden = plan.persistent.prefill_hidden.bind(backing);
    if (plan.persistent.score_hidden) {
        score_hidden = plan.persistent.score_hidden->bind(backing);
    }
    if (plan.persistent.token_counts) {
        token_counts = plan.persistent.token_counts->bind(backing);
    }
    if (plan.persistent.sampling_config) {
        sampling_config = plan.persistent.sampling_config->bind(backing);
    }
    // 初始账本：所有 lane 都"没有绑定续跑"（哨兵值而不是 0），代次从 1 起（0 留给"未初始化"），
    // 三个序列数组按容量预留到位——运行期只增删元素、不再扩容，避免热路径上的分配。
    active_continuations.fill(continuation_capacity);
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) { lane_epochs[lane] = 1; }
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        SequenceState& sequence = continuation_states[index];
        // 预留 capacity + 1：逻辑账本稳态下比设备实际执行的前沿多一条（那条还挂在 pending 事务里），
        // 三个"逐 token 一个元素"的数组长度必须始终相等。
        sequence.ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_digests.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.long_anchors.reserve(context_cache.max_long_anchors_per_continuation.value_or(0));
        // 预留量按最坏情况：每份保留的共享续跑来源都可能与所有固定的逐请求候选同时存在。
        sequence.shared_prefix_references.reserve(8U);
    }
    // 物化事务在用的三份暂存账（与上面逐序列同构，但属于"正在落位的那一次"），同样预留满额。
    materialization_ledger_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_identity_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_prefix_digests_.reserve(static_cast<std::size_t>(capacity) + 1ULL);

    // 图里的内核按"表行号"查 KV 表，行号是设备上的标量。起始行固定为 0，之后每次执行前由执行路径改写；
    // 计分模式没有后端 KV，也就没有它那一行。
    set_device_i32(io.text_kv_table_row, 0);
    if (!causal_scoring) { set_device_i32(io.backend_kv_table_row, 0); }

    // 钉住内存的地址在构造时就分好：ingress / egress 是同一块缓冲的前后两段，且必须清零——内核只写
    // 有效字段，其余字段留旧值会让"这一轮有哪些槽被填过"的判断出错。
    host_tokens = round_host ? static_cast<TokenId*>(round_host->data()) : nullptr;
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<qwen3_5::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3_5::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3_5::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3_5::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3_5::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3_5::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3_5::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3_5::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_5::DFlashDecodeIngress));
        *dflash_host_ingress = {};
        *dflash_host_egress  = {};
    }
    // 设备侧标量同样要归零。它们由内核自增/累加（已产出数、位置、rope 偏移、token 数、采样配置），
    // 不是每轮都全量重写，所以初值必须显式给对；这里在默认流上 memset 后同步，晚于它的图才看得到。
    if (io.dflash_prefill) {
        CUDA_CHECK(cudaMemsetAsync(io.dflash_prefill->produced_count.data, 0,
                                   io.dflash_prefill->produced_count.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    if (!causal_scoring) {
        CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
        CUDA_CHECK(
            cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    }
    // 上面的 memset 与下面的建图之间有同步点：图捕获的是实际执行过的内核序列与地址，必须等设备侧
    // 初始化落地。建完图后把工作区与峰值计数复位，让运行期从干净的基线开始记账。
    device.synchronize();
    if (use_cuda_graph) {
        StartupPhaseScope graph_phase(startup_observer, StartupPhase::CudaGraphPrepare);
        prepare_graphs();
        graph_phase.complete();
    }
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

// 析构只负责把两条流同步掉，然后让各池子/arena 自己回收：设备上下文的销毁必须先于任何仍在流上排队
// 的工作结束，否则回收的是别人还在用的内存。真正的资源归还要靠上层先用完 Program（这边不做兜底检查）。
ProgramImpl::~ProgramImpl() noexcept {
    if (device.transfer_stream != nullptr) { (void)cudaStreamSynchronize(device.transfer_stream); }
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
}

// ----------------------------------------------------------------------------
// 离线打分（CausalScoring）：与生成走同一套执行内核，但没有任何"生成"的东西
//
// 它不建图、不采样、不写上下文缓存，也不需要并发——一次调用就是"把一段提示词吃掉，回一批指定位置的
// logprob"。所以它给整段提示词单独要一份状态镜像（reset 出来的初态，不需要从别处物化）和一条独占的
// KV 地址空间，然后按 prefill_chunk 分段推进、每段推进后提交 KV 前沿。
//
// 打分的位置语义：位置 p 的输出预测的是 token p+1，因此"要打分的目标 token 从 first_target 开始"就
// 等价于"要打分的 predictor 从 first_target-1 开始"，产出个数正好是 token_count - first_target。
// 中间结果按一个 kCausalScoreTile 宽的瓦片攒在 score_hidden 里，攒满或收尾时统一做"投影到 logits →
// 取目标 token 的 logprob → 拷回主机"，这样投影与回拷的次数与瓦片数同阶，而不是逐 token 一次。
// ----------------------------------------------------------------------------
std::vector<float> ProgramImpl::causal_score(PreparedPromptData&& prompt,
                                             std::uint32_t first_target) {
    if (!causal_scoring || !score_hidden || !score_logprobs_host ||
        workspace_plan.causal_score == 0) {
        throw std::logic_error("Program was not constructed for causal scoring");
    }
    if (speculative_backend != SpeculativeBackend::None || vision_enabled || use_cuda_graph ||
        context_cache.enabled) {
        throw std::logic_error("causal scoring Program has generation-only startup features");
    }
    const std::size_t token_count_size = prompt.token_ids.size();
    if (token_count_size < 2 || token_count_size > capacity) {
        throw std::invalid_argument("causal score token count must be in [2,capacity]");
    }
    if (first_target == 0 || first_target >= token_count_size) {
        throw std::invalid_argument("causal score first_target is outside the token window");
    }
    if (prompt.has_media()) {
        throw std::invalid_argument("causal scoring accepts text tokens only");
    }

    const auto token_count                     = static_cast<std::uint32_t>(token_count_size);
    const std::uint32_t predictor_count        = token_count - 1U;
    const std::uint32_t scored_predictor_begin = first_target - 1U;
    const std::uint32_t entitlement            = kv_pages_for_frontier(predictor_count);
    if (entitlement == 0) { throw std::logic_error("causal score has no KV entitlement"); }

    // 这次调用用的两份资源：一份状态镜像 + 一条激活的 KV 地址空间。它们是显式申请、显式归还的
    // （不走 RAII），因为成功与异常两条路径都要归还，且"没还干净"必须被当成错误报出去而不是静默忽略。
    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> address;
    const auto cleanup = [&] {
        bool released = true;
        if (address) {
            if (text_kv_addresses->active(*address)) { text_kv_addresses->deactivate(*address); }
            released = text_kv_addresses->release(*address) && released;
            address.reset();
        }
        if (state) {
            released = state_store->release(*state) && released;
            state.reset();
        }
        if (!released) { throw std::logic_error("causal score resources could not be released"); }
    };

    std::vector<float> output;
    output.reserve(token_count_size - first_target);
    std::vector<TokenId> staged_targets;
    staged_targets.reserve(kCausalScoreTile);
    std::uint32_t staged_columns = 0;

    try {
        // 状态用 reset 出来的初态，不从任何已有序列继承；KV 地址空间按整段预测长度一次性预留，随后
        // 逐段 commit_frontier 推进。计分模式只有一条序列，所以它必须落在 Main KV 的第 0 行。
        state = state_store->reserve_reset(device.stream);
        if (!state) { throw std::bad_alloc(); }
        address = text_kv_addresses->create_active(entitlement, 0);
        if (!address) { throw std::bad_alloc(); }
        if (text_kv_addresses->bound_row(*address) != 0) {
            throw std::logic_error("causal score did not bind the unique Main KV row");
        }
        text_kv_addresses->ensure_mapped_to_tokens(*address, predictor_count, device.stream);

        const std::int32_t state_slot = state_store->physical_slot(*state);
        // 冲刷一个瓦片：把攒下的 hidden 用输出头投影成 logits，用 target_logprobs 取出**目标 token 那
        // 一列**的 logprob，再拷回主机追加到结果里。工作区在整个过程里被反复 reset 复用，所以每一步都
        // 显式重置并按这一步的量记峰值。
        const auto flush              = [&] {
            if (staged_columns == 0) { return; }
            if (staged_columns != staged_targets.size() || staged_columns > kCausalScoreTile) {
                throw std::logic_error("causal score staging has an invalid shape");
            }
            work.reset();
            mark_workspace_usage(workspace_plan.causal_score);
            const auto columns = static_cast<std::int32_t>(staged_columns);
            Tensor logits      = work.alloc(
                DType::BF16, {dimension(parameters.model.config().text.vocab_size), columns});
            Tensor target_ids = work.alloc(DType::I32, {columns});
            Tensor logprobs   = work.alloc(DType::FP32, {columns});
            Tensor hidden     = score_hidden->slice(1, 0, columns);
            execution::project(hidden, parameters.text.output_head, logits, work, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(target_ids.data, staged_targets.data(), target_ids.bytes(),
                                                    cudaMemcpyHostToDevice, device.stream));
            ops::target_logprobs(logits, target_ids,
                                              dimension(parameters.model.resources().public_token_count),
                                              logprobs, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(score_logprobs_host->data(), logprobs.data, logprobs.bytes(),
                                                    cudaMemcpyDeviceToHost, device.stream));
            device.synchronize();
            const auto* host = static_cast<const float*>(score_logprobs_host->data());
            output.insert(output.end(), host, host + staged_columns);
            staged_targets.clear();
            staged_columns = 0;
            work.reset();
        };

        // 分段推进：每段最多 prefill_chunk 个 token，处理多少就提交多少 KV 前沿（前沿只前进，不回退），
        // 然后把这批新算出的、落在打分区间内的位置拷进瓦片缓冲。位置 first_target 之前的不算，所以
        // 第一个被选中的位置是 first_target-1。
        std::uint32_t cursor = 0;
        while (cursor < predictor_count) {
            const std::uint32_t nominal = std::min(prefill_chunk, predictor_count - cursor);
            execution::PrefillContext schedule_state{
                {device, parameters, work, state_images->linear(), nullptr, io, prefill_hidden,
                 prefill_chunk, proposal_head},
                decoder->text_kv.execution_view(text_kv_addresses->execution_row(*address)),
                {},
                decoder->text_kv,
                nullptr,
                nullptr,
                cursor,
                nullptr,
                nullptr,
                state_slot,
                state_slot,
                0,
                nullptr};
            mark_workspace_usage(workspace_plan.text_prefill);
            const execution::PrefillChunkResult result = execution::prefill_text_chunk(
                schedule_state, std::span<const TokenId>(prompt.token_ids), nominal, std::nullopt,
                false);
            if (result.finalized || result.processed_tokens == 0 ||
                result.processed_tokens > nominal) {
                throw std::logic_error("causal score Prefill made invalid progress");
            }
            const std::uint32_t chunk_begin = cursor;
            cursor += result.processed_tokens;
            text_kv_addresses->commit_frontier(*address, cursor);

            // 把这一段里要打分的位置搬进 score_hidden（设备到设备），同时把每个位置的目标 token id
            // 记在主机侧，等冲刷时一起送下去。
            std::uint32_t selected = std::max(chunk_begin, scored_predictor_begin);
            while (selected < cursor) {
                const std::uint32_t available = cursor - selected;
                const std::uint32_t room      = kCausalScoreTile - staged_columns;
                const std::uint32_t count     = std::min(available, room);
                Tensor source =
                    prefill_hidden.slice(1, static_cast<std::int32_t>(selected - chunk_begin),
                                         static_cast<std::int32_t>(count));
                Tensor destination = score_hidden->slice(
                    1, static_cast<std::int32_t>(staged_columns), static_cast<std::int32_t>(count));
                CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                           cudaMemcpyDeviceToDevice, device.stream));
                for (std::uint32_t column = 0; column < count; ++column) {
                    staged_targets.push_back(prompt.token_ids[selected + column + 1U]);
                }
                selected += count;
                staged_columns += count;
                if (staged_columns == kCausalScoreTile) { flush(); }
            }
        }
        flush();
        // 个数不对说明前端给的范围与执行推进对不上，宁可报错也不返回一段"看起来对"的打分。
        if (output.size() != token_count_size - first_target) {
            throw std::logic_error("causal score produced the wrong number of logprobs");
        }
        cleanup();
        return output;
    } catch (...) {
        // 异常路径：先同步（让流上残留的工作结束），再尽力归还资源；归还本身再失败就忽略——原始错误更
        // 有诊断价值，且此时已经无法恢复。
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        try {
            cleanup();
        } catch (...) {}
        throw;
    }
}

// 上下文搬运的计时：三个计时器按 ContextResourceClass 分槽（状态 / 主 KV / 后端 KV），下标换算只有
// context_resource_index 一处，顺序绑定在 resources.h 的枚举声明上。事务在搬运的一段前后配对调用
// start / stop，结果由下面的 context_transfer_observation 组装成观测值报给 Runtime。
void ProgramImpl::start_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].start();
}

void ProgramImpl::stop_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].record_stop();
}

// 把计时结果组装成对外观测值。单位随资源类型变：状态镜像按"份"计，KV 按字节计（work.payload_bytes）
// ——量纲不同，报给外部的字段也就跟着不同。elapsed_ns 取 1 的下界，0 留给"压根没测到"。
runtime::ContextTransferObservation ProgramImpl::context_transfer_observation(
    runtime::ContextResourceClass resource, runtime::ContextTransferDirection direction,
    TransferWork work, std::uint32_t page_count, std::uint64_t state_images) const {
    const double elapsed_ns =
        static_cast<double>(
            context_transfer_timers_[context_resource_index(resource)].elapsed_ms()) *
        1'000'000.0;
    const std::uint64_t measured_ns =
        elapsed_ns >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())
            ? std::numeric_limits<std::uint64_t>::max()
            : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(elapsed_ns + 0.5));
    return runtime::ContextTransferObservation{
        .resource  = resource,
        .direction = direction,
        .units =
            resource == runtime::ContextResourceClass::State ? state_images : work.payload_bytes,
        .page_count = page_count,
        .work       = work,
        .elapsed_ns = measured_ns,
    };
}

// 对外内存快照：把各块账合成一份。分成四组——权重（模型自己的，只报容量）、常驻区（persistent，
// 容量/已用/峰值）、工作区（含视觉那一段）、Host 侧两个池子的占用。峰值统一取"实测峰值"与"逻辑峰值"
// 的较大者：有些分配走的是逻辑记账而非实际 alloc，只报其中一个都会偏低。
MemorySummary ProgramImpl::memory_summary() const noexcept {
    MemorySummary out;
    out.device          = device.device;
    out.max_context     = capacity;
    out.kv_capacity     = kv_capacity;
    out.kv_cache        = kv_storage;
    const auto& weights = parameters.model.storage_stats();
    out.weights = ArenaMemorySummary{weights.device_capacity_bytes, weights.device_capacity_bytes,
                                     weights.device_capacity_bytes};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    // 视觉交接缓冲与工作区共用一块区域，但它的起点不一定是工作区开头（handoff_offset_bytes），所以
    // 活跃量要按"偏移 + 当前活跃字节"算；各请求取最大，因为交接是逐请求进行的。
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    std::size_t active_workspace_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        active_workspace_bytes =
            std::max(active_workspace_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), active_workspace_bytes,
                                       std::max(work.peak_used(), workspace_logical_peak_bytes)};
    if (workspace_plan.vision) {
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(capacity, kMaximumPromptVisionTokens)),
            .max_item_tokens        = workspace_plan.vision->max_merged_tokens,
            .general_capacity_bytes = workspace_plan.vision->general_capacity_bytes,
            .encode_peak_bytes      = workspace_plan.vision->encode_peak_bytes,
            .handoff_offset_bytes   = workspace_plan.vision->handoff_offset_bytes,
            .handoff_capacity_bytes = workspace_plan.vision->handoff_capacity_bytes,
            .handoff_active_bytes   = active_handoff_bytes,
            .handoff_peak_bytes     = vision_handoff_peak_bytes,
        };
    }
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    if (host_state_images) {
        out.host_state_capacity_slots = host_state_images->capacity();
        out.host_state_occupied_slots = host_state_images->occupied();
    }
    if (host_kv_arena) {
        out.host_kv_capacity_bytes = host_kv_arena->capacity_bytes();
        out.host_kv_occupied_bytes = host_kv_arena->occupied_bytes();
    }
    return out;
}

// 把峰值基线挪到"现在"：常驻区与工作区各自复位，逻辑峰值与视觉交接峰值以当前活跃量为新起点。
// 交接峰值必须单独重置——它不属于工作区那套记账，一次多模态请求留下的峰值会一直顶高后续的报账。
void ProgramImpl::reset_memory_peaks() noexcept {
    persistent.reset_peak();
    work.reset_peak();
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    vision_handoff_peak_bytes    = active_handoff_bytes;
    workspace_logical_peak_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        workspace_logical_peak_bytes =
            std::max(workspace_logical_peak_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
}


} // namespace ninfer::models::qwen3_5::detail
