#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/context_cache/materialization_planner.h"
#include "runtime/engine/context_cache/shared_capture_planner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// ============================================================================
// context_cache/resource_manager.h —— 上下文缓存的**逻辑账本与策略层**
// ============================================================================
//
// 分工（与 Program 的边界）：
//   * Program 是物理权威，只回答"这么做在设备上成立吗"；
//   * ResourceManager 是逻辑权威，回答"值不值得做、该用谁换谁、做完之后账怎么记"。
// 它从不自己判断可行性，也从不自己碰设备：每一个物理决定都被包成一个对 Program 不透明的资源方案
// （ResourcePlan，封印在某个 resource_revision 上），由 Program 去落地。
//
// 它维护五份账：
//   * lanes_           逻辑 lane 状态机（Free / Materializing / Active / TerminalPending）
//   * catalog_         私有续跑点的目录（谁、什么版本、有哪些 checkpoint、保留策略）
//   * shared_catalog_  共享前缀的目录（同样带版本与迁移中的 pin）
//   * session_index_   会话键 → 续跑点的开放寻址索引（缓存复用的入口）
//   * prefix_index_    前缀摘要 → 可用来源的**短名单**（每次 inspect 前重建，见"排序提示不是证明"）
// 外加 transaction_：当前未完成的那个资源事务（物化或活动捕获），至多一个。
//
// 一次请求的完整走向（四个阶段，缺一不可）：
//   inspect   → 前置检查、重建索引、枚举所有可用来源与候选，交给 planner 挑出一个 Choice
//   reserve   → **先冻结逻辑账本**（来源/牺牲者改 Claimed、发布格改 Reserved*），再让 Program 动手；
//               被拒就把账本回滚回去
//   progress  → 分步推进（可能跨多轮、可被取消），Program 每次回报一个形态
//   adopt     → 拿 Program 的**绝对最终状态**逐条与自己的预登记对齐，一致才落地成新的账
//
// 三条贯穿全文件的纪律：
//   1. **逻辑先冻结，物理后才动**，且冻结必须可逆（rollback_*）；两边的账不会出现"物理已变、逻辑没记"。
//   2. **每个目录项都带 generation（revision）**：跨越规划期的动作要重新比对身份，对不上就是 Stale /
//      nullopt，绝不"尽力而为"地继续。规划期算出的东西（Choice）也只是**意图**，动手前必须再验一次。
//   3. **牺牲清单（OwnerClaim）必须逐条对账**：Program 回报的受害者数量、id 唯一性、disposition 都要
//      与预登记严格对应；对不上是两边账本跑偏，直接抛异常而不是猜测。
//
// 并发：只由单个 worker 线程调用，因此全类无锁；容量在构造时预留（观测、需求窗口等），运行期不再增长。
//
// 模板参数 ModelContract 是刻意的模型无关做法：本层只通过它取到模型侧的类型别名（Program、各类句柄、
// 计划与结果类型），因此它不认识 Qwen3.5 的任何具体类型，也不依赖具体模型。

namespace ninfer::runtime {

inline constexpr std::uint32_t kInvalidCatalogSlot = std::numeric_limits<std::uint32_t>::max();

// 逻辑 lane 的四态。注意它们描述的是**逻辑归属**，不是设备上的忙碌程度：
//   Free            无主，可以作为新请求的目的地
//   Materializing   已被一次进行中的物化占用（物理还没就绪）
//   Active          有活跃请求
//   TerminalPending 该请求已经结束但还没收尾（finish / abort 之前的中转态）
enum class LogicalLaneState : std::uint8_t {
    Free,
    Materializing,
    Active,
    TerminalPending,
};

// 保留策略的观测值：这个条目被"打中"过多少次、最近一次是什么时候（epoch 单调递增）。
// 策略据此决定淘汰顺序——被反复命中的东西更该留下。
struct RetentionObservation {
    RetentionClass retention_class   = RetentionClass::RecentPrivate;
    std::uint64_t selected_hit_count = 0;
    std::uint64_t last_hit_epoch     = 0;
};

// 指向"某份观测"的键。它必须带 revision：观测是绑定到某个具体版本的条目的，条目被释放又重建之后，
// 旧键不能再命中新条目的观测（否则命中统计就会跨身份累积）。
struct PolicyObservationKey {
    bool shared            = false;
    std::uint32_t slot     = kInvalidCatalogSlot;
    std::uint64_t owner_id = 0;
    std::uint64_t revision = 0;
    CheckpointRef checkpoint;

    [[nodiscard]] friend constexpr bool operator==(const PolicyObservationKey&,
                                                   const PolicyObservationKey&) noexcept = default;
};

struct CheckpointObservation {
    CheckpointRef checkpoint;
    RetentionObservation observation;
};

// ResourceManager 只拥有逻辑策略。每一个物理可行性判断与物理变更，都被表示为一个对模型不透明的
// ModelContract::ResourcePlan——它封印在某个 Program::resource_revision() 上，动手前必须重新校验。
template <class ModelContract>
class ResourceManager {
public:
    using Program                 = typename ModelContract::Program;
    using PreparedPrompt          = typename ModelContract::PreparedPrompt;
    using RequestBasePlan         = typename ModelContract::RequestBasePlan;
    using AdmissionCandidate      = typename ModelContract::AdmissionCandidate;
    using ResourcePlan            = typename ModelContract::ResourcePlan;
    using PersistentBackfillProof = typename ModelContract::PersistentBackfillProof;
    using SequenceHandle          = typename ModelContract::SequenceHandle;
    using ContinuationHandle      = typename ModelContract::ContinuationHandle;
    using SharedPrefixHandle      = typename ModelContract::SharedPrefixHandle;
    using CaptureOffer            = typename ModelContract::CaptureOffer;
    using ContinuationSummary     = typename ModelContract::ContinuationSummary;
    using SharedPrefixSummary     = typename ModelContract::SharedPrefixSummary;
    using PrefixShortlistKey =
        std::remove_cvref_t<decltype(std::declval<SharedPrefixSummary>().checkpoint.shortlist_key)>;
    using CaptureAssessment                 = typename ModelContract::CaptureAssessment;
    using ProgramActiveCaptureResult        = typename ModelContract::ActiveCaptureResult;
    using CacheSessionKey                   = typename ModelContract::CacheSessionKey;
    using ProgramContextTransactionProgress = typename ModelContract::ContextTransactionProgress;
    using ProgramMaterializationResult      = typename ModelContract::MaterializationResult;
    using StartResult                       = typename ModelContract::StartResult;
    using FinishResult                      = typename ModelContract::FinishResult;
    using AbortResult                       = typename ModelContract::AbortResult;
    using Planner                           = MaterializationPlanner<ModelContract>;
    using CapturePlanner                    = SharedCapturePlanner<ModelContract>;

private:
    // 事务能力（capability）是某一时刻的结构快照；而活跃引用（active edge）是对 owner 的**持久逻辑
    // 租约**，它刻意不冻结那份快照的 generation：同一个 owner 还活着的时候，另一个读者完全可以改变
    // 它的副本驻留状态。也就是说——租约绑的是"这个 owner 还在"，不是"它的第几版还在"。
    struct ActiveOwnerEdge {
        LogicalOwnerKey owner;
        std::uint32_t slot = kInvalidCatalogSlot;

        [[nodiscard]] friend constexpr bool operator==(ActiveOwnerEdge,
                                                       ActiveOwnerEdge) noexcept = default;
    };

    struct OwnerClaim {
        PlanningOwnerId planning_id;
        CatalogCapability capability;
        VictimDisposition disposition = VictimDisposition::Retained;
        std::vector<CheckpointRef> dropped_checkpoints;
    };

    struct PlanningOwnerRecord {
        PlanningOwnerId id;
        CatalogCapability capability;
    };

public:
    // 复用域：把"这次需求来自哪个复用时域"折叠成一对整数。同一会话（或同一发布批次，在没有会话键时）
    // 属于同一域。用途是给共享前缀加权——**多个互不相干的域都要同一段前缀**，才说明它值得长期留下。
    struct ReuseDomainId {
        std::uint64_t low  = 0;
        std::uint64_t high = 0;

        [[nodiscard]] friend constexpr bool operator==(ReuseDomainId,
                                                       ReuseDomainId) noexcept = default;
    };

    // 需求窗口里的一条记录：某次请求想要哪些前缀、哪些其实已经在缓存里、最后用了哪一个。
    // "想要"与"已有"分开记，是因为两者对保留策略的意义不同（前者是未来价值，后者是已兑现的价值）。
    struct PrefixDemandRecord {
        ReuseDomainId domain;
        std::vector<PrefixShortlistKey> candidate_keys;
        std::vector<PrefixShortlistKey> exact_resident_keys;
        std::optional<PrefixShortlistKey> selected_source_key;
    };

    // 私有目录项的状态机。Claimed / ReservedForActive 都是**事务期**的临时状态：
    //   Vacant             空位
    //   Catalogued         正常可用
    //   Claimed            已被某个进行中的事务预定（可能被牺牲，也可能只是来源）
    //   ReservedForActive  已被预定为"物化完成后给它腾出的发布格"
    // 事务中止时这些临时状态必须被还原——rollback/restore 一条都不能漏。
    enum class CatalogState : std::uint8_t {
        Vacant,
        Catalogued,
        Claimed,
        ReservedForActive,
    };

    // 共享目录的同款状态机，最后一项换成 ReservedCapture：共享前缀只有捕获会写到这一格。
    enum class SharedCatalogState : std::uint8_t {
        Vacant,
        Catalogued,
        Claimed,
        ReservedCapture,
    };

    // 一次规划的结果：去哪个 lane、用哪个方案、用谁当来源、打算牺牲谁、发布到哪一格。
    // 它是一张**意图清单**，不是承诺：只有 ResourceManager 能造出来（构造私有），只能被移动、只能被
    // reserve_materialization 消费一次；而消费前还会再验一遍世界是否还和规划时一致。
    class Choice {
    public:
        Choice(Choice&&) noexcept        = default;
        Choice& operator=(Choice&&)      = delete;
        Choice(const Choice&)            = delete;
        Choice& operator=(const Choice&) = delete;

        [[nodiscard]] const RequestPlanSummary& summary() const noexcept {
            return plan_->summary();
        }

        [[nodiscard]] LaneId destination() const noexcept { return destination_; }

        [[nodiscard]] bool needs_transfer() const noexcept { return plan_->needs_transfer(); }

        [[nodiscard]] ProgramResourceRevision resource_revision() const noexcept {
            return plan_->resource_revision();
        }

    private:
        Choice(LaneId destination, ResourcePlan&& plan, std::uint32_t catalog_capacity,
               std::optional<CacheSessionKey> session, RetentionClass retention,
               bool update_session_index, std::uint64_t publication_order)
            : destination_(destination), plan_(std::move(plan)), session_(std::move(session)),
              retention_(retention), update_session_index_(update_session_index),
              publication_order_(publication_order) {
            private_claims_.reserve(catalog_capacity);
            shared_claims_.reserve(catalog_capacity);
        }

        LaneId destination_{};
        std::optional<ResourcePlan> plan_;
        PrivateSourceMode source_mode_ = PrivateSourceMode::ConsumeToActive;
        std::optional<CatalogCapability> private_source_;
        std::optional<CatalogCapability> shared_source_;
        std::uint32_t publication_slot_ = kInvalidCatalogSlot;
        std::vector<OwnerClaim> private_claims_;
        std::vector<OwnerClaim> shared_claims_;
        std::optional<PolicyObservationKey> selected_observation_;
        std::optional<CacheSessionKey> session_;
        RetentionClass retention_        = RetentionClass::RecentPrivate;
        bool update_session_index_       = true;
        std::uint64_t publication_order_ = 0;
        MaterializationDiagnostics diagnostics_;
        PrefixDemandRecord demand_;

        friend class ResourceManager;
    };

    // "物理已就绪、逻辑尚未确认"那一步的凭据。物化成功之后 lane 停在 Materializing，必须由 adopt() 把它
    // 转为 Active 才算真正生效。之所以不让 Program 的成功直接改逻辑状态，是因为调用方（EngineCore）需要
    // 一个明确的采纳点来决定请求何时算"已在跑"。
    // 它持有 ResourceManager 的反向指针，因此不能跨管理器使用；状态不对时 adopt 直接 terminate。
    class PublishedActivation {
    public:
        PublishedActivation(PublishedActivation&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), result_(std::move(other.result_)),
              destination_(other.destination_) {}

        PublishedActivation& operator=(PublishedActivation&&)      = delete;
        PublishedActivation(const PublishedActivation&)            = delete;
        PublishedActivation& operator=(const PublishedActivation&) = delete;

        [[nodiscard]] const SequenceHandle& sequence() const {
            if (!result_) { throw std::logic_error("published activation is empty"); }
            return result_->sequence;
        }

    private:
        PublishedActivation(ResourceManager& owner, StartResult&& result, LaneId destination)
            : owner_(&owner), result_(std::move(result)), destination_(destination) {}

        ResourceManager* owner_ = nullptr;
        std::optional<StartResult> result_;
        LaneId destination_{};

        friend class ResourceManager;
    };

    struct MaterializationOutcome {
        ContextTransactionStatus status = ContextTransactionStatus::Aborted;
        std::optional<PublishedActivation> activation;
        MaterializationDiagnostics diagnostics;
    };

    // 预约物化的三种结局，语义必须分清：
    //   Reserved  逻辑账已冻结、Program 已开始动手
    //   Stale     **方案过期**（revision 对不上，或 Program 侧物理状态已变）——该重新规划
    //   Aborted   被取消（调用方要求停机/取消），与 Stale 的区别在于是"外部要求"而非"世界变了"
    enum class MaterializationReserveResult : std::uint8_t {
        Reserved,
        Stale,
        Aborted,
    };

    // 捕获只有两种结局：启动成功，或**放弃**（Skipped）。捕获是可选的锦上添花，放弃永远合法，
    // 因此这里没有"失败"这一档。
    enum class ActiveCaptureReserveResult : std::uint8_t {
        Reserved,
        Skipped,
    };

    struct ActiveCaptureOutcome {
        ContextTransactionStatus status = ContextTransactionStatus::Aborted;
    };

    // 事务推进的三种形态，与 Program 侧同名类型一一对应：还在进行中 / 收尾为一次物化 / 收尾为一次捕获。
    using ContextTransactionOutcome =
        std::variant<ContextTransactionInProgress, MaterializationOutcome, ActiveCaptureOutcome>;

    // inspect 的答案：只有 Ready 才带 Choice。其余三档都是"现在不能做"，但**原因不同**——
    // 永久不可行要上报给调度去终止，暂时受阻只是等下一轮。
    struct Inspection {
        Readiness readiness = Readiness::TemporarilyBlocked;
        std::optional<Choice> choice;
    };

    // 构造即定型：lane 数与两个目录容量决定了这层能记多少账。约束（lane 数不超过启动并发、私有目录
    // 不小于 lane 数）在构造时校验；观测与需求窗口的容量也在这里一次性预留，运行期不再分配。
    ResourceManager(std::uint32_t lane_count, std::uint32_t private_catalog_capacity,
                    std::uint32_t shared_catalog_capacity, bool cache_enabled,
                    std::uint32_t max_long_anchors, ContextMachineCostModel cost_model)
        : lane_count_(lane_count), catalog_count_(private_catalog_capacity),
          shared_catalog_count_(shared_catalog_capacity), cache_enabled_(cache_enabled),
          catalog_(private_catalog_capacity), shared_catalog_(shared_catalog_capacity),
          session_index_(private_catalog_capacity),
          prefix_index_(checked_prefix_index_capacity(private_catalog_capacity,
                                                      shared_catalog_capacity, max_long_anchors)),
          max_long_anchors_(max_long_anchors), cost_model_(std::move(cost_model)) {
        if (lane_count == 0 || lane_count > kMaximumConcurrency ||
            private_catalog_capacity < lane_count) {
            throw std::invalid_argument("logical resource-manager bounds are invalid");
        }
        const std::size_t observation_capacity = 3U + max_long_anchors_;
        observation_scratch_.reserve(observation_capacity);
        demand_window_.reserve(kDemandWindowCapacity);
        for (CatalogEntry& entry : catalog_) {
            entry.summary.long_anchors.reserve(max_long_anchors_);
            entry.observations.reserve(observation_capacity);
        }
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            active_[lane].shared_sources.reserve(shared_catalog_capacity);
        }
    }

    // 准入探查：给定一条请求，回答"现在能不能进来、进来的话怎么进"。
    //
    // 顺序上有三层判断，先便宜后昂贵：
    //   1. 事务冲突 → TemporarilyBlocked（本层或 Program 已有未完成事务，谁都别想插队）；
    //   2. **孤立可行性** → PermanentlyInfeasible（不借助任何复用都放不下，那再怎么等也不会变可行，
    //      由 ResourceManager 上报给调度去终止这条请求，而不是反复重试）；
    //   3. 有没有空闲 lane → 没有就 TemporarilyBlocked。
    // 通过之后才重建前缀索引、逐一枚举可用来源（各自成为一个候选），连同"孤立根候选"一起交给 planner，
    // 由后者挑出一个 Choice。方向：这里只负责**枚举**，不负责排序与取舍——取舍在 planner 里。
    [[nodiscard]] Inspection inspect(Program& program, const PreparedPrompt& prompt,
                                     const RequestBasePlan& base, std::uint64_t publication_order,
                                     PlanningAllowance allowance = {}) {
        if (!std::holds_alternative<std::monostate>(transaction_) ||
            program.has_context_transaction()) {
            return {.readiness = Readiness::TemporarilyBlocked};
        }
        if (publication_order == 0) {
            throw std::invalid_argument("request publication order is zero");
        }
        if (!program.isolated_request_feasible(base)) {
            return {.readiness = Readiness::PermanentlyInfeasible};
        }
        std::optional<LaneId> destination;
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            if (lanes_[lane] == LogicalLaneState::Free) {
                destination = LaneId{lane};
                break;
            }
        }
        if (!destination) { return {.readiness = Readiness::TemporarilyBlocked}; }

        const typename Planner::Clock::time_point planning_started = Planner::Clock::now();
        rebuild_prefix_index();
        PrefixDemandRecord provisional_demand;
        provisional_demand.domain =
            reuse_domain(base.context_cache().session_key, publication_order);
        provisional_demand.candidate_keys.reserve(base.context_cache().opportunities.size());
        provisional_demand.exact_resident_keys.reserve(prefix_index_.size());
        if (cache_enabled_) {
            for (const auto& opportunity : base.context_cache().opportunities) {
                if (opportunity.kind != PromptCacheMarkerKind::SharedStablePrefix) { continue; }
                const std::optional<PrefixShortlistKey> key =
                    base.prefix_shortlist_key(opportunity.frontier);
                if (key) { append_unique(provisional_demand.candidate_keys, *key); }
            }
        }
        std::optional<std::size_t> current_session_cell;
        if (cache_enabled_ && base.context_cache().session_key &&
            base.context_cache().update_session_index) {
            current_session_cell = find_session_cell(*base.context_cache().session_key);
        }
        std::vector<Candidate> candidates;
        candidates.reserve(1U + prefix_index_.size());
        std::optional<AdmissionCandidate> root = program.inspect_admission(
            prompt, base, *destination, nullptr, nullptr, std::nullopt, false);
        if (!root) { throw std::logic_error("Program rejected isolated root planning"); }
        candidates.push_back(Candidate{.plan = std::move(*root)});

        if (cache_enabled_) {
            for (const PrefixIndexEntry& index : prefix_index_) {
                if (!valid_prefix_index_entry(index)) { continue; }
                const std::optional<PrefixShortlistKey> incoming =
                    base.prefix_shortlist_key(index.key.frontier);
                if (!incoming || *incoming != index.key) { continue; }

                if (!index.shared) {
                    const CatalogEntry& entry = catalog_[index.slot];
                    if (entry.state != CatalogState::Catalogued || !entry.handle ||
                        private_has_active_edge(index.slot)) {
                        continue;
                    }
                    const bool retain =
                        entry.session && (!base.context_cache().session_key ||
                                          *entry.session != *base.context_cache().session_key ||
                                          !base.context_cache().update_session_index);
                    std::optional<AdmissionCandidate> plan =
                        program.inspect_admission(prompt, base, *destination, &*entry.handle,
                                                  nullptr, index.checkpoint, retain);
                    if (!plan) { continue; }
                    if (plan->summary().reusable_prompt_tokens == 0 ||
                        (retain &&
                         plan->identity_assessment().source_mode != PrivateSourceMode::Retain)) {
                        throw std::logic_error("Program returned an invalid private candidate");
                    }
                    const bool current_session_binding =
                        current_session_cell &&
                        session_index_[*current_session_cell].slot == index.slot &&
                        session_index_[*current_session_cell].owner_id == entry.id &&
                        session_index_[*current_session_cell].revision == entry.revision;
                    append_unique(provisional_demand.exact_resident_keys, index.key);
                    candidates.push_back(Candidate{
                        .plan                    = std::move(*plan),
                        .current_session_binding = current_session_binding,
                        .private_source          = private_capability(index.slot),
                        .selected_observation =
                            PolicyObservationKey{
                                .shared     = false,
                                .slot       = index.slot,
                                .owner_id   = entry.id,
                                .revision   = entry.revision,
                                .checkpoint = index.checkpoint,
                            },
                        .source_key = index.key,
                    });
                    continue;
                }

                const SharedCatalogEntry& entry = shared_catalog_[index.slot];
                if (entry.state != SharedCatalogState::Catalogued || !entry.handle) { continue; }
                std::optional<AdmissionCandidate> plan = program.inspect_admission(
                    prompt, base, *destination, nullptr, &*entry.handle, index.checkpoint, false);
                if (!plan) { continue; }
                if (plan->summary().reusable_prompt_tokens == 0 ||
                    plan->identity_assessment().source_mode != PrivateSourceMode::Retain) {
                    throw std::logic_error("Program returned an invalid shared candidate");
                }
                append_unique(provisional_demand.exact_resident_keys, index.key);
                candidates.push_back(Candidate{
                    .plan          = std::move(*plan),
                    .shared_source = shared_capability(index.slot),
                    .selected_observation =
                        PolicyObservationKey{
                            .shared     = true,
                            .slot       = index.slot,
                            .owner_id   = entry.id,
                            .revision   = entry.revision,
                            .checkpoint = index.checkpoint,
                        },
                    .source_key = index.key,
                });
            }
        }

        std::optional<Choice> selected =
            plan_materialization(program, prompt, base, *destination, candidates, publication_order,
                                 planning_started, provisional_demand, allowance);
        if (!selected) { return {.readiness = Readiness::TemporarilyBlocked}; }
        return {
            .readiness = selected->needs_transfer() ? Readiness::NeedsTransfer : Readiness::Ready,
            .choice    = std::move(selected),
        };
    }

    // 为持续回填取证：先确认这份方案还没过期（否则证明没有意义），再把举证责任交给 Program。
    [[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(Program& program, const RequestBasePlan& blocked_head,
                              const Choice& candidate,
                              std::span<const SequenceHandle> persistent_borrowers) const {
        if (!candidate.plan_ || candidate.resource_revision() != program.resource_revision()) {
            return std::nullopt;
        }
        return program.prove_persistent_backfill(blocked_head, *candidate.plan_,
                                                 persistent_borrowers);
    }

    // 预约并启动一次物化。这里是"逻辑先行"最集中的地方，顺序不能颠倒：
    //   1. 校验（当前不能有事务 / 方案没坏 / **revision 仍相等** / Choice 内部自洽）——
    //      revision 不等直接返回 Stale，一个字节都没动；
    //   2. 把 Choice 变成 MaterializationRecord 存进 transaction_，并**先冻结逻辑账本**
    //      （来源与牺牲者转 Claimed、发布格转 Reserved*）；
    //   3. 才把方案交给 Program 动手；
    //   4. Program 拒绝（Aborted）就把第 2 步的账本回滚，事务清空——所以外部看到的要么全无、要么全有。
    [[nodiscard]] MaterializationReserveResult
    reserve_materialization(Program& program, Choice&& choice, PreparedPrompt&& prompt,
                            CancellationFlagView cancellation) {
        if (!std::holds_alternative<std::monostate>(transaction_) ||
            program.has_context_transaction()) {
            throw std::logic_error("ResourceManager already owns a resource transaction");
        }
        const ProgramResourceRevision resource_revision = program.resource_revision();
        if (!choice.plan_ || resource_revision.value == 0) {
            throw std::logic_error("resource choice is malformed");
        }
        if (choice.plan_->resource_revision() != resource_revision) {
            return MaterializationReserveResult::Stale;
        }
        validate_choice(choice, resource_revision);
        MaterializationRecord record = take_materialization_record(choice);
        transaction_.template emplace<MaterializationRecord>(std::move(record));
        MaterializationRecord& open = std::get<MaterializationRecord>(transaction_);
        reserve_logical_materialization(open);

        const ContextTransactionReserveStatus status = program.start_resource_transaction(
            std::move(*choice.plan_), std::move(prompt), cancellation);
        choice.plan_.reset();
        if (status == ContextTransactionReserveStatus::Aborted) {
            rollback_logical_materialization(open);
            transaction_.template emplace<std::monostate>();
            return cancellation.requested() ? MaterializationReserveResult::Aborted
                                            : MaterializationReserveResult::Stale;
        }
        observe_planner_diagnostics(open.diagnostics);
        return MaterializationReserveResult::Reserved;
    }

    // 当前开着的事务属于哪一类（没有则为 nullopt）。调用方据此决定推进时该走哪条分支。
    [[nodiscard]] std::optional<ContextTransactionKind> context_transaction_kind() const noexcept {
        if (std::holds_alternative<MaterializationRecord>(transaction_)) {
            return ContextTransactionKind::Materialization;
        }
        if (std::holds_alternative<ActiveCaptureRecord>(transaction_)) {
            return ContextTransactionKind::ActiveCapture;
        }
        return std::nullopt;
    }

    // 推进事务：让 Program 往前走一步，然后把它的**终态**吸收进自己的账（adopt_*_progress）。
    // 还在进行中时原样返回，调用方下一轮再来。注意吸收失败即抛异常——那意味着两边的账已经不一致，
    // 继续跑只会错得更远。
    [[nodiscard]] ContextTransactionOutcome
    progress_context_transaction(Program& program, CancellationFlagView cancellation) {
        if (std::holds_alternative<std::monostate>(transaction_) ||
            !program.has_context_transaction()) {
            throw std::logic_error("ResourceManager has no progressable resource transaction");
        }
        ProgramContextTransactionProgress progress =
            program.progress_context_transaction(cancellation);
        return std::visit(
            [&](auto&& result) -> ContextTransactionOutcome {
                using Result = std::decay_t<decltype(result)>;
                if constexpr (std::is_same_v<Result, ContextTransactionInProgress>) {
                    return ContextTransactionInProgress{};
                } else if constexpr (std::is_same_v<Result, ProgramMaterializationResult>) {
                    return adopt_materialization_progress(program, std::move(result));
                } else if constexpr (std::is_same_v<Result, ProgramActiveCaptureResult>) {
                    return adopt_active_capture_progress(program, std::move(result));
                } else {
                    throw std::logic_error("Program returned an unsupported resource operation");
                }
            },
            std::move(progress));
    }

    // 采纳：把 lane 从 Materializing 正式转为 Active，并关掉 Program 侧的事务。
    // 前置条件全部不满足时直接 terminate —— 注意这里**不是抛异常**：走到这一步说明物理资源已经就绪，
    // 逻辑却对不上，属于不可恢复的不变量破裂，任何"优雅处理"都只会掩盖资源泄漏。
    void adopt(Program& program, PublishedActivation&& activation) noexcept {
        if (activation.owner_ != this || !activation.result_ ||
            activation.destination_.value >= lane_count_ ||
            lanes_[activation.destination_.value] != LogicalLaneState::Materializing ||
            !active_[activation.destination_.value].occupied ||
            !std::holds_alternative<MaterializationRecord>(transaction_) ||
            !program.has_context_transaction()) {
            std::terminate();
        }
        lanes_[activation.destination_.value] = LogicalLaneState::Active;
        activation.result_.reset();
        activation.owner_ = nullptr;
        transaction_.template emplace<std::monostate>();
        program.finalize_context_transaction();
    }

    // 处理一次捕获机会（offer 从待提交批次里带出来，只有当本轮真的生成了 Begin 令牌时才存在）。
    //
    // 三条走向，按"便宜且确定"到"昂贵且需要搜索"排列：
    //   1. 已被占用（本层或 Program 有事务）→ skip_capture，明确放弃，不留悬念；
    //   2. 恰好命中一个**已在册的完全相同的共享前缀** → 直接发布私有部分，不必搜索；
    //   3. 否则才进入 scenario 搜索：为一个空槽位、或为若干可被替换的共享条目，逐个评估"值不值得"，
    //      选中者可能还附带一个压力方案（牺牲谁），由 Program 落地。
    // 方向：捕获是**可选收益**，所以任何一步不确定都退化成 Skipped——绝不能因为捕获而威胁到已承诺的请求。
    [[nodiscard]] ActiveCaptureReserveResult
    reserve_active_capture(Program& program, LaneId lane, CaptureOffer&& offer,
                           std::uint32_t blocked_runnable_requests,
                           CancellationFlagView cancellation) {
        require_lane(lane, LogicalLaneState::Active);
        const bool manager_transaction = !std::holds_alternative<std::monostate>(transaction_);
        const bool program_transaction = program.has_context_transaction();
        if (manager_transaction != program_transaction) {
            throw std::logic_error("capture observes inconsistent transaction ownership");
        }
        if (program_transaction) {
            program.skip_capture(std::move(offer));
            return ActiveCaptureReserveResult::Skipped;
        }
        rebuild_prefix_index();

        CaptureAssessment private_baseline =
            program.inspect_capture(offer, nullptr, nullptr, std::nullopt, false);
        std::optional<CheckpointRef> private_replacement;
        if (!private_baseline.private_replacement_candidates.empty()) {
            private_replacement =
                *std::min_element(private_baseline.private_replacement_candidates.begin(),
                                  private_baseline.private_replacement_candidates.end(),
                                  [](CheckpointRef lhs, CheckpointRef rhs) {
                                      return std::tuple{lhs.kind, lhs.frontier, lhs.ordinal} <
                                             std::tuple{rhs.kind, rhs.frontier, rhs.ordinal};
                                  });
            private_baseline =
                program.inspect_capture(offer, nullptr, nullptr, private_replacement, false);
        }

        CaptureAssessment candidate =
            program.inspect_capture(offer, nullptr, nullptr, private_replacement, true);
        const SharedPrefixHandle* exact_shared = nullptr;
        if (candidate.publishes_shared) {
            for (const PrefixIndexEntry& index : prefix_index_) {
                if (!index.shared || !valid_prefix_index_entry(index) ||
                    index.key != candidate.shortlist_key) {
                    continue;
                }
                SharedCatalogEntry& entry = shared_catalog_[index.slot];
                if (program.shared_capture_matches(offer, *entry.handle)) {
                    exact_shared = &*entry.handle;
                    break;
                }
            }
        }
        if (exact_shared != nullptr) {
            if (!private_baseline.publishes_private || !private_baseline.physically_feasible) {
                program.skip_capture(std::move(offer));
                return ActiveCaptureReserveResult::Skipped;
            }
            transaction_.template emplace<ActiveCaptureRecord>(ActiveCaptureRecord{
                .lane              = lane,
                .publishes_private = true,
            });
            const ContextTransactionReserveStatus reserved = program.reserve_active_capture(
                std::move(offer), exact_shared, nullptr, private_replacement, false, cancellation);
            if (reserved == ContextTransactionReserveStatus::Aborted) {
                transaction_.template emplace<std::monostate>();
                return ActiveCaptureReserveResult::Skipped;
            }
            return ActiveCaptureReserveResult::Reserved;
        }

        struct CaptureScenario {
            CaptureAssessment assessment;
            std::uint32_t publication_slot        = kInvalidCatalogSlot;
            const SharedPrefixHandle* replacement = nullptr;
            std::uint64_t replacement_id          = 0;
            std::uint64_t replacement_revision    = 0;
            std::uint32_t stable_ordinal          = 0;
        };

        struct SelectedCapture {
            CaptureScenario scenario;
            typename CapturePlanner::Result plan;
        };

        std::optional<SelectedCapture> selected;
        std::vector<PlanningOwnerRecord> capture_owner_records;
        if (candidate.publishes_shared) {
            const bool pressure_evidence =
                has_shared_candidate_evidence(candidate.shared_evidence,
                                              SharedCandidateEvidence::ExplicitBoundary) ||
                has_shared_candidate_evidence(candidate.shared_evidence,
                                              SharedCandidateEvidence::RequestedAutomatic) ||
                matching_reuse_domains(candidate.shortlist_key) >= 2U;

            std::vector<CaptureScenario> scenarios;
            scenarios.reserve(static_cast<std::size_t>(shared_catalog_count_) + 1U);
            for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
                if (shared_catalog_[slot].state != SharedCatalogState::Vacant) { continue; }
                scenarios.push_back(CaptureScenario{
                    .assessment       = candidate,
                    .publication_slot = slot,
                    .stable_ordinal   = 0,
                });
                break;
            }
            if (pressure_evidence) {
                for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
                    SharedCatalogEntry& entry = shared_catalog_[slot];
                    if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                        entry.transaction_pins != 0 || shared_active_edge_count(slot) != 0) {
                        continue;
                    }
                    CaptureAssessment assessment = program.inspect_capture(
                        offer, nullptr, &*entry.handle, private_replacement, true);
                    if (!assessment.publishes_shared) { continue; }
                    scenarios.push_back(CaptureScenario{
                        .assessment           = std::move(assessment),
                        .publication_slot     = slot,
                        .replacement          = &*entry.handle,
                        .replacement_id       = entry.id,
                        .replacement_revision = entry.revision,
                        .stable_ordinal       = 1U + slot,
                    });
                }
            }

            std::vector<typename CapturePlanner::OwnerPolicy> owner_policies;
            std::vector<typename CapturePlanner::CheckpointPolicy> checkpoint_policies;
            owner_policies.reserve(catalog_count_ + shared_catalog_count_);
            checkpoint_policies.reserve(prefix_index_.size());
            capture_owner_records.reserve(catalog_count_ + shared_catalog_count_);
            const auto append_private_checkpoint = [&](PlanningOwnerId owner, std::uint32_t slot,
                                                       const auto& checkpoint) {
                const CatalogEntry& entry = catalog_[slot];
                checkpoint_policies.push_back(typename CapturePlanner::CheckpointPolicy{
                    .owner                = owner,
                    .checkpoint           = checkpoint.ref,
                    .demand_mask          = committed_demand_mask_for(checkpoint.shortlist_key),
                    .rebuild_ns           = cost_model_.prefill_ns(checkpoint.rebuild_work),
                    .baseline_recovery_ns = price_checkpoint_recovery_work(
                        cost_model_,
                        program.checkpoint_recovery_work(*entry.handle, checkpoint.ref)),
                });
            };
            for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
                const CatalogEntry& entry = catalog_[slot];
                if (entry.state != CatalogState::Catalogued || !entry.handle ||
                    private_has_active_edge(slot)) {
                    continue;
                }
                const PlanningOwnerId owner{
                    .value = static_cast<std::uint32_t>(capture_owner_records.size())};
                capture_owner_records.push_back(PlanningOwnerRecord{
                    .id = owner,
                    .capability =
                        CatalogCapability{
                            .owner =
                                LogicalOwnerKey{
                                    .kind = LogicalOwnerKind::PrivateContinuation,
                                    .id   = entry.id,
                                },
                            .slot       = slot,
                            .generation = entry.revision,
                        },
                });
                owner_policies.push_back(typename CapturePlanner::OwnerPolicy{
                    .owner                    = owner,
                    .private_retention_weight = private_retention_weight(entry.retention),
                });
                if (entry.summary.endpoint) {
                    append_private_checkpoint(owner, slot, *entry.summary.endpoint);
                }
                if (entry.summary.rewrite) {
                    append_private_checkpoint(owner, slot, *entry.summary.rewrite);
                }
                for (const auto& checkpoint : entry.summary.long_anchors) {
                    append_private_checkpoint(owner, slot, checkpoint);
                }
            }
            for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
                const SharedCatalogEntry& entry = shared_catalog_[slot];
                if (entry.state != SharedCatalogState::Catalogued || !entry.handle) { continue; }
                const PlanningOwnerId owner{
                    .value = static_cast<std::uint32_t>(capture_owner_records.size())};
                capture_owner_records.push_back(PlanningOwnerRecord{
                    .id = owner,
                    .capability =
                        CatalogCapability{
                            .owner =
                                LogicalOwnerKey{
                                    .kind = LogicalOwnerKind::SharedPrefix,
                                    .id   = entry.id,
                                },
                            .slot       = slot,
                            .generation = entry.revision,
                        },
                });
                owner_policies.push_back(typename CapturePlanner::OwnerPolicy{
                    .owner                    = owner,
                    .private_retention_weight = 0,
                    .explicit_shared_credit   = entry.explicit_credit,
                });
                checkpoint_policies.push_back(typename CapturePlanner::CheckpointPolicy{
                    .owner      = owner,
                    .checkpoint = entry.summary.checkpoint.ref,
                    .demand_mask =
                        committed_demand_mask_for(entry.summary.checkpoint.shortlist_key),
                    .rebuild_ns = cost_model_.prefill_ns(entry.summary.checkpoint.rebuild_work),
                    .baseline_recovery_ns = price_checkpoint_recovery_work(
                        cost_model_, program.checkpoint_recovery_work(
                                         *entry.handle, entry.summary.checkpoint.ref)),
                });
            }

            const std::uint32_t scenario_budget =
                scenarios.empty()
                    ? 0
                    : std::max<std::uint32_t>(1U, CapturePlanner::kTargetBudget /
                                                      static_cast<std::uint32_t>(scenarios.size()));
            for (CaptureScenario& scenario : scenarios) {
                std::vector<const ContinuationHandle*> private_owners;
                std::vector<PlanningOwnerId> private_owner_ids;
                std::vector<const SharedPrefixHandle*> shared_owners;
                std::vector<PlanningOwnerId> shared_owner_ids;
                const auto owner_id_for = [&](LogicalOwnerKind kind,
                                              std::uint32_t slot) -> PlanningOwnerId {
                    const auto found =
                        std::find_if(capture_owner_records.begin(), capture_owner_records.end(),
                                     [&](const auto& record) {
                                         return record.capability.owner.kind == kind &&
                                                record.capability.slot == slot;
                                     });
                    if (found == capture_owner_records.end()) {
                        throw std::logic_error("capture owner has no planning ID");
                    }
                    return found->id;
                };
                if (pressure_evidence) {
                    private_owners.reserve(catalog_count_);
                    private_owner_ids.reserve(catalog_count_);
                    shared_owners.reserve(shared_catalog_count_);
                    shared_owner_ids.reserve(shared_catalog_count_);
                    for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
                        const CatalogEntry& entry = catalog_[slot];
                        if (entry.state != CatalogState::Catalogued || !entry.handle ||
                            private_has_active_edge(slot)) {
                            continue;
                        }
                        private_owners.push_back(&*entry.handle);
                        private_owner_ids.push_back(
                            owner_id_for(LogicalOwnerKind::PrivateContinuation, slot));
                    }
                    for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
                        const SharedCatalogEntry& entry = shared_catalog_[slot];
                        if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                            entry.transaction_pins != 0 || shared_active_edge_count(slot) != 0 ||
                            entry.id == scenario.replacement_id) {
                            continue;
                        }
                        shared_owners.push_back(&*entry.handle);
                        shared_owner_ids.push_back(
                            owner_id_for(LogicalOwnerKind::SharedPrefix, slot));
                    }
                }
                const typename CapturePlanner::Input input{
                    .capture             = &scenario.assessment,
                    .private_owners      = private_owners,
                    .private_owner_ids   = private_owner_ids,
                    .shared_owners       = shared_owners,
                    .shared_owner_ids    = shared_owner_ids,
                    .owner_policies      = owner_policies,
                    .checkpoint_policies = checkpoint_policies,
                    .direct_shared_victim =
                        scenario.replacement == nullptr
                            ? std::nullopt
                            : std::optional<PlanningOwnerId>(owner_id_for(
                                  LogicalOwnerKind::SharedPrefix, scenario.publication_slot)),
                    .candidate_demand_mask =
                        committed_demand_mask_for(scenario.assessment.shortlist_key),
                    .candidate_rebuild_ns =
                        cost_model_.prefill_ns(scenario.assessment.protected_rebuild_work),
                    .private_baseline_immediate_ns = price_context_transfer_requirements(
                        cost_model_, private_baseline.transfer_requirements),
                    .blocked_runnable_requests = blocked_runnable_requests,
                    .stable_scenario_ordinal   = scenario.stable_ordinal,
                    .target_budget             = scenario_budget,
                };
                std::optional<typename CapturePlanner::Result> planned =
                    capture_planner_.plan(program, cost_model_, input);
                if (!planned) { continue; }
                const bool better =
                    !selected || planned->net_gain > selected->plan.net_gain ||
                    (planned->net_gain == selected->plan.net_gain &&
                     std::tie(planned->stable_scenario_ordinal, planned->stable_target_ordinal) <
                         std::tie(selected->plan.stable_scenario_ordinal,
                                  selected->plan.stable_target_ordinal));
                if (better) {
                    selected.emplace(SelectedCapture{
                        .scenario = std::move(scenario),
                        .plan     = std::move(*planned),
                    });
                }
            }
        }

        if (!selected) {
            if (!private_baseline.publishes_private || !private_baseline.physically_feasible) {
                program.skip_capture(std::move(offer));
                return ActiveCaptureReserveResult::Skipped;
            }
            transaction_.template emplace<ActiveCaptureRecord>(ActiveCaptureRecord{
                .lane              = lane,
                .publishes_private = true,
            });
            const ContextTransactionReserveStatus reserved = program.reserve_active_capture(
                std::move(offer), nullptr, nullptr, private_replacement, false, cancellation);
            if (reserved == ContextTransactionReserveStatus::Aborted) {
                transaction_.template emplace<std::monostate>();
                return ActiveCaptureReserveResult::Skipped;
            }
            return ActiveCaptureReserveResult::Reserved;
        }

        ActiveCaptureRecord record{
            .lane                 = lane,
            .publishes_private    = selected->scenario.assessment.publishes_private,
            .publishes_shared     = true,
            .publication_slot     = selected->scenario.publication_slot,
            .replacement_id       = selected->scenario.replacement_id,
            .replacement_revision = selected->scenario.replacement_revision,
            .shared_evidence      = selected->scenario.assessment.shared_evidence,
        };
        for (const PressureOwnerOutcome& outcome : selected->plan.owner_outcomes) {
            const auto owner_record =
                std::find_if(capture_owner_records.begin(), capture_owner_records.end(),
                             [&](const PlanningOwnerRecord& candidate) {
                                 return candidate.id == outcome.owner;
                             });
            if (owner_record == capture_owner_records.end()) {
                throw std::logic_error("shared capture pressure owner ID is invalid");
            }
            const bool shared =
                owner_record->capability.owner.kind == LogicalOwnerKind::SharedPrefix;
            const std::uint32_t slot = owner_record->capability.slot;
            if (!shared) {
                if (slot >= catalog_count_) {
                    throw std::logic_error("shared capture private pressure owner is invalid");
                }
                const CatalogEntry& entry = catalog_[slot];
                if (entry.state != CatalogState::Catalogued || !entry.handle ||
                    entry.id != owner_record->capability.owner.id ||
                    entry.revision != owner_record->capability.generation ||
                    private_has_active_edge(slot)) {
                    throw std::logic_error("shared capture private pressure owner is stale");
                }
                std::vector<CheckpointRef> dropped = selected_checkpoint_drops(
                    outcome.owner, outcome.disposition, outcome.dropped_checkpoints,
                    selected->plan.checkpoint_outcomes,
                    continuation_checkpoint_count(entry.summary), [&](CheckpointRef checkpoint) {
                        return continuation_contains_checkpoint(entry.summary, checkpoint);
                    });
                record.private_claims.push_back(OwnerClaim{
                    .planning_id         = outcome.owner,
                    .capability          = owner_record->capability,
                    .disposition         = outcome.disposition,
                    .dropped_checkpoints = std::move(dropped),
                });
            } else {
                if (slot >= shared_catalog_count_ || slot == record.publication_slot) {
                    throw std::logic_error("shared capture shared pressure owner is duplicated");
                }
                const SharedCatalogEntry& entry = shared_catalog_[slot];
                if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                    entry.id != owner_record->capability.owner.id ||
                    entry.revision != owner_record->capability.generation ||
                    entry.transaction_pins != 0 || shared_active_edge_count(slot) != 0) {
                    throw std::logic_error("shared capture shared pressure owner is stale");
                }
                std::vector<CheckpointRef> dropped = selected_checkpoint_drops(
                    outcome.owner, outcome.disposition, outcome.dropped_checkpoints,
                    selected->plan.checkpoint_outcomes, 1U, [&](CheckpointRef checkpoint) {
                        return checkpoint == entry.summary.checkpoint.ref;
                    });
                record.shared_claims.push_back(OwnerClaim{
                    .planning_id         = outcome.owner,
                    .capability          = owner_record->capability,
                    .disposition         = outcome.disposition,
                    .dropped_checkpoints = std::move(dropped),
                });
            }
        }
        if (!selected->plan.pressure) {
            throw std::logic_error("selected shared capture has no pressure plan");
        }
        if (record.publication_slot >= shared_catalog_count_) {
            throw std::logic_error("selected shared publication slot is invalid");
        }
        const SharedCatalogEntry& publication = shared_catalog_[record.publication_slot];
        if (record.replacement_id == 0) {
            if (publication.state != SharedCatalogState::Vacant || publication.id != 0 ||
                publication.handle || publication.transaction_pins != 0 ||
                shared_active_edge_count(record.publication_slot) != 0) {
                throw std::logic_error("selected vacant shared publication slot changed");
            }
        } else if (publication.state != SharedCatalogState::Catalogued || !publication.handle ||
                   publication.id != record.replacement_id ||
                   publication.revision != record.replacement_revision ||
                   publication.transaction_pins != 0 ||
                   shared_active_edge_count(record.publication_slot) != 0) {
            throw std::logic_error("selected shared replacement changed before reservation");
        }

        transaction_.template emplace<ActiveCaptureRecord>(std::move(record));
        ActiveCaptureRecord& open = std::get<ActiveCaptureRecord>(transaction_);
        reserve_logical_active_capture(open);
        const ContextTransactionReserveStatus reserved =
            program.reserve_active_capture_with_pressure(
                std::move(offer), nullptr, selected->scenario.replacement, private_replacement,
                true, std::move(*selected->plan.pressure), cancellation);
        if (reserved == ContextTransactionReserveStatus::Aborted) {
            rollback_logical_active_capture(open);
            transaction_.template emplace<std::monostate>();
            return ActiveCaptureReserveResult::Skipped;
        }
        return ActiveCaptureReserveResult::Reserved;
    }

    // 把 lane 标记为"待收尾"：请求已经结束，但资源还没交还。finish / abort 是唯一的出口。
    void mark_terminal_pending(LaneId lane) {
        require_lane(lane, LogicalLaneState::Active);
        lanes_[lane.value] = LogicalLaneState::TerminalPending;
    }

    // 正常结束一条序列，并决定它的续跑点**是否值得留在缓存里**（这是本层而非 Program 的决定）：
    //   * 缓存关闭，或 Program 判定为 Released → 清格、释放引用、lane 回 Free；
    //   * 值得保留 → 把发布格从 ReservedForActive 转成 Catalogued，接管续跑点句柄、摘要、会话与保留策略，
    //     推进 revision，并（按需）更新会话索引。
    // 有一条降级路径值得注意：Program 若报出"无法保留"（返回非 Consumed），本层会退回 abort 语义——
    // 宁可当作异常结束，也不留下一个既没保留又没释放的悬空条目。
    [[nodiscard]] FinishResult finish(Program& program, LaneId lane, SequenceHandle sequence) {
        require_lane(lane, LogicalLaneState::TerminalPending);
        if (!std::holds_alternative<std::monostate>(transaction_) ||
            program.has_context_transaction()) {
            throw std::logic_error("terminal finish overlaps an open resource transaction");
        }
        ActiveEntry& active = active_[lane.value];
        FinishResult result = program.finish(sequence);
        if (result.status != ConsumeStatus::Consumed) {
            AbortResult discarded = program.abort(sequence);
            if (discarded.status != ConsumeStatus::Consumed) {
                throw std::logic_error(
                    "Program could neither retain nor discard terminal sequence");
            }
            release_active_references(lane);
            clear_catalog_entry(catalog_.at(active.publication_slot));
            reset_active_entry(active);
            lanes_[lane.value] = LogicalLaneState::Free;

            FinishResult released;
            released.status      = ConsumeStatus::Consumed;
            released.disposition = FinishDisposition::Released;
            released.timings     = discarded.timings;
            released.speculative = std::move(discarded.speculative);
            return released;
        }
        CatalogEntry& publication = catalog_.at(active.publication_slot);
        if (!cache_enabled_ || result.disposition == FinishDisposition::Released) {
            if (result.disposition != FinishDisposition::Released || result.continuation) {
                throw std::logic_error("released finish returned a continuation");
            }
            release_active_references(lane);
            clear_catalog_entry(publication);
            reset_active_entry(active);
            lanes_[lane.value] = LogicalLaneState::Free;
            return result;
        }
        if (result.disposition != FinishDisposition::Catalogued || !result.continuation ||
            !valid_continuation_summary(result.summary) ||
            publication.state != CatalogState::ReservedForActive ||
            publication.id != active.continuation_id) {
            if (result.continuation) {
                (void)program.release_continuation(std::move(*result.continuation));
                result.continuation.reset();
            }
            throw std::logic_error("Program returned an invalid terminal continuation");
        }

        release_active_references(lane);
        publication.state = CatalogState::Catalogued;
        assign_continuation_summary(publication.summary, result.summary);
        publication.handle.emplace(std::move(*result.continuation));
        result.continuation.reset();
        publication.session   = active.session;
        publication.retention = active.retention;
        migrate_observations(publication, result.summary, active.retention);
        advance_revision(publication.revision);
        if (publication.session && active.update_session_index) {
            if (!publish_session(*publication.session, active.publication_slot, publication.id,
                                 publication.revision, active.publication_order)) {
                publication.session.reset();
                publication.retention = RetentionClass::RecentPrivate;
            }
        }
        reset_active_entry(active);
        lanes_[lane.value] = LogicalLaneState::Free;
        return result;
    }

    // 异常结束：不保留任何东西，一律清格 + 释放引用 + lane 回 Free。Program 若没消费掉序列即抛错。
    [[nodiscard]] AbortResult abort(Program& program, LaneId lane, SequenceHandle sequence) {
        if (!std::holds_alternative<std::monostate>(transaction_) ||
            program.has_context_transaction()) {
            throw std::logic_error("terminal abort overlaps an open resource transaction");
        }
        if (lanes_.at(lane.value) == LogicalLaneState::Active) {
            lanes_[lane.value] = LogicalLaneState::TerminalPending;
        }
        require_lane(lane, LogicalLaneState::TerminalPending);
        AbortResult result = program.abort(sequence);
        if (result.status != ConsumeStatus::Consumed) {
            throw std::logic_error("Program did not consume aborted sequence");
        }
        release_active_references(lane);
        clear_catalog_entry(catalog_.at(active_[lane.value].publication_slot));
        reset_active_entry(active_[lane.value]);
        lanes_[lane.value] = LogicalLaneState::Free;
        return result;
    }

    // 按提交结果推进逻辑 lane：仍是 Active 的什么都不做，被标记 Finishable 的转 TerminalPending，
    // 被取消的立刻释放。行与 lane 必须严格对齐——对不上就是调用方传错了成员表。
    void apply_commit(std::span<const LaneId> lanes,
                      const typename ModelContract::CommitResult& result) {
        if (lanes.size() != result.row_count) {
            throw std::logic_error("commit result membership is not row aligned");
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const LaneId lane = lanes[row];
            require_lane(lane, LogicalLaneState::Active);
            switch (result.rows[row].disposition) {
            case CommitDisposition::Active:
                break;
            case CommitDisposition::Finishable:
                lanes_[lane.value] = LogicalLaneState::TerminalPending;
                break;
            case CommitDisposition::CancelledReleased:
                release_cancelled_lane(lane);
                break;
            }
        }
    }

    // 丢弃待处理事务时，这**整批** lane 都按取消处理（不像 commit 那样逐行区分）。
    // 前提是 Program 确实消费掉了那批成员，否则抛错——丢弃必须是真的丢弃。
    void apply_discard(std::span<const LaneId> lanes,
                       const typename ModelContract::DiscardResult& result) {
        if (lanes.size() != result.row_count || result.status != ConsumeStatus::Consumed) {
            throw std::logic_error("pending discard did not consume its membership");
        }
        for (const LaneId lane : lanes) { release_cancelled_lane(lane); }
    }

    // 提交失败后的兜底释放：尽力把每个仍被占据的 lane 收回，回收过程中的任何异常都被吞掉——
    // 这条路径本身已经处在错误处理里，它唯一的目标是别把资源漏掉，不是报告问题。
    void release_failed_commit(std::span<const LaneId> lanes) noexcept {
        for (const LaneId lane : lanes) {
            if (lane.value < lane_count_ && active_[lane.value].occupied) {
                try {
                    release_cancelled_lane(lane);
                } catch (...) {}
            }
        }
    }

    // 汇总统计：本层记的逻辑账（搬运计数/字节/耗时、压力搜索的各种计数）来自自己的 context_stats_，
    // 而"占了多少"这类物理量直接问 Program 的 physical_usage()。共享引用数在这里现算——它反映的是
    // 当前活跃的租约数，属于逻辑事实。
    void populate_runtime_stats(Program& program, RuntimeStats& out) const noexcept {
        out.state_moves                        = context_stats_.state_moves;
        out.state_forks                        = context_stats_.state_forks;
        out.state_restores                     = context_stats_.state_restores;
        out.state_d2h_count                    = context_stats_.state_d2h_count;
        out.state_h2d_count                    = context_stats_.state_h2d_count;
        out.state_d2d_count                    = context_stats_.state_d2d_count;
        out.state_d2h_bytes                    = context_stats_.state_d2h_bytes;
        out.state_h2d_bytes                    = context_stats_.state_h2d_bytes;
        out.state_d2d_bytes                    = context_stats_.state_d2d_bytes;
        out.state_d2h_seconds                  = context_stats_.state_d2h_seconds;
        out.state_h2d_seconds                  = context_stats_.state_h2d_seconds;
        out.state_d2d_seconds                  = context_stats_.state_d2d_seconds;
        out.main_kv_d2h_pages                  = context_stats_.main_kv_d2h_pages;
        out.main_kv_h2d_pages                  = context_stats_.main_kv_h2d_pages;
        out.main_kv_d2d_pages                  = context_stats_.main_kv_d2d_pages;
        out.main_kv_d2h_bytes                  = context_stats_.main_kv_d2h_bytes;
        out.main_kv_h2d_bytes                  = context_stats_.main_kv_h2d_bytes;
        out.main_kv_d2d_bytes                  = context_stats_.main_kv_d2d_bytes;
        out.main_kv_d2h_seconds                = context_stats_.main_kv_d2h_seconds;
        out.main_kv_h2d_seconds                = context_stats_.main_kv_h2d_seconds;
        out.main_kv_d2d_seconds                = context_stats_.main_kv_d2d_seconds;
        out.backend_kv_d2h_pages               = context_stats_.backend_kv_d2h_pages;
        out.backend_kv_h2d_pages               = context_stats_.backend_kv_h2d_pages;
        out.backend_kv_d2d_pages               = context_stats_.backend_kv_d2d_pages;
        out.backend_kv_d2h_bytes               = context_stats_.backend_kv_d2h_bytes;
        out.backend_kv_h2d_bytes               = context_stats_.backend_kv_h2d_bytes;
        out.backend_kv_d2d_bytes               = context_stats_.backend_kv_d2d_bytes;
        out.backend_kv_d2h_seconds             = context_stats_.backend_kv_d2h_seconds;
        out.backend_kv_h2d_seconds             = context_stats_.backend_kv_h2d_seconds;
        out.backend_kv_d2d_seconds             = context_stats_.backend_kv_d2d_seconds;
        out.pressure_spill_pages               = context_stats_.pressure_spill_pages;
        out.partial_tail_cow_pages             = context_stats_.partial_tail_cow_pages;
        out.pressure_private_owners_degraded   = context_stats_.pressure_private_owners_degraded;
        out.pressure_private_owners_evicted    = context_stats_.pressure_private_owners_evicted;
        out.pressure_shared_owners_degraded    = context_stats_.pressure_shared_owners_degraded;
        out.pressure_shared_owners_evicted     = context_stats_.pressure_shared_owners_evicted;
        out.pressure_checkpoints_dropped       = context_stats_.pressure_checkpoints_dropped;
        out.pressure_searches                  = context_stats_.pressure_searches;
        out.pressure_search_budget_exhaustions = context_stats_.pressure_search_budget_exhaustions;
        out.pressure_maximal_fallback_selections =
            context_stats_.pressure_maximal_fallback_selections;
        out.historical_fork_hits            = context_stats_.historical_fork_hits;
        out.actual_context_transfer_seconds = context_stats_.actual_context_transfer_seconds;

        const auto usage                     = program.physical_usage();
        out.device_state_occupied_slots      = usage.device_state_slots;
        out.host_state_occupied_slots        = usage.host_state_slots;
        out.device_main_kv_occupied_pages    = usage.device_main_kv_pages;
        out.device_backend_kv_occupied_pages = usage.device_backend_kv_pages;
        out.host_kv_occupied_bytes           = usage.host_kv_bytes;
        std::uint64_t shared_references      = 0;
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            if (active_[lane].occupied) {
                shared_references += active_[lane].shared_sources.size();
            }
        }
        out.shared_active_references = shared_references > std::numeric_limits<std::uint32_t>::max()
                                           ? std::numeric_limits<std::uint32_t>::max()
                                           : static_cast<std::uint32_t>(shared_references);
    }

    [[nodiscard]] CatalogState catalog_state(std::uint32_t slot) const noexcept {
        return slot < catalog_count_ ? catalog_[slot].state : CatalogState::Vacant;
    }

    [[nodiscard]] LogicalLaneState lane_state(LaneId lane) const noexcept {
        return lane.value < lane_count_ ? lanes_[lane.value] : LogicalLaneState::Free;
    }

    // Program 做过故障清理（fail_all_cleanup）之后，本层的全部账随之作废：句柄、目录、索引、需求窗口
    // 一律清空，lane 全部回 Free。此后本层不再假设任何既有句柄有效——这是两边重新对齐的唯一方式。
    void clear_after_program_cleanup() noexcept {
        transaction_.template emplace<std::monostate>();
        for (CatalogEntry& entry : catalog_) {
            entry.handle.reset();
            clear_catalog_entry(entry);
        }
        for (SharedCatalogEntry& entry : shared_catalog_) {
            entry.handle.reset();
            clear_shared_entry(entry);
        }
        for (SessionIndexEntry& entry : session_index_) { entry = {}; }
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            lanes_[lane] = LogicalLaneState::Free;
            reset_active_entry(active_[lane]);
        }
        demand_window_.clear();
        demand_epoch_ = 0;
    }

private:
    // ---- 内部记账结构 ----
    //
    // 分两类，职责不要混淆：
    //   * 目录（CatalogEntry / SharedCatalogEntry）是**长期**账，代表真实持有的东西；
    //   * 记录（MaterializationRecord / ActiveCaptureRecord）是**事务期**账，只活在一次事务里，
    //     用来在动手前冻结意图、在结束时逐条对账。
    // 索引（SessionIndexEntry / PrefixIndexEntry）都是可重建的派生视图，不是事实来源。

    // 一个待评估的准入候选：一条附加了"用哪个来源、用哪个 checkpoint"的具体方案。
    // selected_observation 记录"如果选了它，该给谁记一次命中"，因此它也是策略观测的埋点。
    struct Candidate {
        std::optional<AdmissionCandidate> plan;
        bool current_session_binding = false;
        std::optional<CatalogCapability> private_source;
        std::optional<CatalogCapability> shared_source;
        std::optional<PolicyObservationKey> selected_observation;
        std::optional<PrefixShortlistKey> source_key;
    };

    // 私有续跑点的目录项。revision 是它的**身份代次**：每次内容变化就前进一格，于是任何跨越规划期的
    // 引用（CatalogCapability.generation）都能被判定是否过期。summary 是本层对物理状态的逻辑摘要副本，
    // handle 才是那份物理能力的真正句柄。
    struct CatalogEntry {
        CatalogState state     = CatalogState::Vacant;
        std::uint64_t id       = 0;
        std::uint64_t revision = 1;
        ContinuationSummary summary;
        std::optional<ContinuationHandle> handle;
        std::optional<CacheSessionKey> session;
        std::vector<CheckpointObservation> observations;
        RetentionClass retention = RetentionClass::RecentPrivate;
    };

    // 共享前缀条目。与私有条目的差别有三处：只有一个 checkpoint（共享前缀就是一段固定前缀）、
    // 用一份 observation 而不是每个 checkpoint 一份、以及多了 transaction_pins——正在被事务引用的
    // 共享条目不能被当作牺牲者（迁移中的东西不能动）。
    struct SharedCatalogEntry {
        SharedCatalogState state = SharedCatalogState::Vacant;
        std::uint64_t id         = 0;
        std::uint64_t revision   = 1;
        SharedPrefixSummary summary;
        std::optional<SharedPrefixHandle> handle;
        RetentionObservation observation{.retention_class = RetentionClass::SharedStable};
        std::uint32_t transaction_pins    = 0;
        bool explicit_credit              = false;
        std::uint64_t credit_expiry_epoch = 0;
    };

    enum class SessionIndexState : std::uint8_t {
        Empty,
        Occupied,
        Deleted,
    };

    // 会话索引单元：会话键 → 当前代表它的那个续跑点。Deleted 是开放寻址的墓碑（删除后不能直接置空，
    // 否则会截断探测链），因此查找要在 Empty 处停、在 Deleted 处继续。
    struct SessionIndexEntry {
        SessionIndexState state = SessionIndexState::Empty;
        CacheSessionKey key;
        std::uint32_t slot              = kInvalidCatalogSlot;
        std::uint64_t owner_id          = 0;
        std::uint64_t revision          = 0;
        std::uint64_t publication_order = 0;
    };

    // 前缀索引条目：把"前缀摘要"映射到"哪个目录项能提供它"。它是**短名单**，只用来把探查范围缩小到
    // 可能命中的那几个来源；命中与否仍由精确比对（以及 Program 的评估）决定。每次 inspect 前整体重建，
    // 因此它永远是派生视图而非事实来源。
    struct PrefixIndexEntry {
        bool occupied = false;
        bool shared   = false;
        PrefixShortlistKey key;
        std::uint32_t slot     = kInvalidCatalogSlot;
        std::uint64_t owner_id = 0;
        std::uint64_t revision = 0;
        CheckpointRef checkpoint;
    };

    // 一个活跃 lane 的逻辑持有物。retained_private_source / shared_sources 是**租约**（ActiveOwnerEdge），
    // 它们说明"这条活跃请求还压着谁"，从而决定那些来源当前不可被牺牲、也不可被重复选中。
    struct ActiveEntry {
        bool occupied                  = false;
        std::uint32_t publication_slot = kInvalidCatalogSlot;
        std::uint64_t continuation_id  = 0;
        std::optional<CacheSessionKey> session;
        RetentionClass retention        = RetentionClass::RecentPrivate;
        bool update_session_index       = true;
        std::uint64_t publication_order = 0;
        std::optional<ActiveOwnerEdge> retained_private_source;
        std::vector<ActiveOwnerEdge> shared_sources;
    };

    // 进行中的物化事务：Choice 的冻结副本 + 本层在动手前就改好的临时账（Claimed / Reserved）。
    // demand 也挂在这里，等到真正发布成功才提交进需求窗口——没成功过的需求不该影响保留策略。
    struct MaterializationRecord {
        LaneId destination;
        std::optional<CatalogCapability> private_source;
        PrivateSourceMode source_mode = PrivateSourceMode::ConsumeToActive;
        std::optional<CatalogCapability> shared_source;
        std::uint32_t publication_slot = kInvalidCatalogSlot;
        std::vector<OwnerClaim> private_claims;
        std::vector<OwnerClaim> shared_claims;
        std::optional<PolicyObservationKey> selected_observation;
        std::optional<CacheSessionKey> session;
        RetentionClass retention        = RetentionClass::RecentPrivate;
        bool update_session_index       = true;
        std::uint64_t publication_order = 0;
        MaterializationDiagnostics diagnostics;
        PrefixDemandRecord demand;
    };

    // 进行中的捕获事务：记录它要写进哪个共享格、是否替换掉现有条目（replacement_id）、以及它打算
    // 牺牲的私有/共享条目清单。replacement_* 非零就意味着这一格在被替换，收尾时旧条目要么被换掉、
    // 要么原样留下——两种结局都要能还原。
    struct ActiveCaptureRecord {
        LaneId lane;
        bool publishes_private                  = false;
        bool publishes_shared                   = false;
        std::uint32_t publication_slot          = kInvalidCatalogSlot;
        std::uint64_t replacement_id            = 0;
        std::uint64_t replacement_revision      = 0;
        SharedCandidateEvidence shared_evidence = SharedCandidateEvidence::None;
        std::vector<OwnerClaim> private_claims;
        std::vector<OwnerClaim> shared_claims;
    };

    [[nodiscard]] CatalogCapability private_capability(std::uint32_t slot) const {
        const CatalogEntry& entry = catalog_.at(slot);
        return CatalogCapability{
            .owner =
                LogicalOwnerKey{
                    .kind = LogicalOwnerKind::PrivateContinuation,
                    .id   = entry.id,
                },
            .slot       = slot,
            .generation = entry.revision,
        };
    }

    [[nodiscard]] CatalogCapability shared_capability(std::uint32_t slot) const {
        const SharedCatalogEntry& entry = shared_catalog_.at(slot);
        return CatalogCapability{
            .owner =
                LogicalOwnerKey{
                    .kind = LogicalOwnerKind::SharedPrefix,
                    .id   = entry.id,
                },
            .slot       = slot,
            .generation = entry.revision,
        };
    }

    [[nodiscard]] static constexpr ActiveOwnerEdge
    active_edge(CatalogCapability capability) noexcept {
        return ActiveOwnerEdge{.owner = capability.owner, .slot = capability.slot};
    }

    // 有活跃租约的目录项不能被动：既不能当牺牲者，也不能被重复选为来源。下面两个查询是这条规则的
    // 唯一实施点——私有侧最多一条租约所以只问"有没有"，共享侧可能被多条活跃请求同时压着所以数个数。
    [[nodiscard]] bool private_has_active_edge(std::uint32_t slot) const noexcept {
        return std::any_of(active_.begin(), active_.begin() + lane_count_,
                           [&](const ActiveEntry& active) {
                               return active.occupied && active.retained_private_source &&
                                      active.retained_private_source->slot == slot;
                           });
    }

    [[nodiscard]] std::uint32_t shared_active_edge_count(std::uint32_t slot) const noexcept {
        std::uint32_t count = 0;
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            const ActiveEntry& active = active_[lane];
            if (!active.occupied) { continue; }
            count += static_cast<std::uint32_t>(
                std::count_if(active.shared_sources.begin(), active.shared_sources.end(),
                              [&](const ActiveOwnerEdge& edge) { return edge.slot == slot; }));
        }
        return count;
    }

    void reset_active_entry(ActiveEntry& active) noexcept {
        active.occupied         = false;
        active.publication_slot = kInvalidCatalogSlot;
        active.continuation_id  = 0;
        active.session.reset();
        active.retention            = RetentionClass::RecentPrivate;
        active.update_session_index = true;
        active.publication_order    = 0;
        active.retained_private_source.reset();
        active.shared_sources.clear();
    }

    [[nodiscard]] static std::size_t checked_prefix_index_capacity(std::uint32_t private_capacity,
                                                                   std::uint32_t shared_capacity,
                                                                   std::uint32_t max_long_anchors) {
        const std::size_t width = static_cast<std::size_t>(max_long_anchors) + 2U;
        if (private_capacity != 0 &&
            width >
                (std::numeric_limits<std::size_t>::max() - shared_capacity) / private_capacity) {
            throw std::overflow_error("prefix index capacity overflow");
        }
        return static_cast<std::size_t>(private_capacity) * width + shared_capacity;
    }

    // 身份代次只能前进，且**跳过 0**：0 在整套契约里表示"无效/空"，任何有效条目都不该拿到它。
    static void advance_revision(std::uint64_t& revision) noexcept {
        if (++revision == 0) { ++revision; }
    }

    static void saturating_increment(std::uint64_t& value) noexcept {
        if (value != std::numeric_limits<std::uint64_t>::max()) { ++value; }
    }

    static constexpr std::size_t kDemandWindowCapacity = 32U;

    static void append_unique(std::vector<PrefixShortlistKey>& destination,
                              const PrefixShortlistKey& key) {
        if (std::find(destination.begin(), destination.end(), key) == destination.end()) {
            destination.push_back(key);
        }
    }

    [[nodiscard]] static bool demand_matches(const PrefixDemandRecord& demand,
                                             const PrefixShortlistKey& key) noexcept {
        return std::find(demand.candidate_keys.begin(), demand.candidate_keys.end(), key) !=
                   demand.candidate_keys.end() ||
               std::find(demand.exact_resident_keys.begin(), demand.exact_resident_keys.end(),
                         key) != demand.exact_resident_keys.end() ||
               (demand.selected_source_key && *demand.selected_source_key == key);
    }

    [[nodiscard]] static ReuseDomainId reuse_domain(const std::optional<CacheSessionKey>& session,
                                                    std::uint64_t publication_order) noexcept {
        if (!session) {
            return ReuseDomainId{
                .low  = publication_order,
                .high = publication_order ^ 0xD6E8FEB86659FD93ULL,
            };
        }
        std::uint64_t low  = 1469598103934665603ULL;
        std::uint64_t high = 1099511628211ULL ^ 0x9E3779B97F4A7C15ULL;
        for (const unsigned char value : session->view()) {
            low ^= value;
            low *= 1099511628211ULL;
            high ^= static_cast<std::uint64_t>(value) + 0x9E3779B97F4A7C15ULL + (high << 6U) +
                    (high >> 2U);
            high *= 0xD6E8FEB86659FD93ULL;
        }
        return ReuseDomainId{.low = low, .high = high};
    }

    [[nodiscard]] std::uint32_t
    demand_mask_for(const PrefixShortlistKey& key,
                    const PrefixDemandRecord& provisional) const noexcept {
        std::uint32_t mask      = 0;
        std::uint32_t bit       = 0;
        const std::size_t begin = demand_window_.size() == kDemandWindowCapacity ? 1U : 0U;
        for (std::size_t index = begin; index < demand_window_.size(); ++index, ++bit) {
            if (demand_matches(demand_window_[index], key)) { mask |= 1U << bit; }
        }
        if (bit < kDemandWindowCapacity && demand_matches(provisional, key)) { mask |= 1U << bit; }
        return mask;
    }

    [[nodiscard]] std::uint32_t
    committed_demand_mask_for(const PrefixShortlistKey& key) const noexcept {
        std::uint32_t mask = 0;
        for (std::uint32_t bit = 0; bit < demand_window_.size(); ++bit) {
            if (demand_matches(demand_window_[bit], key)) { mask |= 1U << bit; }
        }
        return mask;
    }

    [[nodiscard]] std::size_t matching_reuse_domains(const PrefixShortlistKey& key) const noexcept {
        std::array<ReuseDomainId, kDemandWindowCapacity> domains{};
        std::size_t count = 0;
        for (const PrefixDemandRecord& demand : demand_window_) {
            if (!demand_matches(demand, key) ||
                std::find(domains.begin(), domains.begin() + static_cast<std::ptrdiff_t>(count),
                          demand.domain) != domains.begin() + static_cast<std::ptrdiff_t>(count)) {
                continue;
            }
            domains[count++] = demand.domain;
        }
        return count;
    }

    [[nodiscard]] std::size_t
    matching_reuse_domains(const PrefixShortlistKey& key,
                           const PrefixDemandRecord& provisional) const noexcept {
        std::array<ReuseDomainId, kDemandWindowCapacity> domains{};
        std::size_t count       = 0;
        const std::size_t begin = demand_window_.size() == kDemandWindowCapacity ? 1U : 0U;
        const auto append       = [&](const PrefixDemandRecord& demand) {
            if (!demand_matches(demand, key) ||
                std::find(domains.begin(), domains.begin() + static_cast<std::ptrdiff_t>(count),
                                demand.domain) != domains.begin() + static_cast<std::ptrdiff_t>(count)) {
                return;
            }
            domains[count++] = demand.domain;
        };
        for (std::size_t index = begin; index < demand_window_.size(); ++index) {
            append(demand_window_[index]);
        }
        append(provisional);
        return count;
    }

    // 保留权重：越"贵"的类别越不该被牺牲（LiveSession 最贵、SharedStable 在私有侧为 0 因为它本就属于共享）。
    // 这是交给 planner 的**偏好**，不是硬约束——硬约束来自租约与 Program 的评估。
    [[nodiscard]] static std::uint32_t private_retention_weight(RetentionClass retention) noexcept {
        switch (retention) {
        case RetentionClass::Disposable:
            return 1;
        case RetentionClass::RecentPrivate:
            return 4;
        case RetentionClass::LiveSession:
            return 16;
        case RetentionClass::SharedStable:
            return 0;
        }
        return 0;
    }

    // lane 状态断言：越界、状态不符，或"声称 Active/TerminalPending 但 ActiveEntry 是空的"都算违约。
    void require_lane(LaneId lane, LogicalLaneState expected) const {
        if (lane.value >= lane_count_ || lanes_[lane.value] != expected ||
            ((expected == LogicalLaneState::Active ||
              expected == LogicalLaneState::TerminalPending) &&
             !active_[lane.value].occupied)) {
            throw std::logic_error("logical lane is not in the required state");
        }
    }

    [[nodiscard]] bool valid_checkpoint_summary(const auto& checkpoint,
                                                CheckpointScope scope) const noexcept {
        return checkpoint.ref.frontier != 0 && checkpoint.ref.ordinal == 0 &&
               checkpoint.scope == scope &&
               checkpoint.shortlist_key.frontier == checkpoint.ref.frontier &&
               checkpoint.required_kv.main_pages != 0 && checkpoint.rebuild_work.tokens != 0;
    }

    [[nodiscard]] bool
    valid_continuation_summary(const ContinuationSummary& summary) const noexcept {
        if ((!summary.endpoint && !summary.rewrite && summary.long_anchors.empty()) ||
            summary.long_anchors.size() > max_long_anchors_) {
            return false;
        }
        if (summary.endpoint &&
            (summary.endpoint->ref.kind != CheckpointKind::SessionEndpoint ||
             !valid_checkpoint_summary(*summary.endpoint, CheckpointScope::Private))) {
            return false;
        }
        if (summary.rewrite &&
            (summary.rewrite->ref.kind == CheckpointKind::SessionEndpoint ||
             summary.rewrite->ref.kind == CheckpointKind::SharedStablePrefix ||
             summary.rewrite->ref.kind == CheckpointKind::LongAnchor ||
             !valid_checkpoint_summary(*summary.rewrite, CheckpointScope::Private))) {
            return false;
        }
        for (std::size_t index = 0; index < summary.long_anchors.size(); ++index) {
            const auto& anchor = summary.long_anchors[index];
            if (anchor.ref.kind != CheckpointKind::LongAnchor || anchor.ref.ordinal == 0 ||
                anchor.ref.ordinal > max_long_anchors_ || anchor.ref.frontier == 0 ||
                anchor.scope != CheckpointScope::Private ||
                anchor.shortlist_key.frontier != anchor.ref.frontier ||
                anchor.required_kv.main_pages == 0 || anchor.rebuild_work.tokens == 0) {
                return false;
            }
            for (std::size_t previous = 0; previous < index; ++previous) {
                if (summary.long_anchors[previous].ref.ordinal == anchor.ref.ordinal) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] static bool
    valid_shared_prefix_summary(const SharedPrefixSummary& summary) noexcept {
        const auto& checkpoint = summary.checkpoint;
        return checkpoint.ref.kind == CheckpointKind::SharedStablePrefix &&
               checkpoint.ref.frontier != 0 && checkpoint.ref.ordinal == 0 &&
               checkpoint.scope == CheckpointScope::Shared &&
               checkpoint.shortlist_key.frontier == checkpoint.ref.frontier &&
               checkpoint.required_kv.main_pages != 0 && checkpoint.rebuild_work.tokens != 0;
    }

    static void assign_continuation_summary(ContinuationSummary& destination,
                                            const ContinuationSummary& source) noexcept {
        if (source.long_anchors.size() > destination.long_anchors.capacity()) { std::terminate(); }
        destination.endpoint          = source.endpoint;
        destination.rewrite           = source.rewrite;
        destination.active_references = 0;
        destination.long_anchors.clear();
        for (const auto& anchor : source.long_anchors) {
            destination.long_anchors.push_back(anchor);
        }
    }

    static RetentionObservation* find_observation(std::vector<CheckpointObservation>& observations,
                                                  CheckpointRef checkpoint) noexcept {
        const auto found = std::find_if(
            observations.begin(), observations.end(),
            [&](const CheckpointObservation& value) { return value.checkpoint == checkpoint; });
        return found == observations.end() ? nullptr : &found->observation;
    }

    static const RetentionObservation*
    find_observation(const std::vector<CheckpointObservation>& observations,
                     CheckpointRef checkpoint) noexcept {
        const auto found = std::find_if(
            observations.begin(), observations.end(),
            [&](const CheckpointObservation& value) { return value.checkpoint == checkpoint; });
        return found == observations.end() ? nullptr : &found->observation;
    }

    void migrate_observations(CatalogEntry& entry, const ContinuationSummary& summary,
                              RetentionClass retention) noexcept {
        observation_scratch_.clear();
        const auto append = [&](const auto& checkpoint) {
            if (observation_scratch_.size() == observation_scratch_.capacity()) {
                std::terminate();
            }
            RetentionObservation observation{.retention_class = retention};
            if (const RetentionObservation* old =
                    find_observation(entry.observations, checkpoint.ref)) {
                observation                 = *old;
                observation.retention_class = retention;
            }
            observation_scratch_.push_back(
                CheckpointObservation{.checkpoint = checkpoint.ref, .observation = observation});
        };
        if (summary.endpoint) { append(*summary.endpoint); }
        if (summary.rewrite) { append(*summary.rewrite); }
        for (const auto& anchor : summary.long_anchors) { append(anchor); }
        entry.observations.clear();
        for (const CheckpointObservation& observation : observation_scratch_) {
            entry.observations.push_back(observation);
        }
    }

    void clear_catalog_entry(CatalogEntry& entry) noexcept {
        entry.state = CatalogState::Vacant;
        entry.id    = 0;
        entry.summary.endpoint.reset();
        entry.summary.rewrite.reset();
        entry.summary.long_anchors.clear();
        entry.summary.active_references = 0;
        entry.handle.reset();
        entry.session.reset();
        entry.observations.clear();
        entry.retention = RetentionClass::RecentPrivate;
        advance_revision(entry.revision);
    }

    void clear_shared_entry(SharedCatalogEntry& entry) noexcept {
        entry.state = SharedCatalogState::Vacant;
        entry.id    = 0;
        entry.handle.reset();
        entry.summary     = {};
        entry.observation = RetentionObservation{.retention_class = RetentionClass::SharedStable};
        entry.transaction_pins    = 0;
        entry.explicit_credit     = false;
        entry.credit_expiry_epoch = 0;
        advance_revision(entry.revision);
    }

    // 重建前缀短名单。时机是每次 inspect：规划必须看到**当下**可用的来源，所以索引从不跨轮复用。
    // 容量是构造时算好的固定值（私有条目 × (长锚点上限 + 2) + 共享条目），超了说明容量算式与目录状态
    // 不一致，直接抛——不静默丢弃，否则会悄悄漏掉可用来源。
    void rebuild_prefix_index() {
        for (PrefixIndexEntry& entry : prefix_index_) { entry = {}; }
        std::size_t cursor = 0;
        const auto append  = [&](bool shared, std::uint32_t slot, std::uint64_t owner_id,
                                std::uint64_t revision, const auto& checkpoint) {
            if (cursor >= prefix_index_.size()) {
                throw std::logic_error("prefix index exceeded fixed capacity");
            }
            prefix_index_[cursor++] = PrefixIndexEntry{
                 .occupied   = true,
                 .shared     = shared,
                 .key        = checkpoint.shortlist_key,
                 .slot       = slot,
                 .owner_id   = owner_id,
                 .revision   = revision,
                 .checkpoint = checkpoint.ref,
            };
        };
        for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
            const CatalogEntry& entry = catalog_[slot];
            if (entry.state != CatalogState::Catalogued || !entry.handle) { continue; }
            if (entry.summary.endpoint) {
                append(false, slot, entry.id, entry.revision, *entry.summary.endpoint);
            }
            if (entry.summary.rewrite) {
                append(false, slot, entry.id, entry.revision, *entry.summary.rewrite);
            }
            for (const auto& anchor : entry.summary.long_anchors) {
                append(false, slot, entry.id, entry.revision, anchor);
            }
        }
        for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
            const SharedCatalogEntry& entry = shared_catalog_[slot];
            if (entry.state == SharedCatalogState::Catalogued && entry.handle) {
                append(true, slot, entry.id, entry.revision, entry.summary.checkpoint);
            }
        }
    }

    // 索引项是否仍指向同一个活着的条目：id 与 revision 都要对得上。索引是快照，目录会变，
    // 因此每次使用前都要过这一关——这正是"排序提示不是证明"在索引层的体现。
    [[nodiscard]] bool valid_prefix_index_entry(const PrefixIndexEntry& index) const noexcept {
        if (!index.occupied) { return false; }
        if (!index.shared) {
            if (index.slot >= catalog_count_) { return false; }
            const CatalogEntry& entry = catalog_[index.slot];
            return entry.state == CatalogState::Catalogued && entry.handle &&
                   entry.id == index.owner_id && entry.revision == index.revision;
        }
        if (index.slot >= shared_catalog_count_) { return false; }
        const SharedCatalogEntry& entry = shared_catalog_[index.slot];
        return entry.state == SharedCatalogState::Catalogued && entry.handle &&
               entry.id == index.owner_id && entry.revision == index.revision;
    }

    [[nodiscard]] std::uint64_t newest_hit_epoch(const CatalogEntry& entry) const noexcept {
        std::uint64_t epoch = 0;
        for (const CheckpointObservation& observation : entry.observations) {
            epoch = std::max(epoch, observation.observation.last_hit_epoch);
        }
        return epoch;
    }

    // 挑选"顺带要捕获的共享前缀":一次物化本来就要算 prompt，如果其中某些前沿顺手也能发布成共享前缀,
    // 就是白赚的复用资产。这里逐个机会判断值不值：已经常驻的跳过、本次私有来源自身覆盖的跳过，
    // 剩下的按证据强度与成本决定，最后交给 Program 表态。返回的是前沿列表（最终调度意图）。
    template <class SplitCostFn>
    [[nodiscard]] std::vector<std::uint32_t> select_materialization_shared_captures(
        Program& program, const RequestBasePlan& base, const Candidate& selected_candidate,
        const RequestPlanSummary& selected_summary, const PrefixDemandRecord& provisional_demand,
        SplitCostFn&& split_cost) const {
        struct ProjectedSharedCandidate {
            PrefixShortlistKey key;
            SharedCandidateEvidence evidence = SharedCandidateEvidence::None;
            std::uint32_t frontier           = 0;
            std::uint32_t demand_mask        = 0;
            std::uint64_t rebuild_ns         = 0;
            bool pressure_capable            = false;
        };

        std::vector<ProjectedSharedCandidate> shared_candidates;
        shared_candidates.reserve(base.context_cache().opportunities.size());
        const std::uint32_t vacant_shared_slots = static_cast<std::uint32_t>(
            std::count_if(shared_catalog_.begin(), shared_catalog_.end(), [](const auto& entry) {
                return entry.state == SharedCatalogState::Vacant;
            }));
        for (const auto& opportunity : base.context_cache().opportunities) {
            if (opportunity.kind != PromptCacheMarkerKind::SharedStablePrefix ||
                opportunity.frontier < selected_summary.reusable_prompt_tokens) {
                continue;
            }
            const std::optional<PrefixShortlistKey> key =
                base.prefix_shortlist_key(opportunity.frontier);
            if (!key) { continue; }
            const bool selected_private_base =
                opportunity.frontier == selected_summary.reusable_prompt_tokens &&
                selected_candidate.private_source && selected_candidate.source_key &&
                *selected_candidate.source_key == *key;
            const bool exact_shared_resident = std::any_of(
                prefix_index_.begin(), prefix_index_.end(), [&](const PrefixIndexEntry& entry) {
                    return entry.shared && valid_prefix_index_entry(entry) && entry.key == *key;
                });
            const bool exact_resident =
                std::find(provisional_demand.exact_resident_keys.begin(),
                          provisional_demand.exact_resident_keys.end(),
                          *key) != provisional_demand.exact_resident_keys.end();
            if (exact_shared_resident || (exact_resident && !selected_private_base)) { continue; }
            const bool declared =
                has_shared_candidate_evidence(opportunity.evidence,
                                              SharedCandidateEvidence::ExplicitBoundary) ||
                has_shared_candidate_evidence(opportunity.evidence,
                                              SharedCandidateEvidence::RequestedAutomatic);
            const bool repeated = matching_reuse_domains(*key, provisional_demand) >= 2U;
            const bool surplus_candidate =
                vacant_shared_slots != 0 &&
                (has_shared_candidate_evidence(opportunity.evidence,
                                               SharedCandidateEvidence::DefaultAutomatic) ||
                 has_shared_candidate_evidence(opportunity.evidence,
                                               SharedCandidateEvidence::EngineStructural));
            if (!declared && !repeated && !surplus_candidate) { continue; }
            const std::optional<PrefillWork> rebuild =
                base.shared_candidate_rebuild_work(opportunity.frontier);
            if (!rebuild) {
                throw std::logic_error("prepared shared candidate has no canonical rebuild work");
            }
            shared_candidates.push_back(ProjectedSharedCandidate{
                .key              = *key,
                .evidence         = opportunity.evidence,
                .frontier         = opportunity.frontier,
                .demand_mask      = demand_mask_for(*key, provisional_demand),
                .rebuild_ns       = cost_model_.prefill_ns(*rebuild),
                .pressure_capable = declared || repeated,
            });
        }

        std::vector<ContextPortfolioOwnerPolicy> projected_owners;
        std::vector<ContextPortfolioCheckpointValue> projected_checkpoints;
        std::uint32_t next_projected_owner = 0;
        projected_owners.reserve(catalog_count_ + shared_catalog_count_ + shared_candidates.size());
        projected_checkpoints.reserve(prefix_index_.size() + shared_candidates.size());
        const auto append_existing = [&](PlanningOwnerId owner, const auto& handle,
                                         const auto& checkpoint) {
            const std::uint64_t rebuild  = cost_model_.prefill_ns(checkpoint.rebuild_work);
            const std::uint64_t recovery = price_checkpoint_recovery_work(
                cost_model_, program.checkpoint_recovery_work(handle, checkpoint.ref));
            projected_checkpoints.push_back(ContextPortfolioCheckpointValue{
                .owner       = owner,
                .demand_mask = demand_mask_for(checkpoint.shortlist_key, provisional_demand),
                .rebuild_ns  = rebuild,
                .baseline_recovery_ns = recovery,
                .target_recovery_ns   = recovery,
            });
        };
        for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
            const CatalogEntry& entry = catalog_[slot];
            if (entry.state != CatalogState::Catalogued || !entry.handle ||
                private_has_active_edge(slot) ||
                (selected_candidate.private_source &&
                 slot == selected_candidate.private_source->slot)) {
                continue;
            }
            const PlanningOwnerId owner{.value = next_projected_owner++};
            projected_owners.push_back(ContextPortfolioOwnerPolicy{
                .owner                    = owner,
                .private_retention_weight = private_retention_weight(entry.retention),
            });
            if (entry.summary.endpoint) {
                append_existing(owner, *entry.handle, *entry.summary.endpoint);
            }
            if (entry.summary.rewrite) {
                append_existing(owner, *entry.handle, *entry.summary.rewrite);
            }
            for (const auto& checkpoint : entry.summary.long_anchors) {
                append_existing(owner, *entry.handle, checkpoint);
            }
        }
        for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
            const SharedCatalogEntry& entry = shared_catalog_[slot];
            if (entry.state != SharedCatalogState::Catalogued || !entry.handle) { continue; }
            const PlanningOwnerId owner{.value = next_projected_owner++};
            projected_owners.push_back(ContextPortfolioOwnerPolicy{
                .owner                  = owner,
                .explicit_shared_credit = entry.explicit_credit,
            });
            append_existing(owner, *entry.handle, entry.summary.checkpoint);
        }

        std::vector<std::uint32_t> selected_frontiers;
        std::uint64_t selected_gain = 0;
        ContextPortfolioValue projected_value;
        if (shared_candidates.size() > 7U) {
            throw std::logic_error("prepared shared candidates exceeded the fixed subset bound");
        }
        const std::uint32_t subset_count = 1U << shared_candidates.size();
        for (std::uint32_t mask = 1; mask < subset_count; ++mask) {
            const std::uint32_t selected_count = std::popcount(mask);
            if (selected_count > shared_catalog_count_) { continue; }
            std::uint32_t surplus_only_count = 0;
            std::vector<std::uint32_t> frontiers;
            frontiers.reserve(selected_count);
            std::vector<ContextPortfolioOwnerPolicy> owners          = projected_owners;
            std::vector<ContextPortfolioCheckpointValue> checkpoints = projected_checkpoints;
            for (std::size_t index = 0; index < shared_candidates.size(); ++index) {
                if ((mask & (1U << index)) == 0) { continue; }
                const ProjectedSharedCandidate& candidate = shared_candidates[index];
                if (!candidate.pressure_capable) { ++surplus_only_count; }
                frontiers.push_back(candidate.frontier);
                const PlanningOwnerId owner{.value = next_projected_owner +
                                                     static_cast<std::uint32_t>(index)};
                const bool credit =
                    has_shared_candidate_evidence(candidate.evidence,
                                                  SharedCandidateEvidence::ExplicitBoundary) ||
                    has_shared_candidate_evidence(candidate.evidence,
                                                  SharedCandidateEvidence::RequestedAutomatic);
                owners.push_back(ContextPortfolioOwnerPolicy{
                    .owner                  = owner,
                    .explicit_shared_credit = credit,
                });
                checkpoints.push_back(ContextPortfolioCheckpointValue{
                    .owner                = owner,
                    .demand_mask          = candidate.demand_mask,
                    .rebuild_ns           = candidate.rebuild_ns,
                    .baseline_recovery_ns = candidate.rebuild_ns,
                    .target_recovery_ns   = 0,
                });
            }
            if (surplus_only_count > vacant_shared_slots) { continue; }
            std::sort(frontiers.begin(), frontiers.end());
            const ContextPortfolioValueResult value = projected_value.fold(owners, checkpoints);
            const std::uint64_t schedule_cost       = split_cost(frontiers);
            if (value.saturated ||
                value.private_transition_loss >
                    std::numeric_limits<std::uint64_t>::max() - value.baseline_public_value) {
                continue;
            }
            std::uint64_t threshold = value.baseline_public_value + value.private_transition_loss;
            if (schedule_cost > std::numeric_limits<std::uint64_t>::max() - threshold) { continue; }
            threshold += schedule_cost;
            if (value.target_public_value <= threshold) { continue; }
            const std::uint64_t gain = value.target_public_value - threshold;
            const bool better =
                gain > selected_gain ||
                (gain == selected_gain &&
                 (selected_frontiers.empty() || frontiers.size() < selected_frontiers.size() ||
                  (frontiers.size() == selected_frontiers.size() &&
                   std::lexicographical_compare(frontiers.begin(), frontiers.end(),
                                                selected_frontiers.begin(),
                                                selected_frontiers.end()))));
            if (better) {
                selected_gain      = gain;
                selected_frontiers = std::move(frontiers);
            }
        }
        return selected_frontiers;
    }

    // 组织的规划调用：把"候选 + 所有潜在牺牲者 + 成本/收益政策 + 逻辑目标 + 最终调度意图"打包，
    // 交给 MaterializationPlanner 选出一个方案，再把它翻译成本层的 Choice（含牺牲清单与需求记录）。
    // 三个回调是流程的关键接口：
    //   build_pressure_inputs 惰性构造（只有需要探索压力时才做，且只允许做一次）；
    //   logical_goal 声明"发布格必须落在哪一类格子里"（空位或某个被驱逐者腾出的格子）；
    //   final_schedule 在候选敲定后决定要顺带捕获哪些共享前缀。
    // 注意这里的两次校验：挑选后的方案若与目录状态不符（期间被改动）就返回 nullopt，宁可本轮不干。
    [[nodiscard]] std::optional<Choice>
    plan_materialization(Program& program, const PreparedPrompt& prompt,
                         const RequestBasePlan& base, LaneId destination,
                         std::vector<Candidate>& candidates, std::uint64_t publication_order,
                         typename Planner::Clock::time_point planning_started,
                         PrefixDemandRecord& provisional_demand, PlanningAllowance allowance) {
        std::vector<typename Planner::CandidateInput> candidate_inputs;
        std::vector<const ContinuationHandle*> private_owners;
        std::vector<PlanningOwnerId> private_owner_ids;
        std::vector<const SharedPrefixHandle*> shared_owners;
        std::vector<PlanningOwnerId> shared_owner_ids;
        std::vector<PlanningOwnerRecord> owner_records;
        std::vector<MaterializationOwnerPolicy> owner_policies;
        std::vector<MaterializationCheckpointPolicy> checkpoint_policies;
        candidate_inputs.reserve(candidates.size());

        for (std::size_t index = 0; index < candidates.size(); ++index) {
            if (!candidates[index].plan) {
                throw std::logic_error("materialization candidate is empty");
            }
            candidate_inputs.push_back(typename Planner::CandidateInput{
                .candidate      = &*candidates[index].plan,
                .id             = PlanningCandidateId{.value = static_cast<std::uint32_t>(index)},
                .stable_ordinal = static_cast<std::uint32_t>(index),
                .current_session_binding = candidates[index].current_session_binding,
            });
        }

        bool pressure_inputs_built       = false;
        const auto build_pressure_inputs = [&]() -> typename Planner::PressureInputs {
            if (pressure_inputs_built) {
                throw std::logic_error("materialization pressure inputs requested twice");
            }
            pressure_inputs_built = true;
            private_owners.reserve(catalog_count_);
            private_owner_ids.reserve(catalog_count_);
            shared_owners.reserve(shared_catalog_count_);
            shared_owner_ids.reserve(shared_catalog_count_);
            owner_records.reserve(catalog_count_ + shared_catalog_count_);
            owner_policies.reserve(catalog_count_ + shared_catalog_count_);
            checkpoint_policies.reserve(prefix_index_.size());

            for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
                const CatalogEntry& entry = catalog_[slot];
                if (entry.state != CatalogState::Catalogued || !entry.handle ||
                    private_has_active_edge(slot)) {
                    continue;
                }
                const PlanningOwnerId owner{.value =
                                                static_cast<std::uint32_t>(owner_records.size())};
                private_owners.push_back(&*entry.handle);
                private_owner_ids.push_back(owner);
                owner_records.push_back(PlanningOwnerRecord{
                    .id = owner,
                    .capability =
                        CatalogCapability{
                            .owner =
                                LogicalOwnerKey{
                                    .kind = LogicalOwnerKind::PrivateContinuation,
                                    .id   = entry.id,
                                },
                            .slot       = slot,
                            .generation = entry.revision,
                        },
                });
                std::uint64_t selected_hits  = 0;
                const auto append_checkpoint = [&](const auto& checkpoint) {
                    const RetentionObservation* observation =
                        find_observation(entry.observations, checkpoint.ref);
                    if (observation == nullptr) {
                        throw std::logic_error("catalogued checkpoint has no policy observation");
                    }
                    selected_hits = std::max(selected_hits, observation->selected_hit_count);
                    checkpoint_policies.push_back(MaterializationCheckpointPolicy{
                        .owner              = owner,
                        .checkpoint         = checkpoint.ref,
                        .retention_class    = observation->retention_class,
                        .selected_hit_count = observation->selected_hit_count,
                        .last_hit_epoch     = observation->last_hit_epoch,
                        .demand_mask =
                            demand_mask_for(checkpoint.shortlist_key, provisional_demand),
                        .rebuild_ns           = cost_model_.prefill_ns(checkpoint.rebuild_work),
                        .baseline_recovery_ns = price_checkpoint_recovery_work(
                            cost_model_,
                            program.checkpoint_recovery_work(*entry.handle, checkpoint.ref)),
                    });
                };
                if (entry.summary.endpoint) { append_checkpoint(*entry.summary.endpoint); }
                if (entry.summary.rewrite) { append_checkpoint(*entry.summary.rewrite); }
                for (const auto& checkpoint : entry.summary.long_anchors) {
                    append_checkpoint(checkpoint);
                }
                owner_policies.push_back(MaterializationOwnerPolicy{
                    .owner                    = owner,
                    .retention_class          = entry.retention,
                    .selected_hit_count       = selected_hits,
                    .last_hit_epoch           = newest_hit_epoch(entry),
                    .private_retention_weight = private_retention_weight(entry.retention),
                });
            }
            for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
                const SharedCatalogEntry& entry = shared_catalog_[slot];
                if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                    entry.transaction_pins != 0 || shared_active_edge_count(slot) != 0) {
                    continue;
                }
                const PlanningOwnerId owner{.value =
                                                static_cast<std::uint32_t>(owner_records.size())};
                shared_owners.push_back(&*entry.handle);
                shared_owner_ids.push_back(owner);
                owner_records.push_back(PlanningOwnerRecord{
                    .id = owner,
                    .capability =
                        CatalogCapability{
                            .owner =
                                LogicalOwnerKey{
                                    .kind = LogicalOwnerKind::SharedPrefix,
                                    .id   = entry.id,
                                },
                            .slot       = slot,
                            .generation = entry.revision,
                        },
                });
                owner_policies.push_back(MaterializationOwnerPolicy{
                    .owner                    = owner,
                    .retention_class          = RetentionClass::SharedStable,
                    .selected_hit_count       = entry.observation.selected_hit_count,
                    .last_hit_epoch           = entry.observation.last_hit_epoch,
                    .private_retention_weight = 0,
                    .explicit_shared_credit   = entry.explicit_credit,
                });
                checkpoint_policies.push_back(MaterializationCheckpointPolicy{
                    .owner              = owner,
                    .checkpoint         = entry.summary.checkpoint.ref,
                    .retention_class    = RetentionClass::SharedStable,
                    .selected_hit_count = entry.observation.selected_hit_count,
                    .last_hit_epoch     = entry.observation.last_hit_epoch,
                    .demand_mask =
                        demand_mask_for(entry.summary.checkpoint.shortlist_key, provisional_demand),
                    .rebuild_ns = cost_model_.prefill_ns(entry.summary.checkpoint.rebuild_work),
                    .baseline_recovery_ns = price_checkpoint_recovery_work(
                        cost_model_, program.checkpoint_recovery_work(
                                         *entry.handle, entry.summary.checkpoint.ref)),
                });
            }

            return typename Planner::PressureInputs{
                .private_owners    = private_owners,
                .private_owner_ids = private_owner_ids,
                .shared_owners     = shared_owners,
                .shared_owner_ids  = shared_owner_ids,
                .owner_policy      = owner_policies,
                .checkpoint_policy = checkpoint_policies,
            };
        };

        const auto logical_goal = [&](PlanningCandidateId candidate_id,
                                      PrivateSourceMode source_mode,
                                      std::span<const PressureOwnerOutcome> outcomes)
            -> std::optional<typename Planner::LogicalGoal> {
            const auto candidate_record =
                std::find_if(candidate_inputs.begin(), candidate_inputs.end(),
                             [&](const typename Planner::CandidateInput& input) {
                                 return input.id == candidate_id;
                             });
            if (candidate_record == candidate_inputs.end()) { return std::nullopt; }
            const std::size_t candidate_index =
                static_cast<std::size_t>(candidate_record - candidate_inputs.begin());
            const Candidate& candidate = candidates[candidate_index];
            if (candidate.shared_source && source_mode != PrivateSourceMode::Retain) {
                return std::nullopt;
            }
            if (!candidate.private_source && !candidate.shared_source &&
                source_mode == PrivateSourceMode::Retain) {
                return std::nullopt;
            }
            if (candidate.private_source && source_mode != PrivateSourceMode::Retain &&
                source_mode != PrivateSourceMode::ConsumeToActive) {
                return std::nullopt;
            }

            std::uint32_t publication_slot = kInvalidCatalogSlot;
            if (candidate.private_source && source_mode == PrivateSourceMode::ConsumeToActive) {
                publication_slot = candidate.private_source->slot;
            } else {
                for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
                    if (catalog_[slot].state == CatalogState::Vacant) {
                        publication_slot = slot;
                        break;
                    }
                }
            }

            for (std::size_t row = 0; row < outcomes.size(); ++row) {
                const PressureOwnerOutcome& outcome = outcomes[row];
                if (outcome.disposition != VictimDisposition::Retained &&
                    outcome.disposition != VictimDisposition::Evicted) {
                    return std::nullopt;
                }
                if (std::find_if(outcomes.begin(), outcomes.begin() + row,
                                 [&](const PressureOwnerOutcome& prior) {
                                     return prior.owner == outcome.owner;
                                 }) != outcomes.begin() + row) {
                    return std::nullopt;
                }
                const auto record = std::find_if(
                    owner_records.begin(), owner_records.end(),
                    [&](const PlanningOwnerRecord& item) { return item.id == outcome.owner; });
                if (record == owner_records.end()) { return std::nullopt; }
                const bool shared = record->capability.owner.kind == LogicalOwnerKind::SharedPrefix;
                if (!shared) {
                    const std::uint32_t slot = record->capability.slot;
                    if (slot >= catalog_count_ ||
                        (candidate.private_source && slot == candidate.private_source->slot)) {
                        return std::nullopt;
                    }
                    const CatalogEntry& entry = catalog_[slot];
                    if (entry.state != CatalogState::Catalogued || !entry.handle ||
                        entry.id != record->capability.owner.id ||
                        entry.revision != record->capability.generation ||
                        private_has_active_edge(slot)) {
                        return std::nullopt;
                    }
                    if (publication_slot == kInvalidCatalogSlot &&
                        outcome.disposition == VictimDisposition::Evicted) {
                        publication_slot = slot;
                    }
                } else {
                    const std::uint32_t slot = record->capability.slot;
                    if (slot >= shared_catalog_count_ ||
                        (candidate.shared_source && slot == candidate.shared_source->slot)) {
                        return std::nullopt;
                    }
                    const SharedCatalogEntry& entry = shared_catalog_[slot];
                    if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                        entry.id != record->capability.owner.id ||
                        entry.revision != record->capability.generation ||
                        entry.transaction_pins != 0 || shared_active_edge_count(slot) != 0) {
                        return std::nullopt;
                    }
                }
            }
            if (publication_slot == kInvalidCatalogSlot) { return std::nullopt; }
            return typename Planner::LogicalGoal{.publication_slot = publication_slot};
        };

        const auto final_schedule = [&](PlanningCandidateId candidate_id,
                                        const RequestPlanSummary& summary,
                                        const auto& split_cost) -> std::vector<std::uint32_t> {
            const auto selected = std::find_if(candidate_inputs.begin(), candidate_inputs.end(),
                                               [&](const typename Planner::CandidateInput& input) {
                                                   return input.id == candidate_id;
                                               });
            if (selected == candidate_inputs.end()) {
                throw std::logic_error("final schedule references an unknown candidate");
            }
            const Candidate& candidate =
                candidates[static_cast<std::size_t>(selected - candidate_inputs.begin())];
            return select_materialization_shared_captures(program, base, candidate, summary,
                                                          provisional_demand, split_cost);
        };

        std::optional<typename Planner::Result> planned =
            planner_.plan(program, prompt, cost_model_, candidate_inputs, 0, build_pressure_inputs,
                          logical_goal, final_schedule, planning_started, allowance);
        const auto selected_candidate =
            planned ? std::find_if(candidate_inputs.begin(), candidate_inputs.end(),
                                   [&](const typename Planner::CandidateInput& input) {
                                       return input.id == planned->candidate;
                                   })
                    : candidate_inputs.end();
        if (!planned || !planned->plan || selected_candidate == candidate_inputs.end()) {
            return std::nullopt;
        }

        Candidate& candidate =
            candidates[static_cast<std::size_t>(selected_candidate - candidate_inputs.begin())];

        Choice choice(destination, std::move(*planned->plan), catalog_count_,
                      base.context_cache().session_key, base.context_cache().retention,
                      base.context_cache().update_session_index, publication_order);
        choice.private_source_                 = candidate.private_source;
        choice.source_mode_                    = planned->source_mode;
        choice.shared_source_                  = candidate.shared_source;
        choice.publication_slot_               = planned->publication_slot;
        choice.selected_observation_           = candidate.selected_observation;
        choice.diagnostics_                    = planned->diagnostics;
        provisional_demand.selected_source_key = candidate.source_key;
        choice.demand_                         = std::move(provisional_demand);
        for (const PressureOwnerOutcome& outcome : planned->owner_outcomes) {
            const auto record = std::find_if(
                owner_records.begin(), owner_records.end(),
                [&](const PlanningOwnerRecord& item) { return item.id == outcome.owner; });
            if (record == owner_records.end()) {
                throw std::logic_error("selected pressure outcome has no logical owner record");
            }
            const bool shared = record->capability.owner.kind == LogicalOwnerKind::SharedPrefix;
            if (!shared) {
                const CatalogEntry& entry          = catalog_.at(record->capability.slot);
                std::vector<CheckpointRef> dropped = selected_checkpoint_drops(
                    outcome.owner, outcome.disposition, outcome.dropped_checkpoints,
                    planned->checkpoint_outcomes, continuation_checkpoint_count(entry.summary),
                    [&](CheckpointRef checkpoint) {
                        return continuation_contains_checkpoint(entry.summary, checkpoint);
                    });
                choice.private_claims_.push_back(OwnerClaim{
                    .planning_id         = outcome.owner,
                    .capability          = record->capability,
                    .disposition         = outcome.disposition,
                    .dropped_checkpoints = std::move(dropped),
                });
            } else {
                const SharedCatalogEntry& entry    = shared_catalog_.at(record->capability.slot);
                std::vector<CheckpointRef> dropped = selected_checkpoint_drops(
                    outcome.owner, outcome.disposition, outcome.dropped_checkpoints,
                    planned->checkpoint_outcomes, 1U, [&](CheckpointRef checkpoint) {
                        return checkpoint == entry.summary.checkpoint.ref;
                    });
                choice.shared_claims_.push_back(OwnerClaim{
                    .planning_id         = outcome.owner,
                    .capability          = record->capability,
                    .disposition         = outcome.disposition,
                    .dropped_checkpoints = std::move(dropped),
                });
            }
        }
        return choice;
    }

    // 动手前的最后一道静态校验：规划期算出的 Choice 是可过期的意图，这里逐项确认世界仍然一致——
    // 目标 lane 仍空闲、来源/牺牲者的 id 与 revision 未变、没有新出现的活跃租约、发布格确实可用
    // （要么本来就是空的，要么会被来源或某个被驱逐的牺牲者腾出来）。任一不符即抛，交回上层重新规划。
    void validate_choice(const Choice& choice, ProgramResourceRevision revision) const {
        if (!choice.plan_ || choice.destination_.value >= lane_count_ ||
            lanes_[choice.destination_.value] != LogicalLaneState::Free || revision.value == 0 ||
            choice.publication_slot_ >= catalog_count_ || choice.publication_order_ == 0) {
            throw std::logic_error("resource choice is stale or malformed");
        }
        if (choice.private_source_) {
            const CatalogCapability& capability = *choice.private_source_;
            if (capability.owner.kind != LogicalOwnerKind::PrivateContinuation ||
                capability.slot >= catalog_count_) {
                throw std::logic_error("private source slot is invalid");
            }
            const CatalogEntry& source = catalog_[capability.slot];
            if (source.state != CatalogState::Catalogued || !source.handle ||
                source.id != capability.owner.id || source.revision != capability.generation ||
                private_has_active_edge(capability.slot)) {
                throw std::logic_error("private source changed after planning");
            }
        }
        if (choice.shared_source_) {
            const CatalogCapability& capability = *choice.shared_source_;
            if (capability.owner.kind != LogicalOwnerKind::SharedPrefix ||
                capability.slot >= shared_catalog_count_) {
                throw std::logic_error("shared source slot is invalid");
            }
            const SharedCatalogEntry& source = shared_catalog_[capability.slot];
            if (source.state != SharedCatalogState::Catalogued || !source.handle ||
                source.id != capability.owner.id || source.revision != capability.generation) {
                throw std::logic_error("shared source changed after planning");
            }
        }
        for (const OwnerClaim& claim : choice.private_claims_) {
            const std::uint32_t slot = claim.capability.slot;
            if (slot >= catalog_count_ ||
                (choice.private_source_ && slot == choice.private_source_->slot)) {
                throw std::logic_error("private pressure owner is invalid");
            }
            const CatalogEntry& entry = catalog_[slot];
            if (entry.state != CatalogState::Catalogued || !entry.handle ||
                claim.capability.owner.kind != LogicalOwnerKind::PrivateContinuation ||
                entry.id != claim.capability.owner.id ||
                entry.revision != claim.capability.generation || private_has_active_edge(slot) ||
                (claim.disposition != VictimDisposition::Retained &&
                 claim.disposition != VictimDisposition::Evicted)) {
                throw std::logic_error("private pressure owner changed after planning");
            }
        }
        for (const OwnerClaim& claim : choice.shared_claims_) {
            const std::uint32_t slot = claim.capability.slot;
            if (slot >= shared_catalog_count_ ||
                (choice.shared_source_ && slot == choice.shared_source_->slot)) {
                throw std::logic_error("shared pressure owner is invalid");
            }
            const SharedCatalogEntry& entry = shared_catalog_[slot];
            if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                claim.capability.owner.kind != LogicalOwnerKind::SharedPrefix ||
                entry.id != claim.capability.owner.id ||
                entry.revision != claim.capability.generation || entry.transaction_pins != 0 ||
                shared_active_edge_count(slot) != 0 ||
                (claim.disposition != VictimDisposition::Retained &&
                 claim.disposition != VictimDisposition::Evicted)) {
                throw std::logic_error("shared pressure owner changed after planning");
            }
        }
        const CatalogEntry& publication = catalog_[choice.publication_slot_];
        const bool source_cell          = choice.private_source_ &&
                                 choice.publication_slot_ == choice.private_source_->slot &&
                                 choice.source_mode_ == PrivateSourceMode::ConsumeToActive;
        const auto victim =
            std::find_if(choice.private_claims_.begin(), choice.private_claims_.end(),
                         [&](const OwnerClaim& claim) {
                             return claim.capability.slot == choice.publication_slot_;
                         });
        const bool victim_cell = victim != choice.private_claims_.end() &&
                                 victim->disposition == VictimDisposition::Evicted;
        if (publication.state != CatalogState::Vacant && !source_cell && !victim_cell) {
            throw std::logic_error("resource choice has no publication cell");
        }
    }

    [[nodiscard]] MaterializationRecord take_materialization_record(Choice& choice) {
        return MaterializationRecord{
            .destination          = choice.destination_,
            .private_source       = choice.private_source_,
            .source_mode          = choice.source_mode_,
            .shared_source        = choice.shared_source_,
            .publication_slot     = choice.publication_slot_,
            .private_claims       = std::move(choice.private_claims_),
            .shared_claims        = std::move(choice.shared_claims_),
            .selected_observation = choice.selected_observation_,
            .session              = std::move(choice.session_),
            .retention            = choice.retention_,
            .update_session_index = choice.update_session_index_,
            .publication_order    = choice.publication_order_,
            .diagnostics          = choice.diagnostics_,
            .demand               = std::move(choice.demand_),
        };
    }

    // 冻结与回滚是一对：reserve_* 把涉及到的格子标成事务期状态（别人再也不能选中它们），
    // rollback_* 在 Program 拒绝或事务中止时把它们原样还原。两者必须严格对称，否则会出现
    // "格子永久卡在 Claimed"这类不可见泄漏。
    void reserve_logical_materialization(const MaterializationRecord& record) noexcept {
        lanes_[record.destination.value] = LogicalLaneState::Materializing;
        if (record.private_source) {
            catalog_[record.private_source->slot].state = CatalogState::Claimed;
        }
        if (record.shared_source) {
            ++shared_catalog_[record.shared_source->slot].transaction_pins;
        }
        for (const OwnerClaim& claim : record.private_claims) {
            catalog_[claim.capability.slot].state = CatalogState::Claimed;
        }
        for (const OwnerClaim& claim : record.shared_claims) {
            shared_catalog_[claim.capability.slot].state = SharedCatalogState::Claimed;
        }
        CatalogEntry& publication = catalog_[record.publication_slot];
        if (publication.state == CatalogState::Vacant) {
            publication.state = CatalogState::Claimed;
        }
    }

    void rollback_logical_materialization(const MaterializationRecord& record) noexcept {
        lanes_[record.destination.value] = LogicalLaneState::Free;
        if (record.private_source) {
            catalog_[record.private_source->slot].state = CatalogState::Catalogued;
        }
        if (record.shared_source) {
            SharedCatalogEntry& source = shared_catalog_[record.shared_source->slot];
            if (source.transaction_pins != 0) { --source.transaction_pins; }
        }
        for (const OwnerClaim& claim : record.private_claims) {
            catalog_[claim.capability.slot].state = CatalogState::Catalogued;
        }
        for (const OwnerClaim& claim : record.shared_claims) {
            shared_catalog_[claim.capability.slot].state = SharedCatalogState::Catalogued;
        }
        CatalogEntry& publication = catalog_[record.publication_slot];
        if (publication.id == 0 && !publication.handle) {
            publication.state = CatalogState::Vacant;
        }
    }

    void reserve_logical_active_capture(const ActiveCaptureRecord& record) noexcept {
        for (const OwnerClaim& claim : record.private_claims) {
            catalog_[claim.capability.slot].state = CatalogState::Claimed;
        }
        for (const OwnerClaim& claim : record.shared_claims) {
            shared_catalog_[claim.capability.slot].state = SharedCatalogState::Claimed;
        }
        shared_catalog_[record.publication_slot].state = SharedCatalogState::ReservedCapture;
    }

    void rollback_logical_active_capture(const ActiveCaptureRecord& record) noexcept {
        for (const OwnerClaim& claim : record.private_claims) {
            catalog_[claim.capability.slot].state = CatalogState::Catalogued;
        }
        for (const OwnerClaim& claim : record.shared_claims) {
            shared_catalog_[claim.capability.slot].state = SharedCatalogState::Catalogued;
        }
        shared_catalog_[record.publication_slot].state = record.replacement_id == 0
                                                             ? SharedCatalogState::Vacant
                                                             : SharedCatalogState::Catalogued;
    }

    // 记一次"被选中"：只有**成功发布**的物化才算命中（所以这里在 published 分支才调用）。
    // epoch 单调递增，用来给保留策略提供"最近有多常用"的时间轴。
    void observe_selected_hit(const MaterializationRecord& record) noexcept {
        if (record.selected_observation) {
            RetentionObservation* selected = resolve_observation(*record.selected_observation);
            if (selected) {
                saturating_increment(selected->selected_hit_count);
                selected->last_hit_epoch = ++retention_epoch_;
            }
        }
    }

    // 需求窗口是一个固定长度的滑动窗（最旧的被挤掉）：它把"最近若干次请求想要什么前缀"变成可比较的
    // 证据，用来给共享条目加权、也给 explicit_credit 设置过期。窗口容量在构造时就预留。
    void commit_demand(PrefixDemandRecord&& demand) noexcept {
        if (demand_window_.capacity() < kDemandWindowCapacity) { std::terminate(); }
        if (demand_window_.size() == kDemandWindowCapacity) {
            demand_window_.erase(demand_window_.begin());
        }
        demand_window_.push_back(std::move(demand));
        saturating_increment(demand_epoch_);
        const PrefixDemandRecord& committed = demand_window_.back();
        for (SharedCatalogEntry& entry : shared_catalog_) {
            if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                !entry.explicit_credit) {
                continue;
            }
            if (std::find(committed.exact_resident_keys.begin(),
                          committed.exact_resident_keys.end(),
                          entry.summary.checkpoint.shortlist_key) !=
                committed.exact_resident_keys.end()) {
                entry.explicit_credit     = false;
                entry.credit_expiry_epoch = 0;
                continue;
            }
            if (demand_epoch_ >= entry.credit_expiry_epoch) {
                entry.explicit_credit     = false;
                entry.credit_expiry_epoch = 0;
            }
        }
    }

    void observe_planner_diagnostics(const MaterializationDiagnostics& diagnostics) noexcept {
        if (diagnostics.stop_reason != MaterializationStopReason::NoPressure) {
            saturating_increment(context_stats_.pressure_searches);
        }
        if (diagnostics.budget_exhausted) {
            saturating_increment(context_stats_.pressure_search_budget_exhaustions);
        }
        if (diagnostics.selected_maximal_fallback) {
            saturating_increment(context_stats_.pressure_maximal_fallback_selections);
        }
    }

    // 把观测键解析回具体的观测槽；键里的 revision 与当前条目不一致就返回 nullptr——观测记录跨代次
    // 就没有意义了，宁可丢掉也不能记到新条目头上。
    [[nodiscard]] RetentionObservation*
    resolve_observation(const PolicyObservationKey& key) noexcept {
        if (!key.shared) {
            if (key.slot >= catalog_count_) { return nullptr; }
            CatalogEntry& entry = catalog_[key.slot];
            if (entry.id != key.owner_id || entry.revision != key.revision) { return nullptr; }
            return find_observation(entry.observations, key.checkpoint);
        }
        if (key.slot >= shared_catalog_count_) { return nullptr; }
        SharedCatalogEntry& entry = shared_catalog_[key.slot];
        if (entry.id != key.owner_id || entry.revision != key.revision ||
            entry.summary.checkpoint.ref != key.checkpoint) {
            return nullptr;
        }
        return &entry.observation;
    }

    [[nodiscard]] static bool continuation_contains_checkpoint(const ContinuationSummary& summary,
                                                               CheckpointRef checkpoint) noexcept {
        if (summary.endpoint && summary.endpoint->ref == checkpoint) { return true; }
        if (summary.rewrite && summary.rewrite->ref == checkpoint) { return true; }
        return std::any_of(summary.long_anchors.begin(), summary.long_anchors.end(),
                           [&](const auto& anchor) { return anchor.ref == checkpoint; });
    }

    [[nodiscard]] static std::uint32_t
    continuation_checkpoint_count(const ContinuationSummary& summary) noexcept {
        const std::size_t count = static_cast<std::size_t>(summary.endpoint.has_value()) +
                                  static_cast<std::size_t>(summary.rewrite.has_value()) +
                                  summary.long_anchors.size();
        return count > std::numeric_limits<std::uint32_t>::max()
                   ? std::numeric_limits<std::uint32_t>::max()
                   : static_cast<std::uint32_t>(count);
    }

    // 把 planner 报出的"哪些 checkpoint 没保住"翻译成本层要执行的删除清单，并顺带做对账：
    // 报出的必须属于这个 owner、不能重复、数量要与其持有的 checkpoint 总数吻合，否则抛错——
    // 少报一个就意味着有 checkpoint 会静默泄漏。
    template <class ContainsCheckpoint>
    [[nodiscard]] static std::vector<CheckpointRef> selected_checkpoint_drops(
        PlanningOwnerId owner, VictimDisposition disposition, std::uint32_t expected_drop_count,
        std::span<const PressureCheckpointOutcome> outcomes, std::uint32_t checkpoint_count,
        ContainsCheckpoint&& contains_checkpoint) {
        std::vector<CheckpointRef> dropped;
        dropped.reserve(expected_drop_count);
        std::uint32_t observed = 0;
        for (std::size_t index = 0; index < outcomes.size(); ++index) {
            const PressureCheckpointOutcome& outcome = outcomes[index];
            if (outcome.owner != owner) { continue; }
            if (!contains_checkpoint(outcome.checkpoint) ||
                std::find_if(outcomes.begin(), outcomes.begin() + index,
                             [&](const PressureCheckpointOutcome& prior) {
                                 return prior.owner == owner &&
                                        prior.checkpoint == outcome.checkpoint;
                             }) != outcomes.begin() + index) {
                throw std::logic_error("selected checkpoint outcome is unknown or duplicated");
            }
            ++observed;
            if (!outcome.survives) { dropped.push_back(outcome.checkpoint); }
        }
        if (observed != checkpoint_count || dropped.size() != expected_drop_count ||
            (disposition == VictimDisposition::Evicted) != (dropped.size() == checkpoint_count)) {
            throw std::logic_error("selected checkpoint outcome is incomplete");
        }
        return dropped;
    }

    [[nodiscard]] static bool continuation_matches_claim(
        const ContinuationSummary& before, const std::optional<ContinuationSummary>& after,
        VictimDisposition disposition, std::span<const CheckpointRef> expected_drops) noexcept {
        const std::uint32_t before_count = continuation_checkpoint_count(before);
        if (disposition == VictimDisposition::Evicted) {
            return !after && expected_drops.size() == before_count;
        }
        if (disposition != VictimDisposition::Retained || !after ||
            continuation_checkpoint_count(*after) + expected_drops.size() != before_count) {
            return false;
        }
        const auto expected_drop = [&](CheckpointRef checkpoint) {
            return std::find(expected_drops.begin(), expected_drops.end(), checkpoint) !=
                   expected_drops.end();
        };
        const auto check = [&](CheckpointRef checkpoint) {
            return continuation_contains_checkpoint(*after, checkpoint) !=
                   expected_drop(checkpoint);
        };
        if ((before.endpoint && !check(before.endpoint->ref)) ||
            (before.rewrite && !check(before.rewrite->ref))) {
            return false;
        }
        for (const auto& anchor : before.long_anchors) {
            if (!check(anchor.ref)) { return false; }
        }
        return std::all_of(expected_drops.begin(), expected_drops.end(), [&](CheckpointRef drop) {
            return continuation_contains_checkpoint(before, drop);
        });
    }

    [[nodiscard]] static std::uint32_t
    dropped_checkpoint_count(const ContinuationSummary& before,
                             const std::optional<ContinuationSummary>& after,
                             VictimDisposition disposition) noexcept {
        if (disposition == VictimDisposition::Evicted) {
            return continuation_checkpoint_count(before);
        }
        if (!after) { return 0; }
        std::uint32_t dropped = 0;
        const auto visit      = [&](const auto& checkpoint) {
            if (checkpoint && !continuation_contains_checkpoint(*after, checkpoint->ref)) {
                ++dropped;
            }
        };
        visit(before.endpoint);
        visit(before.rewrite);
        for (const auto& anchor : before.long_anchors) {
            if (!continuation_contains_checkpoint(*after, anchor.ref)) { ++dropped; }
        }
        return dropped;
    }

    static void record_checkpoint_drops(RuntimeStats& stats, std::uint32_t count) noexcept {
        for (std::uint32_t index = 0; index < count; ++index) {
            saturating_increment(stats.pressure_checkpoints_dropped);
        }
    }

    // validate_*_action / apply_*_action 是"预登记 vs 实际结局"的对账对，四者成两对。
    // 校验阶段回答"Program 报的 disposition 在我们冻结的意图里合法吗"，应用阶段才真正改账。
    // 分两步是为了：先整体验完再动手，避免验到一半发现不符却已经改了前几条。
    template <class Result>
    void validate_private_action(const OwnerClaim& claim, bool target_committed,
                                 const Result& result) const {
        const std::uint32_t slot = claim.capability.slot;
        if (slot >= catalog_count_) {
            throw std::logic_error("private action result has an invalid slot");
        }
        const CatalogEntry& entry = catalog_[slot];
        if (claim.capability.owner.kind != LogicalOwnerKind::PrivateContinuation ||
            entry.state != CatalogState::Claimed || entry.id != claim.capability.owner.id ||
            entry.revision != claim.capability.generation || !entry.handle) {
            throw std::logic_error("private action owner changed before adoption");
        }
        if (result.final_summary && !valid_continuation_summary(*result.final_summary)) {
            throw std::logic_error("private action returned an invalid final summary");
        }
        const std::uint32_t dropped =
            dropped_checkpoint_count(entry.summary, result.final_summary, result.disposition);
        if (target_committed && !result.pressure_committed) {
            throw std::logic_error("selected private pressure action was not committed");
        }
        if (result.pressure_committed &&
            (result.disposition != claim.disposition ||
             !continuation_matches_claim(entry.summary, result.final_summary, result.disposition,
                                         claim.dropped_checkpoints))) {
            throw std::logic_error("private owner outcome differs from the selected target");
        }
        if (result.disposition == VictimDisposition::Evicted) {
            if (!result.pressure_committed || result.final_summary) {
                throw std::logic_error("private eviction result is malformed");
            }
            return;
        }
        if (result.disposition != VictimDisposition::Retained) {
            throw std::logic_error("private pressure action returned an invalid disposition");
        }
        if (!result.pressure_committed &&
            (result.disposition != VictimDisposition::Retained || dropped != 0 ||
             (result.final_summary &&
              !continuation_matches_claim(entry.summary, result.final_summary,
                                          VictimDisposition::Retained, {})))) {
            throw std::logic_error("uncommitted private pressure action changed checkpoints");
        }
        if (result.pressure_committed && !result.final_summary) {
            throw std::logic_error("committed private pressure action has no final summary");
        }
    }

    template <class Result>
    void validate_shared_action(const OwnerClaim& claim, bool target_committed,
                                const Result& result) const {
        const std::uint32_t slot = claim.capability.slot;
        if (slot >= shared_catalog_count_) {
            throw std::logic_error("shared action result has an invalid slot");
        }
        const SharedCatalogEntry& entry = shared_catalog_[slot];
        if (claim.capability.owner.kind != LogicalOwnerKind::SharedPrefix ||
            entry.state != SharedCatalogState::Claimed || entry.id != claim.capability.owner.id ||
            entry.revision != claim.capability.generation || !entry.handle) {
            throw std::logic_error("shared action owner changed before adoption");
        }
        if (result.final_summary && !valid_shared_prefix_summary(*result.final_summary)) {
            throw std::logic_error("shared action returned an invalid final summary");
        }
        if (target_committed && !result.pressure_committed) {
            throw std::logic_error("selected shared pressure action was not committed");
        }
        const bool exact_committed_outcome =
            result.disposition == VictimDisposition::Evicted
                ? claim.dropped_checkpoints.size() == 1U &&
                      claim.dropped_checkpoints.front() == entry.summary.checkpoint.ref &&
                      !result.final_summary
                : result.disposition == VictimDisposition::Retained &&
                      claim.dropped_checkpoints.empty() && result.final_summary &&
                      result.final_summary->checkpoint.ref == entry.summary.checkpoint.ref;
        if (result.pressure_committed &&
            (result.disposition != claim.disposition || !exact_committed_outcome)) {
            throw std::logic_error("shared owner outcome differs from the selected target");
        }
        if (result.disposition == VictimDisposition::Evicted) {
            if (!result.pressure_committed || result.final_summary) {
                throw std::logic_error("shared eviction result is malformed");
            }
            return;
        }
        if (result.disposition != VictimDisposition::Retained) {
            throw std::logic_error("shared pressure action returned an invalid disposition");
        }
        if (!result.pressure_committed && result.final_summary &&
            result.final_summary->checkpoint.ref != entry.summary.checkpoint.ref) {
            throw std::logic_error(
                "uncommitted shared pressure action changed checkpoint identity");
        }
        if (result.pressure_committed && !result.final_summary) {
            throw std::logic_error("committed shared pressure action has no final summary");
        }
    }

    template <class Result>
    void apply_private_action(const OwnerClaim& claim, bool target_committed,
                              const Result& result) noexcept {
        (void)target_committed;
        const std::uint32_t slot = claim.capability.slot;
        CatalogEntry& entry      = catalog_[slot];
        const std::uint32_t dropped =
            dropped_checkpoint_count(entry.summary, result.final_summary, result.disposition);
        if (result.disposition == VictimDisposition::Evicted) {
            erase_session_if_owner(claim.capability.owner.id);
            clear_catalog_entry(entry);
            saturating_increment(context_stats_.pressure_private_owners_evicted);
            record_checkpoint_drops(context_stats_, dropped);
            return;
        }
        if (!result.pressure_committed) {
            entry.state = CatalogState::Catalogued;
            return;
        }
        assign_continuation_summary(entry.summary, *result.final_summary);
        migrate_observations(entry, *result.final_summary, entry.retention);
        advance_revision(entry.revision);
        refresh_session_owner_revision(claim.capability.owner.id, slot, entry.revision);
        saturating_increment(context_stats_.pressure_private_owners_degraded);
        record_checkpoint_drops(context_stats_, dropped);
        entry.state = CatalogState::Catalogued;
    }

    template <class Result>
    void apply_shared_action(const OwnerClaim& claim, bool target_committed,
                             const Result& result) noexcept {
        (void)target_committed;
        const std::uint32_t slot    = claim.capability.slot;
        SharedCatalogEntry& entry   = shared_catalog_[slot];
        const std::uint32_t dropped = result.disposition == VictimDisposition::Evicted ? 1U : 0U;
        if (result.disposition == VictimDisposition::Evicted) {
            clear_shared_entry(entry);
            saturating_increment(context_stats_.pressure_shared_owners_evicted);
            record_checkpoint_drops(context_stats_, dropped);
            return;
        }
        if (!result.pressure_committed) {
            entry.state = SharedCatalogState::Catalogued;
            return;
        }
        entry.summary = *result.final_summary;
        advance_revision(entry.revision);
        saturating_increment(context_stats_.pressure_shared_owners_degraded);
        record_checkpoint_drops(context_stats_, dropped);
        entry.state = SharedCatalogState::Catalogued;
    }

    // 中止时的还原：只还原那些**还停在 Claimed**的格子（说明 Program 没对它们做出任何报告），
    // 已经落到终态的条目不动。这让"部分生效后中止"也能得到一致账本。
    void restore_unreported_materialization(const MaterializationRecord& record) noexcept {
        if (record.private_source &&
            catalog_[record.private_source->slot].state == CatalogState::Claimed) {
            catalog_[record.private_source->slot].state = CatalogState::Catalogued;
        }
        if (record.shared_source) {
            SharedCatalogEntry& source = shared_catalog_[record.shared_source->slot];
            if (source.transaction_pins != 0) { --source.transaction_pins; }
        }
        for (const OwnerClaim& claim : record.private_claims) {
            CatalogEntry& entry = catalog_[claim.capability.slot];
            if (entry.state == CatalogState::Claimed) { entry.state = CatalogState::Catalogued; }
        }
        for (const OwnerClaim& claim : record.shared_claims) {
            SharedCatalogEntry& entry = shared_catalog_[claim.capability.slot];
            if (entry.state == SharedCatalogState::Claimed) {
                entry.state = SharedCatalogState::Catalogued;
            }
        }
        CatalogEntry& publication = catalog_[record.publication_slot];
        if (publication.state == CatalogState::Claimed && publication.id == 0 &&
            !publication.handle) {
            publication.state = CatalogState::Vacant;
        }
    }

    // 吸收一次物化的终态——本文件里最重的一段，但它的骨架始终是同一句话：
    // **"Program 报的绝对最终状态，必须与我冻结的意图逐条对得上，然后才据实改账。"**
    // 顺序：
    //   1. 形态校验：终态不能是 InProgress、受害者数量必须与预登记一致、每个 id 唯一；
    //   2. 结构校验：目标 lane / 发布格 / 来源能力都还停在预期的临时状态；
    //   3. 逐条 validate_*_action（disposition 是否合法、快照是否自洽）；
    //   4. 逐条 apply_*_action，再处理来源（保留则更新摘要与 revision，被消费则清空）与共享来源的 pin；
    //   5. 中止就还原并释放 lane；发布则把发布格转 ReservedForActive、建立 ActiveEntry 与租约、
    //      提交需求记录，最后交出一张 PublishedActivation——但那还不算生效，要等 adopt()。
    [[nodiscard]] MaterializationOutcome
    adopt_materialization_progress(Program& program, ProgramMaterializationResult&& result) {
        MaterializationRecord* record = std::get_if<MaterializationRecord>(&transaction_);
        if (record == nullptr || !program.has_context_transaction()) {
            throw std::logic_error("materialization result has no logical transaction");
        }
        if (result.status == ContextTransactionStatus::InProgress) {
            throw std::logic_error("terminal materialization result is marked in progress");
        }
        if (result.victims.size() != record->private_claims.size() ||
            result.shared_victims.size() != record->shared_claims.size()) {
            throw std::logic_error("materialization result is not action aligned");
        }
        const auto private_result_for = [&](const OwnerClaim& claim) -> const auto& {
            const auto found =
                std::find_if(result.victims.begin(), result.victims.end(),
                             [&](const auto& row) { return row.owner == claim.planning_id; });
            if (found == result.victims.end() ||
                std::find_if(found + 1, result.victims.end(), [&](const auto& row) {
                    return row.owner == claim.planning_id;
                }) != result.victims.end()) {
                throw std::logic_error("materialization private result ID is not unique");
            }
            return *found;
        };
        const auto shared_result_for = [&](const OwnerClaim& claim) -> const auto& {
            const auto found =
                std::find_if(result.shared_victims.begin(), result.shared_victims.end(),
                             [&](const auto& row) { return row.owner == claim.planning_id; });
            if (found == result.shared_victims.end() ||
                std::find_if(found + 1, result.shared_victims.end(), [&](const auto& row) {
                    return row.owner == claim.planning_id;
                }) != result.shared_victims.end()) {
                throw std::logic_error("materialization shared result ID is not unique");
            }
            return *found;
        };

        const bool published = result.status == ContextTransactionStatus::Published;
        if ((!published && result.status != ContextTransactionStatus::Aborted) ||
            published != result.published.has_value()) {
            throw std::logic_error("materialization terminal status is invalid");
        }
        if (record->destination.value >= lane_count_ ||
            lanes_[record->destination.value] != LogicalLaneState::Materializing ||
            active_[record->destination.value].occupied ||
            active_[record->destination.value].retained_private_source ||
            !active_[record->destination.value].shared_sources.empty() ||
            active_[record->destination.value].shared_sources.capacity() < shared_catalog_count_ ||
            record->publication_slot >= catalog_count_ ||
            catalog_[record->publication_slot].state != CatalogState::Claimed) {
            throw std::logic_error("materialization logical destination changed before adoption");
        }
        for (std::size_t row = 0; row < record->private_claims.size(); ++row) {
            const OwnerClaim& claim = record->private_claims[row];
            if ((record->private_source && claim.capability.slot == record->private_source->slot) ||
                std::find_if(record->private_claims.begin(),
                             record->private_claims.begin() + static_cast<std::ptrdiff_t>(row),
                             [&](const OwnerClaim& prior) {
                                 return prior.planning_id == claim.planning_id ||
                                        prior.capability == claim.capability;
                             }) !=
                    record->private_claims.begin() + static_cast<std::ptrdiff_t>(row)) {
                throw std::logic_error("materialization private claim manifest is not unique");
            }
        }
        for (std::size_t row = 0; row < record->shared_claims.size(); ++row) {
            const OwnerClaim& claim = record->shared_claims[row];
            if (std::any_of(record->private_claims.begin(), record->private_claims.end(),
                            [&](const OwnerClaim& prior) {
                                return prior.planning_id == claim.planning_id;
                            })) {
                throw std::logic_error("materialization owner ID changes kind");
            }
            if ((record->shared_source && claim.capability.slot == record->shared_source->slot) ||
                std::find_if(record->shared_claims.begin(),
                             record->shared_claims.begin() + static_cast<std::ptrdiff_t>(row),
                             [&](const OwnerClaim& prior) {
                                 return prior.planning_id == claim.planning_id ||
                                        prior.capability == claim.capability;
                             }) !=
                    record->shared_claims.begin() + static_cast<std::ptrdiff_t>(row)) {
                throw std::logic_error("materialization shared claim manifest is not unique");
            }
        }
        for (const OwnerClaim& claim : record->private_claims) {
            validate_private_action(claim, published, private_result_for(claim));
        }
        for (const OwnerClaim& claim : record->shared_claims) {
            validate_shared_action(claim, published, shared_result_for(claim));
        }

        if (record->private_source) {
            const CatalogCapability& capability = *record->private_source;
            if (capability.owner.kind != LogicalOwnerKind::PrivateContinuation ||
                capability.slot >= catalog_count_) {
                throw std::logic_error("materialization private source capability is malformed");
            }
            const CatalogEntry& source = catalog_[capability.slot];
            if (!result.source || source.state != CatalogState::Claimed ||
                source.id != capability.owner.id || source.revision != capability.generation ||
                (!published && result.source->mode != PrivateSourceMode::Retain) ||
                (published && result.source->mode != record->source_mode) ||
                (result.source->final_summary &&
                 !valid_continuation_summary(*result.source->final_summary)) ||
                (result.source->mode == PrivateSourceMode::ConsumeToActive &&
                 result.source->final_summary) ||
                (published && result.source->mode == PrivateSourceMode::Retain &&
                 private_has_active_edge(capability.slot))) {
                throw std::logic_error("materialization private source result is invalid");
            }
        } else if (result.source) {
            throw std::logic_error("root materialization returned a private source result");
        }

        if (record->shared_source) {
            const CatalogCapability& capability = *record->shared_source;
            if (capability.owner.kind != LogicalOwnerKind::SharedPrefix ||
                capability.slot >= shared_catalog_count_) {
                throw std::logic_error("materialization shared source capability is malformed");
            }
            const SharedCatalogEntry& source       = shared_catalog_[capability.slot];
            const std::uint32_t current_references = shared_active_edge_count(capability.slot);
            if (!result.shared_source || source.transaction_pins == 0 ||
                source.id != capability.owner.id || source.revision != capability.generation ||
                current_references == std::numeric_limits<std::uint32_t>::max()) {
                throw std::logic_error("materialization shared source result is invalid");
            }
            const std::uint32_t expected_references =
                published ? current_references + 1U : current_references;
            if (result.shared_source->final_summary) {
                const SharedPrefixSummary& final = *result.shared_source->final_summary;
                SharedPrefixSummary expected     = source.summary;
                expected.active_references       = expected_references;
                // Source reuse may restore a replica, but it cannot rewrite the checkpoint
                // identity or its recovery contract.
                expected.checkpoint.state_residency = final.checkpoint.state_residency;
                if (!valid_shared_prefix_summary(final) || final != expected) {
                    throw std::logic_error("materialization shared source summary is invalid");
                }
            }
        } else if (result.shared_source) {
            throw std::logic_error("materialization returned an unexpected shared source result");
        }

        if (published) {
            const CatalogEntry& publication = catalog_[record->publication_slot];
            bool publication_released       = !publication.handle && publication.id == 0;
            publication_released =
                publication_released ||
                (record->private_source &&
                 record->publication_slot == record->private_source->slot && result.source &&
                 result.source->mode == PrivateSourceMode::ConsumeToActive);
            for (const OwnerClaim& claim : record->private_claims) {
                if (publication_released || claim.capability.slot != record->publication_slot) {
                    continue;
                }
                const auto& victim = private_result_for(claim);
                publication_released =
                    victim.disposition == VictimDisposition::Evicted && victim.pressure_committed;
            }
            if (!publication_released) {
                throw std::logic_error(
                    "materialization result cannot release its publication cell");
            }
        }

        if (published) { observe_selected_hit(*record); }
        for (const OwnerClaim& claim : record->private_claims) {
            apply_private_action(claim, published, private_result_for(claim));
        }
        for (const OwnerClaim& claim : record->shared_claims) {
            apply_shared_action(claim, published, shared_result_for(claim));
        }

        bool retained_private_source = false;
        if (record->private_source) {
            const CatalogCapability capability = *record->private_source;
            CatalogEntry& source               = catalog_[capability.slot];
            if (result.source->mode == PrivateSourceMode::Retain) {
                if (result.source->final_summary) {
                    assign_continuation_summary(source.summary, *result.source->final_summary);
                    migrate_observations(source, *result.source->final_summary, source.retention);
                    advance_revision(source.revision);
                    refresh_session_owner_revision(capability.owner.id, capability.slot,
                                                   source.revision);
                }
                source.state            = CatalogState::Catalogued;
                retained_private_source = result.status == ContextTransactionStatus::Published;
            } else if (result.source->mode == PrivateSourceMode::ConsumeToActive) {
                erase_session_if_owner(source.id);
                source.handle.reset();
                source.summary.endpoint.reset();
                source.summary.rewrite.reset();
                source.summary.long_anchors.clear();
                source.observations.clear();
                source.session.reset();
            }
        }

        if (record->shared_source) {
            SharedCatalogEntry& source = shared_catalog_[record->shared_source->slot];
            --source.transaction_pins;
            if (result.shared_source->final_summary) {
                SharedPrefixSummary updated = *result.shared_source->final_summary;
                updated.active_references   = 0;
                if (updated != source.summary) {
                    source.summary = std::move(updated);
                    advance_revision(source.revision);
                }
            }
        }

        observe_transfers(result);
        observe_operations(result);

        if (result.status == ContextTransactionStatus::Aborted) {
            restore_unreported_materialization(*record);
            lanes_[record->destination.value] = LogicalLaneState::Free;
            transaction_.template emplace<std::monostate>();
            program.finalize_context_transaction();
            return {.status = ContextTransactionStatus::Aborted};
        }

        CatalogEntry& publication = catalog_[record->publication_slot];
        publication.state         = CatalogState::ReservedForActive;
        publication.id            = next_continuation_id_++;
        publication.session       = record->session;
        publication.retention     = record->retention;
        advance_revision(publication.revision);
        if (publication.id == 0) { publication.id = next_continuation_id_++; }

        ActiveEntry& active         = active_[record->destination.value];
        active.occupied             = true;
        active.publication_slot     = record->publication_slot;
        active.continuation_id      = publication.id;
        active.session              = record->session;
        active.retention            = record->retention;
        active.update_session_index = record->update_session_index;
        active.publication_order    = record->publication_order;
        if (retained_private_source) {
            active.retained_private_source =
                active_edge(private_capability(record->private_source->slot));
        }
        if (record->shared_source) {
            active.shared_sources.push_back(
                active_edge(shared_capability(record->shared_source->slot)));
        }
        StartResult start = std::move(*result.published);
        result.published.reset();
        commit_demand(std::move(record->demand));
        return MaterializationOutcome{
            .status      = ContextTransactionStatus::Published,
            .activation  = PublishedActivation(*this, std::move(start), record->destination),
            .diagnostics = record->diagnostics,
        };
    }

    // 吸收一次捕获的终态：同一套对账骨架，但把"发布格"换成共享目录格，且多一种结局——被替换的那个
    // 共享条目要么已被换掉、要么原样保留，两种都要如实反映到账上。
    [[nodiscard]] ActiveCaptureOutcome
    adopt_active_capture_progress(Program& program, ProgramActiveCaptureResult&& result) {
        ActiveCaptureRecord* record = std::get_if<ActiveCaptureRecord>(&transaction_);
        if (record == nullptr || !program.has_context_transaction()) {
            throw std::logic_error("active capture result has no logical transaction");
        }
        if (result.status == ContextTransactionStatus::InProgress) {
            throw std::logic_error("terminal capture result is marked in progress");
        }
        if (result.victims.size() != record->private_claims.size() ||
            result.shared_victims.size() != record->shared_claims.size()) {
            throw std::logic_error("active capture result is not pressure-action aligned");
        }
        const auto private_result_for = [&](const OwnerClaim& claim) -> const auto& {
            const auto found =
                std::find_if(result.victims.begin(), result.victims.end(),
                             [&](const auto& row) { return row.owner == claim.planning_id; });
            if (found == result.victims.end() ||
                std::find_if(found + 1, result.victims.end(), [&](const auto& row) {
                    return row.owner == claim.planning_id;
                }) != result.victims.end()) {
                throw std::logic_error("active capture private result ID is not unique");
            }
            return *found;
        };
        const auto shared_result_for = [&](const OwnerClaim& claim) -> const auto& {
            const auto found =
                std::find_if(result.shared_victims.begin(), result.shared_victims.end(),
                             [&](const auto& row) { return row.owner == claim.planning_id; });
            if (found == result.shared_victims.end() ||
                std::find_if(found + 1, result.shared_victims.end(), [&](const auto& row) {
                    return row.owner == claim.planning_id;
                }) != result.shared_victims.end()) {
                throw std::logic_error("active capture shared result ID is not unique");
            }
            return *found;
        };
        const bool published = result.status == ContextTransactionStatus::Published;
        if (!published && result.status != ContextTransactionStatus::Aborted) {
            throw std::logic_error("active capture terminal status is invalid");
        }
        for (std::size_t row = 0; row < record->private_claims.size(); ++row) {
            const OwnerClaim& claim = record->private_claims[row];
            if (std::find_if(record->private_claims.begin(),
                             record->private_claims.begin() + static_cast<std::ptrdiff_t>(row),
                             [&](const OwnerClaim& prior) {
                                 return prior.planning_id == claim.planning_id ||
                                        prior.capability == claim.capability;
                             }) !=
                record->private_claims.begin() + static_cast<std::ptrdiff_t>(row)) {
                throw std::logic_error("active capture private claim manifest is not unique");
            }
        }
        for (std::size_t row = 0; row < record->shared_claims.size(); ++row) {
            const OwnerClaim& claim = record->shared_claims[row];
            if (std::any_of(record->private_claims.begin(), record->private_claims.end(),
                            [&](const OwnerClaim& prior) {
                                return prior.planning_id == claim.planning_id;
                            })) {
                throw std::logic_error("active capture owner ID changes kind");
            }
            if (claim.capability.slot == record->publication_slot ||
                std::find_if(record->shared_claims.begin(),
                             record->shared_claims.begin() + static_cast<std::ptrdiff_t>(row),
                             [&](const OwnerClaim& prior) {
                                 return prior.planning_id == claim.planning_id ||
                                        prior.capability == claim.capability;
                             }) !=
                    record->shared_claims.begin() + static_cast<std::ptrdiff_t>(row)) {
                throw std::logic_error("active capture shared claim manifest is not unique");
            }
        }
        for (const OwnerClaim& claim : record->private_claims) {
            validate_private_action(claim, published, private_result_for(claim));
        }
        for (const OwnerClaim& claim : record->shared_claims) {
            validate_shared_action(claim, published, shared_result_for(claim));
        }
        if (record->lane.value >= lane_count_ || !active_[record->lane.value].occupied ||
            lanes_[record->lane.value] != LogicalLaneState::Active) {
            throw std::logic_error("active capture owner left its lane");
        }
        if (record->publishes_shared) {
            if (record->publication_slot >= shared_catalog_count_) {
                throw std::logic_error("active capture shared publication slot is invalid");
            }
            const SharedCatalogEntry& publication = shared_catalog_[record->publication_slot];
            if (publication.state != SharedCatalogState::ReservedCapture ||
                shared_active_edge_count(record->publication_slot) != 0 ||
                (record->replacement_id == 0 && (publication.id != 0 || publication.handle ||
                                                 result.capacity_preparation_committed)) ||
                (record->replacement_id != 0 &&
                 (publication.id != record->replacement_id ||
                  publication.revision != record->replacement_revision || !publication.handle ||
                  (published && !result.capacity_preparation_committed)))) {
                throw std::logic_error("active capture publication changed before adoption");
            }
        } else if (record->publication_slot != kInvalidCatalogSlot || record->replacement_id != 0 ||
                   result.capacity_preparation_committed) {
            throw std::logic_error("private capture has shared publication state");
        }
        if (!published) {
            if (result.shared) {
                throw std::logic_error("aborted active capture published a shared prefix");
            }
        } else {
            const ActiveEntry& active = active_[record->lane.value];
            if (record->publishes_private) {
                if (!valid_continuation_summary(result.active_summary)) {
                    throw std::logic_error("active capture returned an invalid private summary");
                }
                const CatalogEntry& publication = catalog_.at(active.publication_slot);
                if (publication.state != CatalogState::ReservedForActive ||
                    publication.id != active.continuation_id) {
                    throw std::logic_error("active private publication changed during capture");
                }
            }
            if (record->publishes_shared) {
                if (!result.shared ||
                    active.shared_sources.size() == active.shared_sources.capacity() ||
                    !valid_shared_prefix_summary(result.shared->summary) ||
                    result.shared->summary.active_references != 1) {
                    throw std::logic_error("active capture returned an invalid shared publication");
                }
            } else if (result.shared) {
                throw std::logic_error("private capture returned an unexpected shared publication");
            }
        }
        for (const OwnerClaim& claim : record->private_claims) {
            apply_private_action(claim, published, private_result_for(claim));
        }
        for (const OwnerClaim& claim : record->shared_claims) {
            apply_shared_action(claim, published, shared_result_for(claim));
        }
        if (result.status == ContextTransactionStatus::Aborted) {
            if (record->publication_slot != kInvalidCatalogSlot) {
                SharedCatalogEntry& publication = shared_catalog_[record->publication_slot];
                if (record->replacement_id != 0 && result.capacity_preparation_committed) {
                    clear_shared_entry(publication);
                } else if (record->replacement_id != 0) {
                    publication.state = SharedCatalogState::Catalogued;
                } else {
                    clear_shared_entry(publication);
                }
            }
            observe_transfers(result);
            observe_operations(result);
            transaction_.template emplace<std::monostate>();
            program.finalize_context_transaction();
            return {.status = ContextTransactionStatus::Aborted};
        }
        ActiveEntry& active = active_[record->lane.value];
        if (record->publishes_private) {
            CatalogEntry& publication = catalog_[active.publication_slot];
            assign_continuation_summary(publication.summary, result.active_summary);
            migrate_observations(publication, result.active_summary, active.retention);
            advance_revision(publication.revision);
        }
        if (record->publishes_shared) {
            SharedCatalogEntry& publication = shared_catalog_[record->publication_slot];
            publication.handle.reset();
            publication.state = SharedCatalogState::Catalogued;
            publication.id    = next_shared_prefix_id_++;
            if (publication.id == 0) { publication.id = next_shared_prefix_id_++; }
            publication.summary                   = result.shared->summary;
            publication.summary.active_references = 0;
            publication.handle.emplace(std::move(result.shared->handle));
            publication.observation =
                RetentionObservation{.retention_class = RetentionClass::SharedStable};
            publication.transaction_pins = 0;
            publication.explicit_credit =
                has_shared_candidate_evidence(record->shared_evidence,
                                              SharedCandidateEvidence::ExplicitBoundary) ||
                has_shared_candidate_evidence(record->shared_evidence,
                                              SharedCandidateEvidence::RequestedAutomatic);
            publication.credit_expiry_epoch =
                publication.explicit_credit
                    ? (demand_epoch_ >
                               std::numeric_limits<std::uint64_t>::max() - kDemandWindowCapacity
                           ? std::numeric_limits<std::uint64_t>::max()
                           : demand_epoch_ + kDemandWindowCapacity)
                    : 0;
            advance_revision(publication.revision);
            active.shared_sources.push_back(
                active_edge(shared_capability(record->publication_slot)));
        }
        observe_transfers(result);
        observe_operations(result);
        transaction_.template emplace<std::monostate>();
        program.finalize_context_transaction();
        return {.status = ContextTransactionStatus::Published};
    }

    // 交还某条 lane 压着的所有租约。先逐条验它们仍然指向存活的条目（租约绑身份不绑版本，但条目必须还在），
    // 验完再清——租约是"别人不能动这个条目"的唯一依据，静默丢掉会让活跃请求的来源被误牺牲。
    void release_active_references(LaneId lane) {
        ActiveEntry& active = active_[lane.value];
        if (active.retained_private_source) {
            const ActiveOwnerEdge& edge = *active.retained_private_source;
            if (edge.owner.kind != LogicalOwnerKind::PrivateContinuation ||
                edge.slot >= catalog_count_) {
                throw std::logic_error("retained private source edge is malformed");
            }
            const CatalogEntry& source = catalog_[edge.slot];
            if (source.state != CatalogState::Catalogued || !source.handle ||
                source.id != edge.owner.id) {
                throw std::logic_error("retained private source edge is stale");
            }
        }
        for (std::size_t index = 0; index < active.shared_sources.size(); ++index) {
            const ActiveOwnerEdge& edge = active.shared_sources[index];
            if (edge.owner.kind != LogicalOwnerKind::SharedPrefix ||
                edge.slot >= shared_catalog_count_ ||
                std::find(active.shared_sources.begin(), active.shared_sources.begin() + index,
                          edge) != active.shared_sources.begin() + index) {
                throw std::logic_error("shared source edge is malformed");
            }
            const SharedCatalogEntry& source = shared_catalog_[edge.slot];
            if (source.state != SharedCatalogState::Catalogued || !source.handle ||
                source.id != edge.owner.id) {
                throw std::logic_error("shared source edge is stale");
            }
        }
        active.retained_private_source.reset();
        active.shared_sources.clear();
    }

    // 取消一条 lane 并释放它的全部逻辑资源：租约、发布格、ActiveEntry，最后回 Free。
    void release_cancelled_lane(LaneId lane) {
        if (lane.value >= lane_count_ || !active_[lane.value].occupied ||
            (lanes_[lane.value] != LogicalLaneState::Active &&
             lanes_[lane.value] != LogicalLaneState::TerminalPending)) {
            throw std::logic_error("cancelled lane has no logical active owner");
        }
        release_active_references(lane);
        clear_catalog_entry(catalog_.at(active_[lane.value].publication_slot));
        reset_active_entry(active_[lane.value]);
        lanes_[lane.value] = LogicalLaneState::Free;
    }

    [[nodiscard]] static std::uint64_t session_hash(const CacheSessionKey& key) noexcept {
        std::uint64_t hash = 1469598103934665603ULL;
        for (const unsigned char value : key.view()) {
            hash ^= value;
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    // 会话索引：开放寻址的哈希表。查找在 Empty 处终止、在 Deleted 墓碑处继续探测（墓碑不能提前终结
    // 探测链，否则会漏掉后面的同键条目）。这一族函数的职责只有一个：把"同一个会话"稳定地映射到
    // 当前代表它的那个续跑点，从而让多轮对话能复用上一轮的状态。
    [[nodiscard]] std::optional<std::size_t>
    find_session_cell(const CacheSessionKey& key) const noexcept {
        if (session_index_.empty()) { return std::nullopt; }
        const std::size_t begin = session_hash(key) % session_index_.size();
        for (std::size_t probe = 0; probe < session_index_.size(); ++probe) {
            const std::size_t cell         = (begin + probe) % session_index_.size();
            const SessionIndexEntry& entry = session_index_[cell];
            if (entry.state == SessionIndexState::Empty) { return std::nullopt; }
            if (entry.state == SessionIndexState::Occupied && entry.key == key) { return cell; }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::size_t session_insert_cell(const CacheSessionKey& key) const {
        if (session_index_.empty()) {
            throw std::logic_error("session publication has no index capacity");
        }
        const std::size_t begin = session_hash(key) % session_index_.size();
        std::optional<std::size_t> deleted;
        for (std::size_t probe = 0; probe < session_index_.size(); ++probe) {
            const std::size_t cell         = (begin + probe) % session_index_.size();
            const SessionIndexEntry& entry = session_index_[cell];
            if (entry.state == SessionIndexState::Occupied && entry.key == key) { return cell; }
            if (entry.state == SessionIndexState::Deleted && !deleted) { deleted = cell; }
            if (entry.state == SessionIndexState::Empty) { return deleted.value_or(cell); }
        }
        if (deleted) { return *deleted; }
        throw std::logic_error("session index is full");
    }

    void erase_session_if_owner(std::uint64_t owner_id) noexcept {
        if (owner_id == 0) { return; }
        for (SessionIndexEntry& entry : session_index_) {
            if (entry.state == SessionIndexState::Occupied && entry.owner_id == owner_id) {
                entry.state             = SessionIndexState::Deleted;
                entry.slot              = kInvalidCatalogSlot;
                entry.owner_id          = 0;
                entry.revision          = 0;
                entry.publication_order = 0;
            }
        }
    }

    void refresh_session_owner_revision(std::uint64_t owner_id, std::uint32_t slot,
                                        std::uint64_t revision) noexcept {
        if (owner_id == 0 || slot >= catalog_count_) { return; }
        for (SessionIndexEntry& entry : session_index_) {
            if (entry.state == SessionIndexState::Occupied && entry.owner_id == owner_id &&
                entry.slot == slot) {
                entry.revision = revision;
                return;
            }
        }
    }

    // 发布会话绑定。publication_order 是**新旧裁决**依据：更新的发布序可以顶掉旧的（旧的被"降级"，
    // 见 demote_replaced_session），更旧的发布序则直接不采纳。相同发布序却指向不同续跑点属于矛盾，
    // 抛错——那意味着同一轮里出现了两个自称最新的绑定。
    [[nodiscard]] bool publish_session(const CacheSessionKey& key, std::uint32_t slot,
                                       std::uint64_t owner_id, std::uint64_t revision,
                                       std::uint64_t publication_order) {
        const std::size_t cell   = session_insert_cell(key);
        SessionIndexEntry& entry = session_index_[cell];
        std::optional<SessionIndexEntry> previous;
        if (entry.state == SessionIndexState::Occupied) {
            if (entry.publication_order > publication_order) { return false; }
            if (entry.publication_order == publication_order) {
                if (entry.slot != slot || entry.owner_id != owner_id) {
                    throw std::logic_error("equal publication order names two continuations");
                }
                entry.revision = revision;
                return true;
            }
            previous = entry;
        }
        entry = SessionIndexEntry{
            .state             = SessionIndexState::Occupied,
            .key               = key,
            .slot              = slot,
            .owner_id          = owner_id,
            .revision          = revision,
            .publication_order = publication_order,
        };
        if (previous && (previous->slot != slot || previous->owner_id != owner_id)) {
            demote_replaced_session(*previous, slot, owner_id);
        }
        return true;
    }

    // 被顶掉的旧绑定降级为普通私有续跑点：条目本身**不删除**（它仍是有效缓存），只是不再享受
    // "会话当前代表"的身份，保留类别也退回 RecentPrivate。
    void demote_replaced_session(const SessionIndexEntry& previous, std::uint32_t replacement_slot,
                                 std::uint64_t replacement_id) noexcept {
        if (previous.slot >= catalog_count_ ||
            (previous.slot == replacement_slot && previous.owner_id == replacement_id)) {
            return;
        }
        CatalogEntry& prior = catalog_[previous.slot];
        if (prior.state != CatalogState::Catalogued || !prior.handle ||
            prior.id != previous.owner_id || prior.revision != previous.revision) {
            return;
        }
        prior.session.reset();
        prior.retention = RetentionClass::RecentPrivate;
        for (CheckpointObservation& observation : prior.observations) {
            observation.observation.retention_class = RetentionClass::RecentPrivate;
        }
    }

    // 搬运观测：Program 只报"搬了什么、多少、多久"，分类累加属于逻辑层的活。因此分页/分字节/分方向
    // 的统计都在这里成形，而不是让 Program 维护一堆计数器。
    void observe_transfer(const ContextTransferObservation& observation) noexcept {
        const double seconds = static_cast<double>(observation.elapsed_ns) * 1.0e-9;
        context_stats_.actual_context_transfer_seconds += seconds;
        const std::uint64_t bytes = observation.units;
        switch (observation.resource) {
        case ContextResourceClass::State:
            switch (observation.direction) {
            case ContextTransferDirection::DeviceToHost:
                ++context_stats_.state_d2h_count;
                context_stats_.state_d2h_bytes += bytes;
                context_stats_.state_d2h_seconds += seconds;
                break;
            case ContextTransferDirection::HostToDevice:
                ++context_stats_.state_h2d_count;
                context_stats_.state_h2d_bytes += bytes;
                context_stats_.state_h2d_seconds += seconds;
                break;
            case ContextTransferDirection::DeviceToDevice:
                ++context_stats_.state_d2d_count;
                context_stats_.state_d2d_bytes += bytes;
                context_stats_.state_d2d_seconds += seconds;
                break;
            }
            break;
        case ContextResourceClass::MainKV:
            observe_kv_transfer(
                observation, context_stats_.main_kv_d2h_pages, context_stats_.main_kv_h2d_pages,
                context_stats_.main_kv_d2d_pages, context_stats_.main_kv_d2h_bytes,
                context_stats_.main_kv_h2d_bytes, context_stats_.main_kv_d2d_bytes,
                context_stats_.main_kv_d2h_seconds, context_stats_.main_kv_h2d_seconds,
                context_stats_.main_kv_d2d_seconds);
            break;
        case ContextResourceClass::BackendKV:
            observe_kv_transfer(
                observation, context_stats_.backend_kv_d2h_pages,
                context_stats_.backend_kv_h2d_pages, context_stats_.backend_kv_d2d_pages,
                context_stats_.backend_kv_d2h_bytes, context_stats_.backend_kv_h2d_bytes,
                context_stats_.backend_kv_d2d_bytes, context_stats_.backend_kv_d2h_seconds,
                context_stats_.backend_kv_h2d_seconds, context_stats_.backend_kv_d2d_seconds);
            break;
        }
    }

    static void observe_kv_transfer(const ContextTransferObservation& observation,
                                    std::uint64_t& d2h_pages, std::uint64_t& h2d_pages,
                                    std::uint64_t& d2d_pages, std::uint64_t& d2h_bytes,
                                    std::uint64_t& h2d_bytes, std::uint64_t& d2d_bytes,
                                    double& d2h_seconds, double& h2d_seconds,
                                    double& d2d_seconds) noexcept {
        const double seconds = static_cast<double>(observation.elapsed_ns) * 1.0e-9;
        switch (observation.direction) {
        case ContextTransferDirection::DeviceToHost:
            d2h_pages += observation.page_count;
            d2h_bytes += observation.units;
            d2h_seconds += seconds;
            break;
        case ContextTransferDirection::HostToDevice:
            h2d_pages += observation.page_count;
            h2d_bytes += observation.units;
            h2d_seconds += seconds;
            break;
        case ContextTransferDirection::DeviceToDevice:
            d2d_pages += observation.page_count;
            d2d_bytes += observation.units;
            d2d_seconds += seconds;
            break;
        }
    }

    template <class Result>
    void observe_transfers(const Result& result) noexcept {
        for (const ContextTransferObservation& observation : result.transfer_observations) {
            observe_transfer(observation);
        }
    }

    template <class Result>
    void observe_operations(const Result& result) noexcept {
        context_stats_.state_moves += result.operations.state_moves;
        context_stats_.state_forks += result.operations.state_forks;
        context_stats_.state_restores += result.operations.state_restores;
        context_stats_.pressure_spill_pages += result.operations.pressure_spill_pages;
        context_stats_.partial_tail_cow_pages += result.operations.partial_tail_cow_pages;
        context_stats_.historical_fork_hits += result.operations.historical_fork_hits;
    }

    // ---- 状态 ----
    //
    // 事实来源（前三组）：逻辑 lane、两个目录、当前事务；其余都是派生或统计。
    // session_index_ / prefix_index_ 随时可从目录重建；observation_scratch_ / demand_window_ 是
    // 容量固定的观测缓冲；两个 epoch 分别给保留策略与需求窗口提供单调时间轴。
    std::uint32_t lane_count_           = 0;
    std::uint32_t catalog_count_        = 0;
    std::uint32_t shared_catalog_count_ = 0;
    bool cache_enabled_                 = true;
    std::array<LogicalLaneState, kMaximumConcurrency> lanes_{};
    std::vector<CatalogEntry> catalog_;
    std::vector<SharedCatalogEntry> shared_catalog_;
    std::vector<SessionIndexEntry> session_index_;
    std::vector<PrefixIndexEntry> prefix_index_;
    std::vector<CheckpointObservation> observation_scratch_;
    std::vector<PrefixDemandRecord> demand_window_;
    std::uint32_t max_long_anchors_ = 0;
    std::array<ActiveEntry, kMaximumConcurrency> active_{};
    using ContextTransaction =
        std::variant<std::monostate, MaterializationRecord, ActiveCaptureRecord>;
    ContextTransaction transaction_;
    ContextMachineCostModel cost_model_;
    Planner planner_;
    CapturePlanner capture_planner_;
    RuntimeStats context_stats_;
    std::uint64_t next_continuation_id_  = 1;
    std::uint64_t next_shared_prefix_id_ = 1;
    std::uint64_t retention_epoch_       = 0;
    std::uint64_t demand_epoch_          = 0;
};

} // namespace ninfer::runtime
