#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/program_impl.h"
#include <stdexcept>
#include <utility>

// ============================================================================
// models/qwen3_5/program/program.cpp —— Program 契约面的"薄壳"实现
// ============================================================================
//
// 这个文件里没有物理、没有策略。program.h 把方向讲清楚之后，本文件只做三件事：
//   1. 拆包：把调用方手里的模型侧视图（PreparedPrompt 的 view / take）转成实现层认得的形式；
//   2. 转发：调用 detail::*Impl 上真正干活的那一层；
//   3. 装包：把实现层的产物重新包成对外的能力或计划（ResourcePlan、CapturePressurePlan、
//      PrefillProgress…），顺手把该绑的 revision 绑上。
// 所以本文件的注释只标注"边界上发生了什么"；算法、状态机与物理细节在各自的 impl 与 transactions/ 里。
//
// 边界上真正有分量的只有两类判断：
//   * **revision 闸门**：凡是能把"已封印的方案"变成物理副作用的入口（start_resource_transaction、
//     reserve_active_capture_with_pressure、prove_persistent_backfill），都先拿 plan 上的
//     resource_revision 与当前值比对；对不上就返回 Aborted / nullopt，绝不带着过期方案往下走。
//   * **所有权口径**：prompt 是只读观察（view）还是被吃掉（take），决定这次调用之后调用方还能不能用它。
// 其余是纯转发，不再逐条注释。
//
// 文件分两段：前半是契约面方法，后半是几个小值类型（AdmissionCandidate / CapturePressureCandidate /
// CaptureAssessment）的 out-of-line 定义——它们必须写在这里，因为 PIMPL 的移动与析构要求完整类型。

namespace ninfer::models::qwen3_5 {

// ---- 规划：SequencePlan（定稿方案）与 SequencePlanner（方案生成器）----
//
// 分工：规划器可以反复迭代（试不同切分、问容量曲线），但它是**一次性**的——finalize 是 && 限定，
// 收尾即消费，产出的 SequencePlan 才是那个不可变的方案；Program 出生时就带着它。
// 空壳（被移动走之后）在这组接口上返回中性值（0 / 静态空对象），而不是抛异常：查询侧不该因为
// "这东西已经交出去了"就炸掉调用方。真正会抛的是动作侧（finalize）。

SequencePlan::SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlan::SequencePlan(SequencePlan&&) noexcept = default;

SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;

SequencePlan::~SequencePlan() = default;

std::uint32_t SequencePlan::capacity() const noexcept {
    return impl_ != nullptr ? impl_->capacity : 0;
}

std::uint32_t SequencePlan::kv_capacity() const noexcept {
    return impl_ != nullptr ? impl_->kv_capacity : 0;
}

std::uint32_t SequencePlan::max_concurrency() const noexcept {
    return impl_ != nullptr ? impl_->max_concurrency : 0;
}

std::size_t SequencePlan::device_reservation_bytes() const noexcept {
    return impl_ != nullptr ? impl_->device_reservation_bytes : 0;
}

std::size_t SequencePlan::workspace_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->workspace.capacity : 0;
}

SequencePlanner::SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlanner::SequencePlanner(SequencePlanner&&) noexcept = default;

SequencePlanner& SequencePlanner::operator=(SequencePlanner&&) noexcept = default;

SequencePlanner::~SequencePlanner() = default;

// 容量曲线是**规划期**的事实：还没定稿时就要知道在不同并发/页组合下能装多少，规划器据此试算。
// 定稿之后这条曲线就不再被查询——真实容量以 Program 的物理状态为准。
const runtime::SequenceCapacityCurve& SequencePlanner::capacity_curve() const noexcept {
    static const runtime::SequenceCapacityCurve empty;
    return impl_ != nullptr ? impl_->curve : empty;
}

// 收尾规划（&&：只能对右值调用，调用即消费规划器）。main_page_groups 是**唯一**留到这一步才定的参数：
// 它决定最终页布局，此前的一切试算都围着它可变。
SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    return SequencePlan(detail::finalize_sequence_plan_impl(std::move(impl_), main_page_groups));
}

// ---- 请求基线：RequestBasePlan ----
//
// 它是"这条 prompt 按这套配置跑起来物理上是什么样"的答案，不绑定 lane、也不绑定复用来源——
// 正因如此它可以被重复用于多次准入探查（换 target、换来源），而不必每次重算。
// 对外只给三样：摘要、它自带的前缀缓存、以及按前沿取的短名单提示。

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept = default;

RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;

RequestBasePlan::~RequestBasePlan() = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PreparedContextCache& RequestBasePlan::context_cache() const noexcept {
    static const PreparedContextCache empty;
    return impl_ != nullptr ? impl_->context_cache : empty;
}

// 取某个前沿的前缀摘要，用来把 catalog 的检查范围缩小。frontier == 0 或越界返回 nullopt——那是
// "这个前沿没有可比的前缀"，不是错误。方向不变：短名单只负责排序与缩小范围，命不命中由后续的精确比对决定。
std::optional<PrefixShortlistKey>
RequestBasePlan::prefix_shortlist_key(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr || frontier == 0 || frontier > impl_->prefix_digests.size()) {
        return std::nullopt;
    }
    return PrefixShortlistKey{
        .digests      = impl_->prefix_digests.at(frontier),
        .frontier     = frontier,
        .identity_tag = impl_->prefix_identity_tag,
    };
}

// 如果这个前沿上**确实**存在可用的共享候选，回报重建它所需的真实 prefill 工作量；找不到就 nullopt。
// 它是给 Runtime 做成本比较用的：复用共享前缀省下的，正是这份重建工作。
std::optional<runtime::PrefillWork>
RequestBasePlan::shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr) { return std::nullopt; }
    const auto found = std::find_if(impl_->shared_candidates.begin(),
                                    impl_->shared_candidates.end(), [&](const auto& candidate) {
                                        return candidate.frontier == frontier && candidate.identity;
                                    });
    return found == impl_->shared_candidates.end()
               ? std::nullopt
               : std::optional<runtime::PrefillWork>(found->identity->rebuild_work);
}

// ---- 压力会话：把 program.h 的会话契约接到 PressurePlanningSessionImpl 上 ----
//
// 这一组几乎全是转发，但空壳（被移动走之后）的待遇并不统一，值得记住：
//   * 多数入口会检查并抛 logic_error——会话已经交出去了，再用属于调用方违约；
//   * 少数纯转发的入口不检查（直接解引用），用空壳调它们同样是调用方违约，只是不会得到明确报错；
//   * discard_expansion 在空壳上是 no-op：它属于清理路径，清理动作不该在清理路径上再抛一次异常。
// 另外注意 target 的三种取法已经固定了探索的起点：identity（候选自身）、root_maximal（根候选的
// 最大可达）、maximal（某候选的最大可达）——含义由 impl 定义，这里只负责转交。

PressurePlanningSession::PressurePlanningSession(
    std::unique_ptr<detail::PressurePlanningSessionImpl> impl) noexcept
    : impl_(std::move(impl)) {}

PressurePlanningSession::PressurePlanningSession(PressurePlanningSession&&) noexcept = default;

PressurePlanningSession&
PressurePlanningSession::operator=(PressurePlanningSession&&) noexcept = default;

PressurePlanningSession::~PressurePlanningSession() = default;

CapturePressurePlanningSession::CapturePressurePlanningSession(
    CapturePressurePlanningSession&&) noexcept = default;

CapturePressurePlanningSession&
CapturePressurePlanningSession::operator=(CapturePressurePlanningSession&&) noexcept = default;

CapturePressurePlanningSession::~CapturePressurePlanningSession() = default;

PressureTargetHandle
PressurePlanningSession::identity_target(runtime::PlanningCandidateId candidate) const {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->identity_target(candidate);
}

PressureTargetHandle
PressurePlanningSession::root_maximal_target(runtime::PlanningCandidateId root_candidate) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->root_maximal_target(root_candidate);
}

PressureTargetHandle
PressurePlanningSession::maximal_target(runtime::PlanningCandidateId candidate) {
    return impl_->maximal_target(candidate);
}

PressureConstructionCursor PressurePlanningSession::begin_construction(PressureTargetHandle target,
                                                                       bool restore) {
    return impl_->begin_construction(target, restore);
}

runtime::PressureConstructionStep
PressurePlanningSession::next_construction_option(PressureConstructionCursor& cursor) {
    return impl_->next_construction_option(cursor);
}

void PressurePlanningSession::choose_construction(PressureConstructionCursor& cursor,
                                                  runtime::PressureConstructionOptionId option) {
    impl_->choose_construction(cursor, option);
}

std::optional<PressureTargetHandle>
PressurePlanningSession::construction_target(const PressureConstructionCursor& cursor) {
    return impl_->construction_target(cursor);
}

runtime::PressureTargetGuidance PressurePlanningSession::guidance(PressureTargetHandle target) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->guidance(target);
}

AssessedPressureTarget PressurePlanningSession::assess(PressureTargetHandle target) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->assess(target);
}

PreparedPressureExpansion PressurePlanningSession::prepare_expansion(PressureTargetHandle parent,
                                                                     std::uint32_t maximum_owners) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->prepare_expansion(parent, maximum_owners);
}

PressureExpansionView
PressurePlanningSession::commit_expansion(PreparedPressureExpansion&& prepared) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->commit_expansion(std::move(prepared));
}

void PressurePlanningSession::discard_expansion(PreparedPressureExpansion&& prepared) noexcept {
    if (impl_ != nullptr) { impl_->discard_expansion(std::move(prepared)); }
}

runtime::PrefillWork PressurePlanningSession::shared_capture_split_prefill_work(
    const AssessedPressureTarget& assessed, const PreparedPrompt& prompt,
    std::span<const std::uint32_t> frontiers) const {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    return impl_->shared_capture_split_prefill_work(assessed, PreparedPromptAccess::view(prompt),
                                                    frontiers);
}

// 封印：这一步把"评估过的 target"变成"可启动的方案"，同时把两样东西绑死在方案上——
// 会话当时的 resource_revision（start_resource_transaction 会拿它验门），以及 needs_transfer
// （来源在别的设备上时，启动必须知道要做一次搬运）。
// 拿不到封印结果就返回 nullopt：目标不可行是正常答案，不是异常。
std::optional<ResourcePlan> PressurePlanningSession::seal(AssessedPressureTarget&& assessed,
                                                          const PreparedPrompt& prompt,
                                                          runtime::FinalScheduleIntent intent) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    std::optional<AdmissionCandidate> sealed =
        impl_->seal(std::move(assessed), PreparedPromptAccess::view(prompt), intent);
    if (!sealed) { return std::nullopt; }
    const bool needs_transfer = sealed->impl_->needs_transfer;
    return ResourcePlan(std::move(*sealed), impl_->resource_revision, needs_transfer);
}

// capture 专用的封印出口：产物是 CapturePressurePlan 而**不是** ResourcePlan——捕获的压力方案不能
// 被拿去启动一条请求，类型上就分开。（这条出口只给 CapturePressurePlanningSession 用。）
std::optional<CapturePressurePlan>
PressurePlanningSession::seal_capture(AssessedPressureTarget&& assessed) {
    if (impl_ == nullptr) { throw std::logic_error("pressure planning session is empty"); }
    std::optional<CapturePressureCandidate> sealed = impl_->seal_capture(std::move(assessed));
    if (!sealed) { return std::nullopt; }
    return CapturePressurePlan(std::move(*sealed), impl_->resource_revision);
}

// ---- 捕获压力会话：只是把同一个会话限制在"唯一候选"上 ----
//
// 这一组全部转发给内部那个普通会话，唯一的原则性差别在两头：
//   * 入口 identity_target() 不接受候选参数——候选被固定在会话对象内部（candidate_），
//     并且**由会话拥有一条 0 号 id**；调用方拿不到别的候选，也就无法把捕获当成请求去用。
//   * 出口只有 seal()，且它走的是 seal_capture，产物是 CapturePressurePlan。

PressureTargetHandle CapturePressurePlanningSession::identity_target() const {
    if (!candidate_.impl_) { throw std::logic_error("capture pressure candidate is empty"); }
    return session_.identity_target(candidate_id());
}

runtime::PressureTargetGuidance
CapturePressurePlanningSession::guidance(PressureTargetHandle target) {
    return session_.guidance(target);
}

AssessedPressureTarget CapturePressurePlanningSession::assess(PressureTargetHandle target) {
    return session_.assess(target);
}

PreparedPressureExpansion
CapturePressurePlanningSession::prepare_expansion(PressureTargetHandle parent) {
    return session_.prepare_expansion(parent);
}

PressureExpansionView
CapturePressurePlanningSession::commit_expansion(PreparedPressureExpansion&& prepared) {
    return session_.commit_expansion(std::move(prepared));
}

void CapturePressurePlanningSession::discard_expansion(
    PreparedPressureExpansion&& prepared) noexcept {
    session_.discard_expansion(std::move(prepared));
}

std::optional<CapturePressurePlan>
CapturePressurePlanningSession::seal(AssessedPressureTarget&& assessed) {
    return session_.seal_capture(std::move(assessed));
}

// ---- Program 本体 ----
//
// 全部是转发，但转发**顺序**本身承载语义的地方有三处，值得单独说明：
//   * seal_identity：封印其实是三段——固化候选 → 挑出共享捕获前沿 → 重新校验。第三步不能省：
//     挑选会改变方案的实际形态，必须重新确认真实可行，仍然失败就整体返回 nullopt（不留半成品）。
//   * start_resource_transaction / reserve_active_capture_with_pressure：物理副作用之前的最后一道
//     revision 闸门。对不上返回 Aborted——是"过期了，请重新规划"，不是错误。
//   * begin_pressure_planning：把**物理指针**与 Runtime 的**逻辑 id** 两套并行数组在这里对齐后交给会话，
//     于是会话内部可以用逻辑 id 说话，而物理事实始终只在 Program 侧。
// Program 自身拷贝与移动都被删除，不存在"空壳"状态；个别入口里的 impl_ == nullptr 检查是防御性的。
Program::Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept : impl_(std::move(impl)) {}

Program::~Program() noexcept = default;

// 注意 prompt 的所有权口径贯穿整个契约面：
//   * 只读观察（view）：plan_request / inspect_admission / inspect_capture 等——调用后调用方仍持有 prompt；
//   * 被吃掉（take）：causal_score / start_resource_transaction——prompt 归 Program 所有。
// 这条口径就是 PreparedPromptAccess 存在的原因：跨层传引用会立刻破坏它。
RequestBasePlan Program::plan_request(const PreparedPrompt& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(PreparedPromptAccess::view(prompt), options);
}

std::vector<float> Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target) {
    return impl_->causal_score(PreparedPromptAccess::take(std::move(prompt)), first_target);
}

std::optional<AdmissionCandidate> Program::inspect_admission(
    const PreparedPrompt& prompt, const RequestBasePlan& base, runtime::LaneId destination,
    const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
    std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source) {
    return impl_->inspect_admission(PreparedPromptAccess::view(prompt), base, destination, source,
                                    shared_source, checkpoint, must_retain_private_source);
}

// 封印三段式（注意第三段）：先按"物化"固化候选（此处不带任何共享/checkpoint 偏好），再按调度意图
// 挑出可接受的共享捕获前沿，最后**重新校验**物化是否仍然成立。任何一步不成立就返回 nullopt——
// 选择与校验分开，是为了让"排序提示"永远不能直接把一个目标顶成可行。
std::optional<ResourcePlan> Program::seal_identity(const AdmissionCandidate& admission,
                                                   const PreparedPrompt& prompt,
                                                   runtime::FinalScheduleIntent intent) {
    std::optional<AdmissionCandidate> sealed = impl_->seal_materialization(
        admission, PreparedPromptAccess::view(prompt), {}, {}, {}, {}, {}, {});
    if (!sealed) { return std::nullopt; }
    impl_->select_shared_captures(*sealed, PreparedPromptAccess::view(prompt),
                                  intent.shared_capture_frontiers);
    if (impl_->revalidate_materialization(*sealed, PreparedPromptAccess::view(prompt)) !=
        runtime::PreflightStatus::Ready) {
        return std::nullopt;
    }
    const bool needs_transfer = sealed->impl_->needs_transfer;
    return ResourcePlan(std::move(*sealed), impl_->resource_revision(), needs_transfer);
}

// 开一场压力探索。两组入参是**成对**的：物理候选指针 + Runtime 的逻辑候选 id，私有/共享 donor 的
// 行为同理。这里把两者 zip 成会话内部的绑定，之后会话内部一律用逻辑 id 说话，物理事实只在 Program 侧。
// 候选为空即调用方违约（抛 invalid_argument）——空候选的会话没有任何意义。
PressurePlanningSession
Program::begin_pressure_planning(std::span<const AdmissionCandidate* const> candidates,
                                 std::span<const runtime::PlanningCandidateId> candidate_ids,
                                 std::span<const ContinuationHandle* const> private_owners,
                                 std::span<const runtime::PlanningOwnerId> private_owner_ids,
                                 std::span<const SharedPrefixHandle* const> shared_owners,
                                 std::span<const runtime::PlanningOwnerId> shared_owner_ids) {
    using SessionImpl = detail::PressurePlanningSessionImpl;
    std::vector<SessionImpl::PhysicalCandidateBinding> physical_candidates;
    physical_candidates.reserve(candidates.size());
    for (const AdmissionCandidate* candidate : candidates) {
        if (candidate == nullptr || candidate->impl_ == nullptr) {
            throw std::invalid_argument("pressure planning candidate is empty");
        }
        physical_candidates.push_back(SessionImpl::PhysicalCandidateBinding{
            .state     = candidate->impl_.get(),
            .admission = candidate->impl_.get(),
        });
    }
    return PressurePlanningSession(std::make_unique<detail::PressurePlanningSessionImpl>(
        *impl_, physical_candidates, candidate_ids, private_owners, private_owner_ids,
        shared_owners, shared_owner_ids));
}

runtime::PrefillWork
Program::shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                           const PreparedPrompt& prompt,
                                           std::span<const std::uint32_t> frontiers) {
    if (impl_ == nullptr) { throw std::logic_error("Program is empty"); }
    return impl_->shared_capture_split_prefill_work(candidate, PreparedPromptAccess::view(prompt),
                                                    frontiers);
}

// 启动：**唯一**开始产生物理副作用的地方。两件事在这里同时发生——revision 闸门（封印之后物理状态若
// 变过，这份方案就作废）与所有权移交（prompt 被 take 进事务）。闸门不过就 Aborted，一个字节的物理
// 副作用都没有发生。
runtime::ContextTransactionReserveStatus
Program::start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                                    runtime::CancellationFlagView cancellation) {
    if (plan.revision_.value == 0 || plan.revision_ != impl_->resource_revision()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    return impl_->reserve_materialization(
        std::move(plan.admission_), PreparedPromptAccess::take(std::move(prompt)), cancellation);
}

// 持续回填证明：同样先验 revision（证明必须对"我当时看的那份物理状态"成立），再让实现层回答
// "放这个借用者进来，会不会吃掉被阻塞队首翻身所需的最大额度"。给不出证明就 nullopt——回填要举证，
// 不靠推测。返回的证明会带上 revision，Scheduler 那边据此判断证明是否还有效。
std::optional<PersistentBackfillProof>
Program::prove_persistent_backfill(const RequestBasePlan& blocked_head,
                                   const ResourcePlan& candidate,
                                   std::span<const SequenceHandle> persistent_borrowers) const {
    if (candidate.revision_.value == 0 || candidate.revision_ != impl_->resource_revision() ||
        !impl_->persistent_backfill_safe(blocked_head, candidate.admission_,
                                         persistent_borrowers)) {
        return std::nullopt;
    }
    return PersistentBackfillProof(candidate.revision_);
}

ContextTransactionProgress
Program::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    return impl_->progress_context_transaction(cancellation);
}

void Program::finalize_context_transaction() noexcept { impl_->finalize_context_transaction(); }

bool Program::has_context_transaction() const noexcept { return impl_->has_context_transaction(); }

// ---- 执行：两条产出待处理事务的路径 + 一条不产出的路径 ----
//
// 采样只发生在 advance_prefill 与 decode 里，两者都以 PendingBatch 收尾（等 Runtime 采纳）；
// append_forced_tokens 把已知 token 喂进去，不采样、不推进采样器状态，因此没有待处理事务。
// 这一整组的 noexcept 分布也是刻意的：**收尾动作**（abort_pending / finish / abort / release_* /
// fail_all_cleanup）一律 noexcept——走到那一步时已经没有"失败"这个选项，也不能让异常逃出去。
PrefillProgress Program::advance_prefill(SequenceHandle sequence,
                                         runtime::ExecutionTiming* failed_timing) {
    return impl_->advance_prefill(sequence, failed_timing);
}

CaptureAssessment
Program::inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                         const SharedPrefixHandle* replacement,
                         std::optional<runtime::CheckpointRef> private_replacement,
                         bool permit_shared_publication) const {
    return impl_->inspect_capture(offer, exact_shared, replacement, private_replacement,
                                  permit_shared_publication);
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program::checkpoint_recovery_work(const ContinuationHandle& owner,
                                  runtime::CheckpointRef checkpoint) const {
    return impl_->checkpoint_recovery_work(owner, checkpoint);
}

// 捕获的压力会话与普通会话的构造差别只有两点：候选由捕获评估现场造出（make_capture_physical_candidate），
// 并且这个候选**随返回的会话对象一起被拥有**——它没有独立的生命周期，因此永远无法脱离会话变成一条可准入
// 的请求。物理候选的绑定槽用的是 capture 而不是 admission，这一点也由类型层面固定。
CapturePressurePlanningSession Program::begin_capture_pressure_planning(
    const CaptureAssessment& assessment, std::span<const ContinuationHandle* const> private_owners,
    std::span<const runtime::PlanningOwnerId> private_owner_ids,
    std::span<const SharedPrefixHandle* const> shared_owners,
    std::span<const runtime::PlanningOwnerId> shared_owner_ids) {
    CapturePressureCandidate candidate(impl_->make_capture_physical_candidate(assessment));
    using SessionImpl = detail::PressurePlanningSessionImpl;
    const std::array physical_candidates{SessionImpl::PhysicalCandidateBinding{
        .state   = candidate.impl_.get(),
        .capture = candidate.impl_.get(),
    }};
    const std::array candidate_ids{CapturePressurePlanningSession::candidate_id()};
    PressurePlanningSession session(std::make_unique<detail::PressurePlanningSessionImpl>(
        *impl_, physical_candidates, candidate_ids, private_owners, private_owner_ids,
        shared_owners, shared_owner_ids));
    return CapturePressurePlanningSession(std::move(candidate), std::move(session));
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
Program::checkpoint_recovery_work(const SharedPrefixHandle& owner,
                                  runtime::CheckpointRef checkpoint) const {
    return impl_->checkpoint_recovery_work(owner, checkpoint);
}

bool Program::shared_capture_matches(const CaptureOffer& offer,
                                     const SharedPrefixHandle& shared) const {
    return impl_->shared_capture_matches(offer, shared);
}

void Program::skip_capture(CaptureOffer&& offer) { impl_->skip_capture(std::move(offer)); }

runtime::ContextTransactionReserveStatus
Program::reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                                const SharedPrefixHandle* replacement,
                                std::optional<runtime::CheckpointRef> private_replacement,
                                bool permit_shared_publication,
                                runtime::CancellationFlagView cancellation) {
    return impl_->reserve_active_capture(std::move(offer), exact_shared, replacement,
                                         private_replacement, permit_shared_publication,
                                         cancellation);
}

// 带压力的启动捕获：与 start_resource_transaction 同一道 revision 闸门——压力方案是在某份物理状态上
// 算出来的，状态变了就必须重来，因此这里同样先比对再动手。
runtime::ContextTransactionReserveStatus Program::reserve_active_capture_with_pressure(
    CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
    const SharedPrefixHandle* replacement,
    std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
    CapturePressurePlan&& pressure, runtime::CancellationFlagView cancellation) {
    if (pressure.revision_.value == 0 || pressure.revision_ != impl_->resource_revision()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    return impl_->reserve_active_capture_with_pressure(
        std::move(offer), exact_shared, replacement, private_replacement, permit_shared_publication,
        std::move(pressure.pressure_), cancellation);
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences,
                             std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* failed_timing) {
    return impl_->decode(sequences, budgets, failed_timing);
}

runtime::ExecutionTiming
Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                              std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                              std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                              runtime::ExecutionTiming* failed_timing) {
    return impl_->append_forced_tokens(sequences, row_major_tokens, row_stride,
                                       prefix_execution_splits, failed_timing);
}

CommitResult Program::commit(PendingBatch&& pending,
                             std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation observation,
                             runtime::ExecutionTiming* failed_timing) {
    return impl_->commit(std::move(pending), decisions, observation, failed_timing);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept {
    return impl_->abort_pending(std::move(pending));
}

FinishResult Program::finish(SequenceHandle sequence) noexcept { return impl_->finish(sequence); }

AbortResult Program::abort(SequenceHandle sequence) noexcept { return impl_->abort(sequence); }

ReleaseResult Program::release_continuation(ContinuationHandle&& continuation) noexcept {
    return impl_->release_continuation(std::move(continuation));
}

ReleaseResult Program::release_shared_prefix(SharedPrefixHandle&& shared) noexcept {
    return impl_->release_shared_prefix(std::move(shared));
}

// 故障收尾：不产生任何结果，只把物理事实清干净。它之后 Program 不再假设任何遗留句柄有效——
// 与一个个 release 的差别在于"放弃"和"归还"的语义不同。
void Program::fail_all_cleanup() noexcept { impl_->fail_all_cleanup(); }

// ---- 只读诊断：不改变物理状态 ----
// isolated_request_feasible 是 ResourceManager 判断"永久不可行"的依据：不借助任何复用都放不下，
// 那再怎么等也不会变可行。physical_usage 自带 revision，因此"这组数字属于哪个物理状态"是可判定的。
bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    return impl_->isolated_request_feasible(base);
}

runtime::ProgramResourceRevision Program::resource_revision() const noexcept {
    return impl_->resource_revision();
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->physical_usage(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory_summary(); }

void Program::reset_memory_peaks() noexcept { impl_->reset_memory_peaks(); }

// ---- 两个工厂 ----
//
// 顺序是先规划、后建 Program：make_sequence_planner 只产出规划器（此时还没有任何物理资源被铺开），
// create_program 才真正按定稿方案建立物理现实。

SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                      DeviceContext& device, const EngineOptions& options) {
    return SequencePlanner(detail::make_sequence_planner_impl(parameters, device, options));
}

// 建 Program。两道校验都在防同一类错误——把不属于这里的方案拿进来：
//   1. 方案必须存在（已经 finalize 过）；
//   2. 方案必须出自同一个模型实例（用 parameters 的地址做身份比对，因为这件事无法在类型上表达）。
// 之后 plan.impl_.reset() 是刻意的：方案已被 Program 吸收，旧变量立即失效，杜绝"同一份方案建两个 Program"。
std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                        SequencePlan&& plan, DeviceContext& device,
                                        const StartupObserver& startup_observer) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    if (plan.impl_->parameters != &parameters) {
        throw std::invalid_argument("sequence plan belongs to another model instance");
    }
    auto impl =
        std::make_unique<detail::ProgramImpl>(parameters, *plan.impl_, device, startup_observer);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

} // namespace ninfer::models::qwen3_5

// ============================================================================
// 小值类型的壳：AdmissionCandidate / CapturePressureCandidate / CaptureAssessment
// ============================================================================
//
// 它们与 program.h 里的声明分开写在这里，原因很实在：PIMPL 的移动构造与析构要求 impl 类型是完整的，
// 而 impl 只有在本 TU 才完整。所以这一段的职责就是把"不完整"这件事挡住。
//
// 两个候选是同构的 move-only 壳（与 SequencePlan 一样的写法）：外部只能移动、查询，不能改造内部。
// 查询接口在"空壳"上返回静态空值而不是抛异常——查询侧给中性答案，动作侧才用异常表达违约。
// CaptureAssessment 则相反，它的构造**必然**分配那份不透明 payload：评估只要存在，就一定带着它的
// 物理依据，不存在"有结论没载荷"的半成品。

namespace ninfer::models::qwen3_5 {

CaptureAssessment::CaptureAssessment()
    : implementation(std::make_shared<detail::CaptureAssessmentImpl>()) {}

AdmissionCandidate::AdmissionCandidate(
    std::unique_ptr<detail::AdmissionCandidateImpl> impl) noexcept
    : impl_(std::move(impl)) {}

AdmissionCandidate::AdmissionCandidate(AdmissionCandidate&&) noexcept = default;

AdmissionCandidate& AdmissionCandidate::operator=(AdmissionCandidate&&) noexcept = default;

AdmissionCandidate::~AdmissionCandidate() = default;

CapturePressureCandidate::CapturePressureCandidate(
    std::unique_ptr<detail::CapturePressureCandidateImpl> impl) noexcept
    : impl_(std::move(impl)) {}

CapturePressureCandidate::CapturePressureCandidate(CapturePressureCandidate&&) noexcept = default;

CapturePressureCandidate&
CapturePressureCandidate::operator=(CapturePressureCandidate&&) noexcept = default;

CapturePressureCandidate::~CapturePressureCandidate() = default;

const runtime::RequestPlanSummary& AdmissionCandidate::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const runtime::IdentityMaterializationAssessment&
AdmissionCandidate::identity_assessment() const noexcept {
    static const runtime::IdentityMaterializationAssessment empty;
    return impl_ != nullptr ? impl_->identity_assessment : empty;
}

} // namespace ninfer::models::qwen3_5
