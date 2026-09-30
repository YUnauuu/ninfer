#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

// ============================================================================
// models/qwen3_5/program/program.h —— 这个模型的物理权威的对外契约面
// ============================================================================
//
// Program 是 Qwen3.5 的**物理权威**：State/KV 的真实 object、引用与 replica，Device/Host 的池子与
// allocator，continuation 的完整性与前缀的 exact identity，全部由它拥有并解释。它上面是模型无关的
// Runtime，两者之间只通过本文件说话：
//
//     EngineCore（请求生命周期）──▶ ResourceManager（逻辑缓存策略）──▶ **Program（物理）**
//                                                                     │
//                                                               Ops / storage / graphs
//
// 所以这不是普通的内部实现头，而是分界线：Runtime 能看见什么、能拿什么、必须原样交回什么，全在这里
// 定义。整份契约围绕三条方向：
//
//   1. **能力代替查询**。Runtime 不读物理账本、不数 page、不算 refcount，它只持有 Program 铸出的
//      句柄与封印的计划。句柄是 owner + 槽位 + generation 的不可伪造组合：Runtime 只能传递、归还、
//      消费，不能拆解也不能自造；槽位复用造成的新旧混淆由 generation 拦下。
//   2. **封印代替 delta**。Program 针对**完整终态** seal 出不可变、绑定 resource_revision 的计划；
//      Runtime 保留它并在 Start 时交回。计划没有"再改一处"的接口，Runtime 也看不到 allocator 数量、
//      引用与阶段峰值——它只能整份接受或拒绝。
//   3. **终态代替回放**。每个事务结束（commit / abort）报的是**绝对终态**而不是一串 delta，上层采用
//      它即可，不需要、也不允许重放中间 receipt。这也是 adoption 必须预分配且不能失败的原因。
//
// 配套的三条分工（权威文档是 docs/maintainer/resource-scheduling-and-context-cache.md）：
//   * Scheduler 先选请求，资源层不得用 cache value 改变请求顺序；
//   * ResourceManager 是逻辑策略的唯一 authority，Program 是物理事实的唯一 authority；
//   * 同一时刻至多一个 global resource transition，也至多一个未采用的 pending 执行事务。
//
// 本文件只声明这一具体模型的执行与资源契约（数据由模型实例提供）；**模型目标的挑选不在这层**，它在
// Engine 启动时的闭集注册表里发生一次。

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::models::qwen3_5 {

namespace execution {
class Parameters;
}

// 不透明的物理实现（pimpl）：本头文件只声明"有这么个东西"，它的字段与含义留在 Program 的实现里。
// 对外可见的评估结论在 CaptureAssessment 上，物理细节不外泄。
namespace detail {
struct CaptureAssessmentImpl;

} // namespace detail

// 从 Program 真实存储上采样的**只读诊断**，不是记账输入：拿它做可行性判断是错的（那要用 assessment
// 与 revision）。它自带 resource_revision，所以一组数字对应的是哪一个物理状态是可判定的。
struct PhysicalUsageSnapshot {
    runtime::ProgramResourceRevision resource_revision;
    std::uint32_t device_state_slots      = 0;
    std::uint32_t host_state_slots        = 0;
    std::uint32_t device_main_kv_pages    = 0;
    std::uint32_t device_backend_kv_pages = 0;
    std::size_t host_kv_bytes             = 0;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalUsageSnapshot&,
                                                   const PhysicalUsageSnapshot&) noexcept = default;
};

// 一段文本执行属于哪种**形态**：Pre–fill（一个提示前沿连续推进）还是 Verify（批量的一轮，含投机
// 草案的验证）。它用来选执行图与 workspace 路径，描述的是"这一段算什么样"，不是请求的状态机。
enum class TextPhase {
    Prefill,
    Verify,
};

// 启动期为 CUDA Graph 预备的一个**形状桶**：行数区间 [min, max] 加拓扑类别。它回答的是"要提前把哪些
// 形状的图建好"，不是运行期的选择结果。
struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};

// Program 铸出的**候选筛选键**：只用来把 catalog 检查范围缩小，不证明命中。真正选中一个 checkpoint
// 之前，Program 还要重新验证 exact token、位置、媒体与运行期模式——也就是说落在同一个 key 上只是
// "值得看一眼"。（这条与文档里的"排序提示不是证明"是同一条纪律。）
struct PrefixShortlistKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier     = 0;
    std::uint32_t identity_tag = 0;

    [[nodiscard]] friend constexpr bool operator==(PrefixShortlistKey,
                                                   PrefixShortlistKey) noexcept = default;
};

struct TargetKVRequirement {
    std::uint32_t main_frontier    = 0;
    std::uint32_t backend_frontier = 0;
    std::uint32_t main_pages       = 0;
    std::uint32_t backend_pages    = 0;

    [[nodiscard]] friend constexpr bool operator==(TargetKVRequirement,
                                                   TargetKVRequirement) noexcept = default;
};

// ---- 逻辑摘要：ResourceManager 看得见的那一半 ----
//
// 下面这组结构描述的是**逻辑**事实（哪个 checkpoint、覆盖到哪个 frontier、被谁租用、能省下多少重算），
// 刻意不含任何 allocator 数量、引用计数或 placement 细节。ResourceManager 用它们做 catalog、session
// 绑定与 portfolio 价值比较；物理真相始终只能问 Program。

// 一个 checkpoint 的对外摘要。scope 决定它在哪本账上（private continuation 还是 shared prefix），
// residency 说明它当前的 replica 落在哪一侧，required_kv 与 rebuild_work 则是"从它继续跑要补多少"。
struct CheckpointSummary {
    runtime::CheckpointRef ref;
    runtime::CheckpointScope scope = runtime::CheckpointScope::Private;
    PrefixShortlistKey shortlist_key;
    runtime::ReplicaResidency state_residency = runtime::ReplicaResidency::DeviceOnly;
    TargetKVRequirement required_kv;
    runtime::PrefillWork rebuild_work;

    [[nodiscard]] friend bool operator==(const CheckpointSummary&,
                                         const CheckpointSummary&) noexcept = default;
};

// 一条 private continuation 的全貌：endpoint 是它的正常终止点，rewrite 是可用于重写/分支的基点，
// long_anchors 是这条历史里额外保留的锚点（上限由配置定）。active_references 是**逻辑**租用者数量，
// 由 ResourceManager 的 owner edge 派生，不是引用计数镜像。
struct ContinuationSummary {
    std::optional<CheckpointSummary> endpoint;
    std::optional<CheckpointSummary> rewrite;
    std::vector<CheckpointSummary> long_anchors;
    std::uint32_t active_references = 0;

    [[nodiscard]] friend bool operator==(const ContinuationSummary&,
                                         const ContinuationSummary&) noexcept = default;
};

// 一个 shared prefix 的对外摘要。它一旦发布就**不可变**，因此没有"重写点"这类概念，只有一个
// checkpoint 与租用者数量。
struct SharedPrefixSummary {
    CheckpointSummary checkpoint;
    std::uint32_t active_references = 0;

    [[nodiscard]] friend bool operator==(const SharedPrefixSummary&,
                                         const SharedPrefixSummary&) noexcept = default;
};

// impl 前置声明：本文件里的所有实体都是"薄壳 + 私有实现"的 PIMPL 形态，具体状态不在这里暴露。
// 于是这个头可以被 Runtime 广泛包含，而模型侧的规划状态仍然只有 .cpp 知道。
namespace detail {

struct SequencePlanImpl;

struct SequencePlannerImpl;

struct AdmissionCandidateImpl;

struct CapturePressureCandidateImpl;

struct RequestBasePlanImpl;

struct PressurePlanningSessionImpl;

class ProgramImpl;

struct RuntimeContractAccess;
} // namespace detail

class SequencePlanner;

class Program;

class PressurePlanningSession;

class CapturePressurePlanningSession;

class CapturePressurePlan;

class CapturePressureCandidate;

// 这一组是 Qwen 这个具体模型的执行与资源契约；数据由模型实例提供。模型目标的挑选不在这层——它在
// Engine 启动时的闭集注册表里发生一次。

// 启动期定下的**序列容量计划**：这台机器上这套权重能同时容纳多少请求、KV 到多少 token、Device 侧
// 要预留多少字节、workspace 有多大。它在整个进程内不变，是"物理容量"的一部分（另一部分是各轴的具体
// 占用，那由 Program 的存储解释）。
class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;

public:
    // 家族私有的构造/存储接口：只有 make_sequence_planner 与 Program 实现用到。外部只见成品别名，
    // 见不到规划过程。
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl> impl_;

    friend class SequencePlanner;

    friend class detail::ProgramImpl;
};

// 序列容量的**规划器**：由 make_sequence_planner 起步，读到容量曲线后 finalize 出一个 SequencePlan。
// 它只在启动期活着（finalize 是 && 限定的，也就是"用完即弃"）。
class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl> impl_;

    friend SequencePlanner make_sequence_planner(const execution::Parameters&, DeviceContext&,
                                                 const EngineOptions&);
};

// 一条请求的**基础计划**：只由提示本身决定（与当时有哪些邻居无关），所以算一次就能一直被复用——
// EngineCore 正是按"惰性建立、之后随请求走到终态"来用它。它给出调度需要的服务份额估算、上下文缓存
// 的候选机会，以及按 frontier 查询筛选键/重算工作量的入口。
class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const PreparedContextCache& context_cache() const noexcept;
    [[nodiscard]] std::optional<PrefixShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept;
    [[nodiscard]] std::optional<runtime::PrefillWork>
    shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl> impl_;
};

// 一个**已验证的候选来源**：Program 对某个 root / private checkpoint / shared prefix 做完 exact 校验后
// 给出的结果。它固定了 source、destination、prompt 与工作量，但**不**预先决定 Move 还是 Fork、谁当
// victim、最终落到哪一侧——那些要等完整 target 的评估。Runtime 只能看 summary 与身份评估，看不到
// 目标规划状态。
class AdmissionCandidate {
public:
    AdmissionCandidate(AdmissionCandidate&&) noexcept;
    AdmissionCandidate& operator=(AdmissionCandidate&&) noexcept;
    ~AdmissionCandidate();

    AdmissionCandidate(const AdmissionCandidate&)            = delete;
    AdmissionCandidate& operator=(const AdmissionCandidate&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const runtime::IdentityMaterializationAssessment&
    identity_assessment() const noexcept;

public:
    // 家族私有的构造/存储接口：只有 Program 与压力会话能造。Engine 侧只能读 summary()，不能读目标
    // 规划状态——这正是"封印"的边界落在哪里的一个例子。
    explicit AdmissionCandidate(std::unique_ptr<detail::AdmissionCandidateImpl> impl) noexcept;
    std::unique_ptr<detail::AdmissionCandidateImpl> impl_;

    friend class Program;
    friend class PressurePlanningSession;
    friend struct detail::PressurePlanningSessionImpl;
};

// capture 侧的压力候选持有者（家族私有）。它与上面的请求准入候选刻意长得不一样：**没有**请求摘要、
// 也没有准入接口，因此不可能被当成一条请求去检查、封印或执行——它只描述"这次 capture 的物理后态"。

class CapturePressureCandidate {
public:
    CapturePressureCandidate(CapturePressureCandidate&&) noexcept;
    CapturePressureCandidate& operator=(CapturePressureCandidate&&) noexcept;
    ~CapturePressureCandidate();

    CapturePressureCandidate(const CapturePressureCandidate&)            = delete;
    CapturePressureCandidate& operator=(const CapturePressureCandidate&) = delete;

public:
    explicit CapturePressureCandidate(
        std::unique_ptr<detail::CapturePressureCandidateImpl> impl) noexcept;
    std::unique_ptr<detail::CapturePressureCandidateImpl> impl_;

    friend class Program;
    friend class PressurePlanningSession;
    friend class CapturePressurePlan;
    friend struct detail::PressurePlanningSessionImpl;
};

// 一次 active capture 的、只含压力降级的封印后态。payload 归 Program 私有，无法经请求准入接口检查或
// 执行；Runtime 能看到的只有它的 resource_revision，用来在使用时确认世界没变。

class CapturePressurePlan {
public:
    CapturePressurePlan(CapturePressurePlan&&) noexcept            = default;
    CapturePressurePlan& operator=(CapturePressurePlan&&) noexcept = default;
    ~CapturePressurePlan()                                         = default;

    CapturePressurePlan(const CapturePressurePlan&)            = delete;
    CapturePressurePlan& operator=(const CapturePressurePlan&) = delete;

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return revision_;
    }

private:
    CapturePressurePlan(CapturePressureCandidate&& pressure,
                        runtime::ProgramResourceRevision revision) noexcept
        : pressure_(std::move(pressure)), revision_(revision) {}

    CapturePressureCandidate pressure_;
    runtime::ProgramResourceRevision revision_;

    friend class Program;
    friend class PressurePlanningSession;
};

// 一份封印的物理决策，这也是 Runtime 与 Program 之间最主要的交接物。ResourceManager 可以留着它、读它
// 的请求级 summary，但看不到 allocator 数量、引用、预留与阶段 delta。使用时（start）Program 会先校验
// 它绑定的 resource_revision：**对不上就在产生任何物理副作用之前被拒**。seal 之后这份计划没有修改接口，
// 因此"被评估的物理目标"与"实际执行的计划"不可能偏离。

class ResourcePlan {
public:
    ResourcePlan(ResourcePlan&&) noexcept            = default;
    ResourcePlan& operator=(ResourcePlan&&) noexcept = default;
    ~ResourcePlan()                                  = default;

    ResourcePlan(const ResourcePlan&)            = delete;
    ResourcePlan& operator=(const ResourcePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept {
        return admission_.summary();
    }

    [[nodiscard]] bool needs_transfer() const noexcept { return needs_transfer_; }

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return revision_;
    }

private:
    ResourcePlan(AdmissionCandidate&& admission, runtime::ProgramResourceRevision revision,
                 bool needs_transfer) noexcept
        : admission_(std::move(admission)), revision_(revision), needs_transfer_(needs_transfer) {}

    AdmissionCandidate admission_;
    runtime::ProgramResourceRevision revision_;
    bool needs_transfer_ = false;

    friend class Program;
    friend class PressurePlanningSession;
};

// Program 铸出的物理证明：**放这个回填者进来，不会吃掉被阻塞队首的那份最大物理额度**。它证明的是
// 一个假设状态（当前稳定状态 − donor 释放后的预留 + 借用者完整预留 + 队首的 root 物化）可行，与
// 借用者/donor 的预计完成时间无关。Scheduler 只能把这个不透明证明与逻辑身份、revision 绑在一起，
// 无法窥视或复算其中的资源算术——这正是"回填不是插队"的凭据。

class PersistentBackfillProof {
public:
    PersistentBackfillProof(PersistentBackfillProof&&) noexcept            = default;
    PersistentBackfillProof& operator=(PersistentBackfillProof&&) noexcept = default;

    PersistentBackfillProof(const PersistentBackfillProof&)            = delete;
    PersistentBackfillProof& operator=(const PersistentBackfillProof&) = delete;

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return revision_;
    }

private:
    explicit PersistentBackfillProof(runtime::ProgramResourceRevision revision) noexcept
        : revision_(revision) {}

    runtime::ProgramResourceRevision revision_;

    friend class Program;
};

// ---- 能力句柄 ----
//
// 这一组是 Runtime 能持有的全部"物理凭证"。三条共同约定：
//   * **不可伪造**：只有 Program（经 detail::RuntimeContractAccess 这个唯一的口子）能造出有效值，
//     公开接口里没有任何构造器，内部字段也全部私有。
//   * **会过期**：句柄记着 owner、槽位与 generation/epoch，槽位被复用后旧句柄必然失配。所以它回答的
//     是"这个具体实例还活着吗"，而不是"这个编号现在指向谁"。
//   * **一次性**：代表独占持有的（continuation / shared prefix / capture offer / pending batch）
//     一律 move-only 并有一个显式的 consume 动作；代表**身份**的（SequenceHandle、
//     PressureTargetHandle）可以拷贝——身份本来就会被反复使用。
//
// 它们都不携带任何物理数字，因此可以在模型无关的接口里自由传递。

// 一个 active 序列的身份（owner + lane + epoch）。它同时是"这条 lane 上是谁"的凭据，也是
// prefill / decode / commit 各接口的行身份，所以可拷贝。把它交给 finish/abort 就是把这条序列的
// active 生命周期交出去；此后 lane 会被回收，残留的旧句柄由 epoch 拦下。
class SequenceHandle {
public:
    SequenceHandle() noexcept                                 = default;
    SequenceHandle(const SequenceHandle&) noexcept            = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept = default;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    friend struct detail::RuntimeContractAccess;
};

// 一条 private continuation 的**逻辑租约**（owner + catalog 槽位 + generation）。持有它代表这条
// continuation 还没被释放，release_continuation 会消费掉它。注意它是 lease 而不是引用计数镜像：
// 别的 reader 改变同一 owner 的 Device/Host residency 会让 generation 前进，但不会让已有的租约失效。
class ContinuationHandle {
public:
    ContinuationHandle() noexcept = default;
    ~ContinuationHandle()         = default;

    ContinuationHandle(ContinuationHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    ContinuationHandle& operator=(ContinuationHandle&&)      = delete;
    ContinuationHandle(const ContinuationHandle&)            = delete;
    ContinuationHandle& operator=(const ContinuationHandle&) = delete;

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;

    friend struct detail::RuntimeContractAccess;
};

// 一个已发布的 shared prefix 的租约。它比 private 那条更简单：shared checkpoint 一经发布就不可变，
// 所以这个句柄不表达"可变所有者"，只表达"这份 public 缓存还在被谁需要"。
class SharedPrefixHandle {
public:
    SharedPrefixHandle() noexcept = default;
    ~SharedPrefixHandle()         = default;

    SharedPrefixHandle(SharedPrefixHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    SharedPrefixHandle& operator=(SharedPrefixHandle&&)      = delete;
    SharedPrefixHandle(const SharedPrefixHandle&)            = delete;
    SharedPrefixHandle& operator=(const SharedPrefixHandle&) = delete;

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;

    friend struct detail::RuntimeContractAccess;
};

// 压力搜索里的一个 target 身份（session + generation + 下标）。可拷贝、可比较（Runtime 用它做访问
// 去重与队列），但内容只有它所属的那个 session 认识。
class PressureTargetHandle {
public:
    PressureTargetHandle() noexcept = default;

    [[nodiscard]] friend constexpr bool operator==(PressureTargetHandle,
                                                   PressureTargetHandle) noexcept = default;

private:
    const void* session_      = nullptr;
    std::uint32_t generation_ = 0;
    std::uint32_t index_      = 0;

    friend struct detail::PressurePlanningSessionImpl;

    friend class PressurePlanningSession;
};

// "分片构造"的游标：Runtime 用它逐个取出 Program 给出的构造选项、排序、挑一个，再问结果 target。
// 它是 RAII 的——析构即把 session 里那一格还回去，所以中途放弃搜索不需要额外的清理调用。
class PressureConstructionCursor {
public:
    PressureConstructionCursor(PressureConstructionCursor&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)), slot_(other.slot_),
          generation_(other.generation_), release_(other.release_) {}

    PressureConstructionCursor& operator=(PressureConstructionCursor&&)      = delete;
    PressureConstructionCursor(const PressureConstructionCursor&)            = delete;
    PressureConstructionCursor& operator=(const PressureConstructionCursor&) = delete;

    ~PressureConstructionCursor() {
        if (session_) { release_(session_, slot_, generation_); }
    }

private:
    PressureConstructionCursor(const void* session, std::uint32_t slot, std::uint32_t generation,
                               void (*release)(const void*, std::uint32_t, std::uint32_t) noexcept)
        : session_(session), slot_(slot), generation_(generation), release_(release) {}

    const void* session_;
    std::uint32_t slot_;
    std::uint32_t generation_;
    void (*release_)(const void*, std::uint32_t, std::uint32_t) noexcept;

    friend struct detail::PressurePlanningSessionImpl;
};

// 一个 target 的**精确评估结果**，它有两个身份：既是"占着 session 一格"的借用（析构归还），又是封印
// 所需能力的唯一携带者（executable / capture_executable）。方向：只有 assess 过的 target 才能 seal，
// 而 seal 会把它消费掉——同一份评估不可能既封印成物化、又封印成 capture。
class AssessedPressureTarget {
public:
    AssessedPressureTarget(AssessedPressureTarget&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)),
          session_generation_(std::exchange(other.session_generation_, 0)),
          target_index_(other.target_index_), assessment_(other.assessment_),
          assessment_slot_(std::exchange(other.assessment_slot_, 0)),
          assessment_slot_generation_(std::exchange(other.assessment_slot_generation_, 0)),
          release_slot_(std::exchange(other.release_slot_, nullptr)),
          executable_(std::move(other.executable_)),
          capture_executable_(std::move(other.capture_executable_)) {}

    ~AssessedPressureTarget() { reset(); }

    AssessedPressureTarget& operator=(AssessedPressureTarget&& other) noexcept {
        if (this == &other) { return *this; }
        reset();
        session_                    = std::exchange(other.session_, nullptr);
        session_generation_         = std::exchange(other.session_generation_, 0);
        target_index_               = other.target_index_;
        assessment_                 = other.assessment_;
        assessment_slot_            = std::exchange(other.assessment_slot_, 0);
        assessment_slot_generation_ = std::exchange(other.assessment_slot_generation_, 0);
        release_slot_               = std::exchange(other.release_slot_, nullptr);
        executable_                 = std::move(other.executable_);
        capture_executable_         = std::move(other.capture_executable_);
        return *this;
    }

    AssessedPressureTarget(const AssessedPressureTarget&)            = delete;
    AssessedPressureTarget& operator=(const AssessedPressureTarget&) = delete;

    [[nodiscard]] const runtime::PressureTargetAssessment& assessment() const noexcept {
        return assessment_;
    }

private:
    AssessedPressureTarget(const void* session, std::uint32_t session_generation,
                           std::uint32_t target_index, runtime::PressureTargetAssessment assessment,
                           std::uint32_t assessment_slot, std::uint32_t assessment_slot_generation,
                           void (*release_slot)(const void*, std::uint32_t, std::uint32_t) noexcept,
                           std::optional<AdmissionCandidate>&& executable,
                           std::optional<CapturePressureCandidate>&& capture_executable) noexcept
        : session_(session), session_generation_(session_generation), target_index_(target_index),
          assessment_(assessment), assessment_slot_(assessment_slot),
          assessment_slot_generation_(assessment_slot_generation), release_slot_(release_slot),
          executable_(std::move(executable)), capture_executable_(std::move(capture_executable)) {}

    void reset() noexcept {
        if (session_ != nullptr && release_slot_ != nullptr) {
            release_slot_(session_, assessment_slot_, assessment_slot_generation_);
        }
        session_                    = nullptr;
        session_generation_         = 0;
        assessment_slot_            = 0;
        assessment_slot_generation_ = 0;
        release_slot_               = nullptr;
    }

    const void* session_              = nullptr;
    std::uint32_t session_generation_ = 0;
    std::uint32_t target_index_       = 0;
    runtime::PressureTargetAssessment assessment_;
    std::uint32_t assessment_slot_                                            = 0;
    std::uint32_t assessment_slot_generation_                                 = 0;
    void (*release_slot_)(const void*, std::uint32_t, std::uint32_t) noexcept = nullptr;
    std::optional<AdmissionCandidate> executable_;
    std::optional<CapturePressureCandidate> capture_executable_;

    friend class PressurePlanningSession;
    friend struct detail::PressurePlanningSessionImpl;
};

// 已经备好、但还没生效的一次 expansion。两段式的意义就在这里：prepare →（commit | discard），
// 而 discard 保证**批次停止或丢弃绝不产生真实资源 mutation**。
class PreparedPressureExpansion {
public:
    PreparedPressureExpansion(PreparedPressureExpansion&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)),
          session_generation_(std::exchange(other.session_generation_, 0)),
          scratch_generation_(std::exchange(other.scratch_generation_, 0)),
          parent_index_(other.parent_index_), new_canonical_count_(other.new_canonical_count_) {}

    PreparedPressureExpansion& operator=(PreparedPressureExpansion&&)      = delete;
    PreparedPressureExpansion(const PreparedPressureExpansion&)            = delete;
    PreparedPressureExpansion& operator=(const PreparedPressureExpansion&) = delete;

    [[nodiscard]] std::uint32_t new_canonical_count() const noexcept {
        return new_canonical_count_;
    }

private:
    PreparedPressureExpansion(const void* session, std::uint32_t session_generation,
                              std::uint32_t scratch_generation, std::uint32_t parent_index,
                              std::uint32_t new_canonical_count) noexcept
        : session_(session), session_generation_(session_generation),
          scratch_generation_(scratch_generation), parent_index_(parent_index),
          new_canonical_count_(new_canonical_count) {}

    const void* session_               = nullptr;
    std::uint32_t session_generation_  = 0;
    std::uint32_t scratch_generation_  = 0;
    std::uint32_t parent_index_        = 0;
    std::uint32_t new_canonical_count_ = 0;

    friend class PressurePlanningSession;
    friend struct detail::PressurePlanningSessionImpl;
};

// 一次 expansion 的结果视图。children 借自 session 的 scratch 代（下一次 session 操作即失效，调用方
// 必须当场折叠成自有值）；complete = false 表示这一层还没放完（受 session 容量限制），不是失败。
struct PressureExpansionView {
    std::span<const PressureTargetHandle> children;
    std::uint32_t new_canonical_count = 0;
    bool complete                     = true;
};

// ---- 压力规划会话：Program 持有物理域，Runtime 负责搜索 ----
//
// 分工是刻意的：Program 知道"哪些 target 在物理上真的存在、真的可行"，Runtime 知道"当前该优先争取
// 什么、什么时候收手"。于是把搜索本身留在 Runtime：会话只提供 target 的枚举、构造、扩展、评估与封印，
// 一次都不替调用方做选择。
//
// 两条纪律贯穿全部接口：
//   * **排序提示不是证明**：guidance 只是排序用的提示，它把候选排前面不代表可行；只有 assess 给出的
//     精确评估才让 target 有资格被采用或封印。Runtime 若拿 guidance 当可行性依据，就会在封印时失败。
//   * **能力即许可**：能封印的只有 AssessedPressureTarget；seal 把它消费掉，因此同一份评估不可能被
//     用两次（封印一次物化、又封印一次 capture 是不可能的）。
// 寿命上：session 本身是 Program 出借的规划工作区，同一时刻每种域至多一个；它析构即归还，所以失败的
// 搜索不需要补偿动作——没有生效过的准备（PreparedPressureExpansion）直接丢弃即可。
class PressurePlanningSession {
public:
    PressurePlanningSession(PressurePlanningSession&&) noexcept;
    PressurePlanningSession& operator=(PressurePlanningSession&&) noexcept;
    ~PressurePlanningSession();

    PressurePlanningSession(const PressurePlanningSession&)            = delete;
    PressurePlanningSession& operator=(const PressurePlanningSession&) = delete;

    [[nodiscard]] PressureTargetHandle
    identity_target(runtime::PlanningCandidateId candidate) const;
    [[nodiscard]] PressureTargetHandle
    root_maximal_target(runtime::PlanningCandidateId root_candidate);
    [[nodiscard]] PressureTargetHandle maximal_target(runtime::PlanningCandidateId candidate);
    [[nodiscard]] PressureConstructionCursor begin_construction(PressureTargetHandle target,
                                                                bool restore = false);
    [[nodiscard]] runtime::PressureConstructionStep
    next_construction_option(PressureConstructionCursor& cursor);
    void choose_construction(PressureConstructionCursor& cursor,
                             runtime::PressureConstructionOptionId option);
    [[nodiscard]] std::optional<PressureTargetHandle>
    construction_target(const PressureConstructionCursor& cursor);
    [[nodiscard]] runtime::PressureTargetGuidance guidance(PressureTargetHandle target);
    [[nodiscard]] AssessedPressureTarget assess(PressureTargetHandle target);
    [[nodiscard]] PreparedPressureExpansion
    prepare_expansion(PressureTargetHandle parent,
                      std::uint32_t maximum_owners = std::numeric_limits<std::uint32_t>::max());
    [[nodiscard]] PressureExpansionView commit_expansion(PreparedPressureExpansion&& prepared);
    void discard_expansion(PreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const AssessedPressureTarget& assessed,
                                      const PreparedPrompt& prompt,
                                      std::span<const std::uint32_t> frontiers) const;
    [[nodiscard]] std::optional<ResourcePlan> seal(AssessedPressureTarget&& assessed,
                                                   const PreparedPrompt& prompt,
                                                   runtime::FinalScheduleIntent intent);
    [[nodiscard]] std::optional<CapturePressurePlan>
    seal_capture(AssessedPressureTarget&& assessed);

private:
    explicit PressurePlanningSession(
        std::unique_ptr<detail::PressurePlanningSessionImpl> impl) noexcept;

    std::unique_ptr<detail::PressurePlanningSessionImpl> impl_;

    friend class Program;
};

// 一次活跃 capture 专用的、带类型的压力域。它没有 identity_target(candidate) 那种"按 id 取 target"的
// 入口，也没有 ResourcePlan 封印——因为这里的候选（CapturePressureCandidate）始终归 Program 所有，
// 既不能被当请求摘要检查，也不能被当准入候选封印或执行。
// 换句话说：capture 想为腾地方而施压，但它自己绝不会变成一条"请求"混进调度里。
class CapturePressurePlanningSession {
public:
    CapturePressurePlanningSession(CapturePressurePlanningSession&&) noexcept;
    CapturePressurePlanningSession& operator=(CapturePressurePlanningSession&&) noexcept;
    ~CapturePressurePlanningSession();

    CapturePressurePlanningSession(const CapturePressurePlanningSession&)            = delete;
    CapturePressurePlanningSession& operator=(const CapturePressurePlanningSession&) = delete;

    [[nodiscard]] PressureTargetHandle identity_target() const;
    [[nodiscard]] runtime::PressureTargetGuidance guidance(PressureTargetHandle target);
    [[nodiscard]] AssessedPressureTarget assess(PressureTargetHandle target);
    [[nodiscard]] PreparedPressureExpansion prepare_expansion(PressureTargetHandle parent);
    [[nodiscard]] PressureExpansionView commit_expansion(PreparedPressureExpansion&& prepared);
    void discard_expansion(PreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] std::optional<CapturePressurePlan> seal(AssessedPressureTarget&& assessed);

    [[nodiscard]] static constexpr runtime::PlanningCandidateId candidate_id() noexcept {
        return runtime::PlanningCandidateId{.value = 0};
    }

private:
    CapturePressurePlanningSession(CapturePressureCandidate&& candidate,
                                   PressurePlanningSession&& session) noexcept
        : candidate_(std::move(candidate)), session_(std::move(session)) {}

    CapturePressureCandidate candidate_;
    PressurePlanningSession session_;

    friend class Program;
};

// Program 铸出的一次 capture 机会。方向：捕获唯一生效的时机是"新生成了 Begin 令牌"那一步，而 Program
// 没法替调用方决定要不要——所以这里只交给调用方一个不透明凭据（检查 / 预留 / 跳过都由 Runtime 决定），
// 最后在 CommitResult.captures 里按行对齐报告结果。epoch_ 是防串号的：跨轮次留下的旧 offer 会失效。
class CaptureOffer {
public:
    CaptureOffer() noexcept = default;
    ~CaptureOffer()         = default;

    CaptureOffer(CaptureOffer&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), lane_(other.lane_), epoch_(other.epoch_),
          id_(std::exchange(other.id_, 0)) {}

    CaptureOffer& operator=(CaptureOffer&&)      = delete;
    CaptureOffer(const CaptureOffer&)            = delete;
    CaptureOffer& operator=(const CaptureOffer&) = delete;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;
    std::uint64_t id_    = 0;

    friend struct detail::RuntimeContractAccess;
};

// 已产生但**尚未提交**的一批生成结果：一次 prefill 或 verify 的采样输出，等着 Runtime 决定提交还是丢弃。
//
// 它是"未采纳的待处理事务"这一全局不变量的载体——Program 同时只允许一个存在（再要一个会直接抛错），
// 因此不存在两个互相打架的待提交批次。
// 陷阱：tokens_ / row_counts_ 是**借**自 Program 轮次缓冲区的 span，不是自有存储；它的寿命只到本次
// 事务被提交或丢弃为止，调用方必须在这之前把数据取走。移动构造会清空源对象，正是为了让"借来的视图"
// 只有一个持有者。
class PendingBatch {
public:
    PendingBatch() noexcept = default;
    ~PendingBatch()         = default;

    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          transaction_(std::exchange(other.transaction_, 0)), rows_(other.rows_),
          row_count_(std::exchange(other.row_count_, 0)), tokens_(other.tokens_),
          row_counts_(other.row_counts_), row_stride_(other.row_stride_), timing_(other.timing_) {
        other.tokens_     = {};
        other.row_counts_ = {};
        other.row_stride_ = 0;
        other.timing_     = {};
    }

    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }

    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_;

    friend struct detail::RuntimeContractAccess;
};

// 一次 prefill 推进的回报。步进式：每次调用只吃掉一段 prompt，processed_prompt_tokens 是**累计**
// 已处理量（不是本次增量），complete 才表示 prompt 走完。
// complete = true 时 pending 才有值：里面装着本次采样出的第一个生成 token，作为未采纳事务交给 Runtime
// 决定提交或丢弃；同一时刻 Program 只允许一个这样的未决批次存在。
// capture 是可选的捕获机会，同样只在本步真的产生了 Begin 令牌时才有值。
struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    std::optional<CaptureOffer> capture;
};

// 捕获的私有状态放在哪：留在设备的 fork 副本上，还是搬回主机快照。这是**物理放置选择**，不是策略
// 偏好——它决定了后续续跑是零拷贝还是需要一次回传。
enum class CaptureStatePlacement : std::uint8_t {
    DeviceFork,
    HostSnapshot,
};

// 对一个 capture 候选的物理评估结论：由 Program 逐项判定，Runtime 只读它来判断值不值得捕获。
// 方向性：physically_feasible 是这里唯一"可行性"的权威表述（不是短名单或排序提示），而
// shortlist_key / shared_evidence 只是给 catalog 检查缩小范围、给共享判定提供线索的辅助信息。
// 真正的物理细节保留在 Program 私有的不透明 payload 里（implementation），外部看到的只是结论面。
struct CaptureAssessment {
    CaptureAssessment();

    // Program 把物理评估结果留在这个不透明的私有 payload 中。
    std::shared_ptr<detail::CaptureAssessmentImpl> implementation;
    PrefixShortlistKey shortlist_key;
    SharedCandidateEvidence shared_evidence = SharedCandidateEvidence::None;
    runtime::PrefillWork protected_rebuild_work;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::vector<runtime::CheckpointRecoveryAlternativeWork> projected_recovery_work;
    std::vector<runtime::CheckpointRef> private_replacement_candidates;
    std::uint32_t frontier                = 0;
    bool publishes_private                = false;
    bool publishes_shared                 = false;
    bool needs_transfer                   = false;
    bool physically_feasible              = false;
    bool recycles_private_state           = false;
    CaptureStatePlacement state_placement = CaptureStatePlacement::DeviceFork;
};

// 共享前缀发布成功的回执：能力句柄 + 逻辑摘要。Runtime 之后靠句柄引用它，靠摘要判断"我要的那段前缀
// 是不是就是这段"（共享前缀本身不可变，摘要不会过期）。
struct SharedPrefixPublication {
    SharedPrefixHandle handle;
    SharedPrefixSummary summary;
};

// ---- 资源事务的结果 ----（下面这一组：Program 把事务结束时的**绝对最终状态**报给 Runtime）
//
// 共同约定：
//   * 没有"增量 delta"式的回报，一律是结束时的事实——结束之后 Runtime 拿到的就是真相，不需要重放。
//   * status 的默认值是失败侧（Aborted / InvariantMismatch）：忘记赋值就等于宣称失败，不会静默成功。
//   * 结果里逐条列出被牵连的 owner/victim（即使什么都没发生也如实报告 Retained），而不是只报"成功"。
struct MaterializationVictimResult;
struct MaterializationSharedVictimResult;

// capture 事务的终态：除了通用结果面，还多两个 capture 特有的东西——capacity_preparation_committed
// （为腾地方做的准备是否真的已经生效，没生效的话外部资源一件都没动）和 shared（本次顺带发布出来的
// 共享前缀）。
struct ActiveCaptureResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    bool capacity_preparation_committed      = false;
    ContinuationSummary active_summary;
    std::optional<SharedPrefixPublication> shared;
    std::vector<MaterializationVictimResult> victims;
    std::vector<MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

// 启动成功的唯一产物：一条序列的能力句柄。除了这个句柄，Program 不向 Runtime 暴露任何序列内部状态。
struct StartResult {
    SequenceHandle sequence;
};

// 一个被物化波及的 owner 逐条回报。disposition 说的是**实际**结局（被释放 / 保留 / 其他），
// pressure_committed 说明为它做的空间准备是否已落地；final_summary 只在该 owner 被终止时才有值——
// 也就是"它被牺牲换来了什么"的最终账。
struct MaterializationVictimResult {
    runtime::PlanningOwnerId owner;
    runtime::VictimDisposition disposition = runtime::VictimDisposition::Retained;
    bool pressure_committed                = false;
    std::optional<ContinuationSummary> final_summary;
};

// 共享前缀一侧的同款回报：被波及的对象是共享前缀而不是私有 continuation，因此最终账是
// SharedPrefixSummary（同样不可变）。
struct MaterializationSharedVictimResult {
    runtime::PlanningOwnerId owner;
    runtime::VictimDisposition disposition = runtime::VictimDisposition::Retained;
    bool pressure_committed                = false;
    std::optional<SharedPrefixSummary> final_summary;
};

// 物化的**来源**侧结局：它的资源是按什么模式（保留 / 复用 / 移交）处置的，以及来源被终止时的最终账。
struct MaterializationSourceResult {
    runtime::PrivateSourceMode mode = runtime::PrivateSourceMode::Retain;
    std::optional<ContinuationSummary> final_summary;
};

// 共享前缀来源侧的结局：只有最终账，因为共享前缀没有"私有来源模式"那种处置语义。
struct MaterializationSharedSourceResult {
    std::optional<SharedPrefixSummary> final_summary;
};

// 一次物化（把已有请求的状态变成新请求的起点）的终态。published 为空即失败：没有新序列被发布出去。
// 方向性：这是**结束时的绝对状态**——victims / shared_victims 逐条说明被牵连者究竟被释放了还是保住了，
// transfer_observations 说明跨设备搬运的真实结果，operations 是本次事务的计数面。Runtime 据此更新
// 自己的逻辑视图，而不是自己去推断发生了什么。
struct MaterializationResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    std::optional<StartResult> published;
    std::optional<MaterializationSourceResult> source;
    std::optional<MaterializationSharedSourceResult> shared_source;
    std::vector<MaterializationVictimResult> victims;
    std::vector<MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

// 资源事务推进的三种形态：还在进行中 / 已收尾为一次物化 / 已收尾为一次捕获。Runtime 按 variant 分支
// 处理，因此"没结束"和"结束了"在类型上就不会混淆。
using ContextTransactionProgress =
    std::variant<runtime::ContextTransactionInProgress, MaterializationResult, ActiveCaptureResult>;

// 提交时逐行的结局。speculative 是这一行的投机统计——接受/拒绝的草案数在这里对账，而不是靠调用方
// 自己从 token 里反推。
struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

// 提交一次待处理事务的结果。
// captures 与 rows 按行对齐：prompt 前沿的捕获只有在生成的 Begin 令牌被提交之后才算数，而 CaptureOffer
// 是 move-only 的能力，按行摆放既避免了暴露"临时性"的 prompt 状态，也让"这一行的捕获"无法被错配。
// 注意：捕获机会可以按行给，但"未采纳的待处理事务"全局只有一个——提交之后它才真正被采纳。
struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    // prompt 前沿的捕获只有在生成的 Begin 令牌被提交之后才生效。把 move-only 的能力按行摆放，
    // 既避免了暴露临时性的 prompt 状态，也杜绝了行与能力错配。
    std::array<std::optional<CaptureOffer>, kMaximumConcurrency> captures{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

// 丢弃一次待处理事务的结果：Program 侧不变量与实际消费是否一致。status 默认 InvariantMismatch——
// 调用方忘了判断就会当失败处理，而不是默默认为丢弃成功。
struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

// 结束一条序列的终态：disposition 说明它的物理归宿（释放 / 留成可续跑的 continuation），
// continuation 只有在确实留了续跑点时才非空——这个句柄就是"以后还能从这儿接着跑"的凭据。
struct FinishResult {
    runtime::ConsumeStatus status          = runtime::ConsumeStatus::InvariantMismatch;
    runtime::FinishDisposition disposition = runtime::FinishDisposition::Released;
    GenerationTimings timings;
    SpeculativeStats speculative;
    ContinuationSummary summary;
    std::optional<ContinuationHandle> continuation;
};

// 中止一条序列的终态：没有 continuation——异常路径不留续跑点。
struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

// 释放能力的回执：只回报"消费是否与 Program 侧不变量一致"，因为释放本身不产生新的可观察状态。
struct ReleaseResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
};

// ============================ Program：Qwen3.5 的物理权威 ============================
//
// 层次位置：EngineCore（调度与逻辑驻留策略）→ ResourceManager（逻辑缓存账本）→ **Program** → Ops /
// 存储 / CUDA Graph。ResourceManager 知道"谁在逻辑上占着什么"，Program 知道"这些东西在设备上到底
// 长什么样、放在哪、还能不能被复用"。两者各自记账，靠 ProgramResourceRevision 对齐。
//
// 对外只暴露三样东西：能力句柄（不可伪造、会过期、一次性）、精确评估（seal 时校验，admit 的唯一依据）、
// 结果终态（结束时的绝对状态，不是增量）。因此 Runtime 不需要、也无法窥探 Program 的内部状态——这正是
// 这层能保持"物理事实由一方独占"的原因。
//
// 全局不变量（违反即抛异常，因为那是调用方违约，不是运行期状况）：
//   * 同一时刻至多一个**未采纳的待处理事务**（PendingBatch）和一个**未完成的活动捕获**；
//   * 同一时刻至多一个资源事务，且它必须先结束（提交或中止）才能开下一个；
//   * 封印出的 plan 绑定 resource_revision，start 时校验——对不上就在产生任何物理副作用之前被拒；
//   * Start 之后不得再修改 candidate / target。
// 反面：正常的"做不到"不用异常表达，用 nullopt / status 表达（例如准入候选无可行情景）。
//
// 生命周期：Program 由 create_program 构造一次，之后只被单个 worker 线程驱动（无内部锁）；析构前若还有
// 活动事实，由 fail_all_cleanup 统一收尾。
class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // ---- 规划与准入 ----
    //
    // 流程走向：先 plan_request 定下请求的基线形态（要多少空间、按什么形态跑），再 inspect_admission
    // 针对"落到哪个 lane、从谁那儿复用"给出候选；候选经 seal_identity 封印成可启动的 ResourcePlan，
    // 最后由 start_resource_transaction 真正落地。空间不够时走 begin_pressure_planning 那条探索路径。
    // 注意：Engine 管调度与逻辑驻留策略；Program 管物理 lane、不透明能力、模型状态，以及"同一时刻只有一个
    // 未采纳事务"这条约束。

    // 请求基线：不涉及任何具体 lane 与复用来源，只回答"这条 prompt 按这个配置跑，物理上是什么样"。
    [[nodiscard]] RequestBasePlan plan_request(const PreparedPrompt& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    // 离线打分路径（CausalScoring）：一次前向算出各位置的分数。它不开序列，因此与生成路径共用同一套
    // 物理资源，但不进入调度。
    [[nodiscard]] std::vector<float> causal_score(PreparedPrompt&& prompt,
                                                  std::uint32_t first_target);
    // 准入探查：固定 source（私有/共享/checkpoint）与 destination lane 之后，问"这么做可行吗"。
    // 返回 nullopt 表示不可行——那是正常答案，不是错误。must_retain_private_source 用来声明"来源必须留住"，
    // 它会排除掉那些靠吃掉来源才能成立的情景。
    [[nodiscard]] std::optional<AdmissionCandidate> inspect_admission(
        const PreparedPrompt& prompt, const RequestBasePlan& base, runtime::LaneId destination,
        const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
        std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source);
    // 封印：把候选固化成可启动的 plan，并在此刻绑定 resource_revision。空返回值表示这一刻不可行
    // （例如期间发生了物理变化）。封印之后 candidate 不得再被修改。
    [[nodiscard]] std::optional<ResourcePlan> seal_identity(const AdmissionCandidate& candidate,
                                                            const PreparedPrompt& prompt,
                                                            runtime::FinalScheduleIntent intent);
    // 开启压力探索：把一组候选与两组潜在 donor（私有 continuation / 共享前缀）交给 Runtime，
    // 由它排列探索顺序。会话内的 guidance 只排序、不证明；能不能行最终由 assess 说话。
    [[nodiscard]] PressurePlanningSession
    begin_pressure_planning(std::span<const AdmissionCandidate* const> candidates,
                            std::span<const runtime::PlanningCandidateId> candidate_ids,
                            std::span<const ContinuationHandle* const> private_owners,
                            std::span<const runtime::PlanningOwnerId> private_owner_ids,
                            std::span<const SharedPrefixHandle* const> shared_owners,
                            std::span<const runtime::PlanningOwnerId> shared_owner_ids);
    // 共享前缀捕获在若干前沿处切分 prompt 时的真实 prefill 工作量：让 Runtime 在"值不值得为共享做这件事"
    // 上按物理成本算账，而不是按 token 数想当然。
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                      const PreparedPrompt& prompt,
                                      std::span<const std::uint32_t> frontiers);
    // 落地一个已封印的 plan：这里是**唯一**开始产生物理副作用的地方，也是 revision 校验点。
    // 返回 Reserved 之后，事务就交给 progress_context_transaction 推进。
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                               runtime::CancellationFlagView cancellation);
    // 为"被阻塞的队首"开一份持续回填证明：放这个借用者进来，不会吃掉队首翻身所需的那份最大化额度。
    // 给不出证明就返回 nullopt——回填必须证明无害，不能靠推测。
    [[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(const RequestBasePlan& blocked_head, const ResourcePlan& candidate,
                              std::span<const SequenceHandle> persistent_borrowers) const;

    // ---- 资源事务推进 ----
    // 事务是分步的（可能跨多轮、可被取消），因此 Runtime 反复调用 progress 直到拿到终态；终态之后必须
    // finalize 让 Program 释放事务态。has_context_transaction 用来判断"现在还能不能再开一个"。
    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;

    // ---- 执行 ----
    // 两个执行入口都以"未采纳事务"结尾：advance_prefill 与 decode 采样后返回 PendingBatch，由 Runtime
    // 决定提交还是丢弃。append_forced_tokens 不采样，因此不产生待处理事务。
    // 执行期间发现故障时，failed_timing 会把那时的计时带出来，供只读诊断使用。
    [[nodiscard]] PrefillProgress
    advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing = nullptr);
    // 检查一次捕获机会值不值得要：可选地指定要复用的共享前缀 / 私有 checkpoint 与是否允许发布共享。
    // 返回的评估里 physically_feasible 才是可行性依据。
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                    const SharedPrefixHandle* replacement,
                    std::optional<runtime::CheckpointRef> private_replacement,
                    bool permit_shared_publication) const;
    // 从某个 checkpoint 恢复一个 owner 的备选工作量估计（私有 continuation 与共享前缀各一个重载）：
    // 供 Runtime 在"用 checkpoint 恢复"与"别的手段"之间比较成本。
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const ContinuationHandle& owner,
                             runtime::CheckpointRef checkpoint) const;
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const SharedPrefixHandle& owner,
                             runtime::CheckpointRef checkpoint) const;
    // 为一次捕获开启它专属的压力域（候选不可被当请求准入，见 CapturePressurePlanningSession）。
    [[nodiscard]] CapturePressurePlanningSession
    begin_capture_pressure_planning(const CaptureAssessment& assessment,
                                    std::span<const ContinuationHandle* const> private_owners,
                                    std::span<const runtime::PlanningOwnerId> private_owner_ids,
                                    std::span<const SharedPrefixHandle* const> shared_owners,
                                    std::span<const runtime::PlanningOwnerId> shared_owner_ids);
    // 判断这次捕获要的东西与某个共享前缀是不是同一段（共享前缀不可变，所以是纯粹的身份/摘要比对）。
    [[nodiscard]] bool shared_capture_matches(const CaptureOffer& offer,
                                              const SharedPrefixHandle& shared) const;
    // 明确放弃这次捕获机会（消费掉能力）。跳过是合法结局，不是失败。
    void skip_capture(CaptureOffer&& offer);
    // 启动活动捕获：一次占用 offer、在 offer 的生命周期内生效。
    // with_pressure 版本额外带上一个已封印的扩展计划，用于捕获自身需要腾地方的情形。
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                           const SharedPrefixHandle* replacement,
                           std::optional<runtime::CheckpointRef> private_replacement,
                           bool permit_shared_publication,
                           runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
        const SharedPrefixHandle* replacement,
        std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
        CapturePressurePlan&& pressure, runtime::CancellationFlagView cancellation);
    // 解码一轮：一批序列各按自己的预算前进，采样结果打包成待处理事务交回 Runtime。
    // 预算属于每条请求自己（思考预算会限制模型自生成量），Program 只按预算执行、不替它做判断。
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr);
    // 用调用方给定的、按目标序列归好的 token 行推进每条存活序列。它**不采样**，也不推进采样器 RNG /
    // 出现计数等状态——输出发布与预算核算同样归调用方。这使它成为"把已知 token 喂进去"的通道，
    // 而不是另一种生成入口。
    // 每个可选的执行切分点都相对于该行自身强制 token 段的起点。
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    // ---- 采纳与结束 ----
    // 这一组是"把未采纳事务落定 / 把序列收尾 / 把能力还回去"，都不再产生新的执行。
    //
    // 采纳：decisions 由 Runtime 逐行给出（取消 / 终止 / 继续），Program 只照做并回报每行的实际结局。
    // observation 决定回报的粒度：默认 AllRows 会把每行**累计**的计时与投机统计一并填上（诊断友好，
    // 但有额外拷贝）；产品路径只传 ReleasedRowsOnly——只对已被释放的行填累计值，因为那些请求随行销毁，
    // 这是最后一次能读到它们统计的机会。两者不改变任何状态，只改变回报内容。
    [[nodiscard]] CommitResult
    commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming* failed_timing = nullptr);
    // 丢弃待处理事务：这次生成的 token 全部作废，请求回到提交前的状态。它是 commit 的反面，同样是
    // 一次性消费——pending 被吃掉了。
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    // 正常结束一条序列：按 disposition 决定释放还是留下可续跑的 continuation。
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    // 异常中止一条序列：不留续跑点。与 finish 分开是为了让"正常完成"和"出事了"在调用侧无法混淆。
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    // 显式归还能力：私有续跑点与共享前缀各一个入口。归还即消费句柄，之后的句柄不再有效。
    [[nodiscard]] ReleaseResult release_continuation(ContinuationHandle&& continuation) noexcept;
    [[nodiscard]] ReleaseResult release_shared_prefix(SharedPrefixHandle&& shared) noexcept;
    // 故障收尾：把 Program 里所有存活事实一次性清干净（不发结果、不保留续跑点），用于引擎整体停机或
    // 不可恢复错误。调用后 Program 不再假设任何既有能力句柄有效。
    void fail_all_cleanup() noexcept;

    // ---- 只读诊断 ----
    // 这一组不改变任何物理状态，答案是"这一刻的事实"。resource_revision 是两者的对齐口径：
    // 任何封印都必须绑定它，变了就得重新规划。
    // isolated_request_feasible 回答的是"不借助任何复用，这条请求单独放得下吗"——放不下就是永久不可行，
    // ResourceManager 据此把请求标记为不可行而不是反复重试。
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;
    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    // 私有构造 + 唯一好友工厂：Program 只能由 create_program 造出，因此"Program 一定是在规划定稿之后、
    // 带着一个已确定的 SequencePlan 出生的"这件事由类型系统保证，而不是靠约定。
    explicit Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl> impl_;

    friend std::unique_ptr<Program> create_program(const execution::Parameters&, SequencePlan&&,
                                                   DeviceContext&, const StartupObserver&);
};

namespace detail {

// 唯一被授权触碰能力句柄内部字段的地方。方向：Program 与 Runtime 是两个各自记账的主体，句柄的私有字段
// 不能散落在各处被读写——所有 make_* / owner / lane / epoch / consume 都必须经这道门，于是"谁能造出能力、
// 谁能作废能力"在一处可见、可审。consume 的语义就是把句柄掏空（owner 与代次清零），让一次性无法被绕过。
struct RuntimeContractAccess {
    [[nodiscard]] static SequenceHandle make_sequence(const void* owner, runtime::LaneId lane,
                                                      std::uint64_t epoch) noexcept {
        SequenceHandle out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        return out;
    }

    [[nodiscard]] static ContinuationHandle
    make_continuation(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept {
        ContinuationHandle out;
        out.owner_      = owner;
        out.index_      = index;
        out.generation_ = generation;
        return out;
    }

    [[nodiscard]] static SharedPrefixHandle
    make_shared_prefix(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept {
        SharedPrefixHandle out;
        out.owner_      = owner;
        out.index_      = index;
        out.generation_ = generation;
        return out;
    }

    [[nodiscard]] static CaptureOffer make_capture_offer(const void* owner, runtime::LaneId lane,
                                                         std::uint64_t epoch,
                                                         std::uint64_t id) noexcept {
        CaptureOffer out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        out.id_    = id;
        return out;
    }

    [[nodiscard]] static const void* owner(const SequenceHandle& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static const void* owner(const CaptureOffer& offer) noexcept {
        return offer.owner_;
    }

    [[nodiscard]] static runtime::LaneId lane(const CaptureOffer& offer) noexcept {
        return offer.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const CaptureOffer& offer) noexcept {
        return offer.epoch_;
    }

    [[nodiscard]] static std::uint64_t id(const CaptureOffer& offer) noexcept { return offer.id_; }

    static void consume(CaptureOffer& offer) noexcept {
        offer.owner_ = nullptr;
        offer.id_    = 0;
    }

    [[nodiscard]] static runtime::LaneId lane(const SequenceHandle& handle) noexcept {
        return handle.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const SequenceHandle& handle) noexcept {
        return handle.epoch_;
    }

    [[nodiscard]] static const void* owner(const ContinuationHandle& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static std::uint32_t index(const ContinuationHandle& handle) noexcept {
        return handle.index_;
    }

    [[nodiscard]] static std::uint64_t epoch(const ContinuationHandle& handle) noexcept {
        return handle.generation_;
    }

    static void consume(ContinuationHandle& handle) noexcept {
        handle.owner_      = nullptr;
        handle.generation_ = 0;
    }

    [[nodiscard]] static const void* owner(const SharedPrefixHandle& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static std::uint32_t index(const SharedPrefixHandle& handle) noexcept {
        return handle.index_;
    }

    [[nodiscard]] static std::uint64_t epoch(const SharedPrefixHandle& handle) noexcept {
        return handle.generation_;
    }

    static void consume(SharedPrefixHandle& handle) noexcept {
        handle.owner_      = nullptr;
        handle.generation_ = 0;
    }

    [[nodiscard]] static PendingBatch
    make_pending(const void* owner, std::uint64_t transaction, std::span<const SequenceHandle> rows,
                 std::span<const TokenId> tokens, std::span<const std::int32_t> row_counts,
                 std::uint32_t row_stride, runtime::ExecutionTiming timing) {
        PendingBatch out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_     = tokens;
        out.row_counts_ = row_counts;
        out.row_stride_ = row_stride;
        out.timing_     = timing;
        return out;
    }

    [[nodiscard]] static const void* owner(const PendingBatch& pending) noexcept {
        return pending.owner_;
    }

    [[nodiscard]] static std::uint64_t transaction(const PendingBatch& pending) noexcept {
        return pending.transaction_;
    }

    [[nodiscard]] static std::span<const SequenceHandle>
    rows(const PendingBatch& pending) noexcept {
        return {pending.rows_.data(), pending.row_count_};
    }

    static void consume(PendingBatch& pending) noexcept {
        pending.owner_       = nullptr;
        pending.transaction_ = 0;
        pending.row_count_   = 0;
        pending.tokens_      = {};
        pending.row_counts_  = {};
        pending.row_stride_  = 0;
        pending.timing_      = {};
    }
};

} // namespace detail

// 规划器与 Program 是分开构造的：先规划（可能反复迭代配置），规划定稿后再创建 Program 并交出 SequencePlan。
// 这保证"物理资源按哪个方案铺开"在 Program 出生前就定死，之后不会因为运行期调整而变。
[[nodiscard]] SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                                    DeviceContext& device,
                                                    const EngineOptions& options);

[[nodiscard]] std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                                      SequencePlan&& plan, DeviceContext& device,
                                                      const StartupObserver& startup_observer);

} // namespace ninfer::models::qwen3_5
