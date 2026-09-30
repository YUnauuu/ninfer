#pragma once

#include "runtime/contract/request.h"
#include "core/transfer_work.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

// ============================================================================
// runtime/contract/resources.h —— 上下文资源规划的逻辑/物理契约
// ============================================================================
//
// 回答：调度器挑中一个等待中的请求之后，从哪个可恢复位置开始跑、为了腾空间该保留或丢掉哪些
// **不活跃**的上下文，才能以最低的预测代价得到一个物理可执行、且不破坏任何活跃请求完成保证的终态。
//
// 两侧**刻意不共享物理账本**：ResourceManager 管逻辑（lane、checkpoint 目录、保留策略），
// Program 管物理（State/KV 存储、allocator、placement、引用）。所以这里没有任何"Device 剩余
// 字节数"之类的镜像字段——副本必然与真相不同步，要物理数字就去问 Program。
//
// 两条规则，其余是推论：
//   1) 「提示不是证明」——见 PressurePhysicalGuidance。
//   2) 所有 span 都是**借用** PressurePlanningSession 的 scratch 代，下一次 session 操作即失效，
//      planner 必须当场折叠成自有值。

namespace ninfer::runtime {

// Exact features for the startup-selected static prefill cost model. They describe only the
// suffix rebuilt after a selected prefix and remain separate from Scheduler service work.
//
// 预填的**机器工作量**：启动时由硬件类别与 Text/Vision 配置推出 prefill_signature，据此选定一套
// 系数表，整个进程生命周期内不变——所以这些是精确特征而非估计。只覆盖选中前缀之后重建的那段后缀。
// 与 request.h 的 service_work_quanta（调度轮次）是两套记账，别混。
struct PrefillWork {
    std::uint64_t chunks          = 0;  // 要跑几个预填分块（调度步），空后缀时为 0
    std::uint64_t tokens          = 0;  // 要重算的 token 数（= 后缀长度）
    std::uint64_t attention_pairs = 0;  // 见 make_prefill_work
    std::uint64_t vision_items    = 0;  // 视觉条目数（原样透传，不由长度推导）
    std::uint64_t vision_patches  = 0;  // 同上

    [[nodiscard]] friend constexpr bool operator==(PrefillWork, PrefillWork) noexcept = default;
};

// Exact prefill feature definition for a suffix beginning after prefix_tokens. Attention work is
// prefix*suffix + suffix*(suffix+1)/2 and all arithmetic saturates.
//
// prefix_tokens 是**已复用**的前缀长度，自己不重算所以不进 tokens 字段（但仍要被 attend 到，
// 体现在交叉项里）。参数顺序容易看反：第一个是前缀，不是要干的活。
//
// attention_pairs 的交叉项对 prefix 是**线性**的、后缀内部是因果三角数——这正是前缀复用省算力的
// 原因。全部算术饱和而非回绕：输入长度不可信，饱和只会让方案在排序里显得贵，不会把坏方案误判成
// 便宜可行（代价只影响排序，不影响正确性）。
[[nodiscard]] inline PrefillWork make_prefill_work(std::uint64_t prefix_tokens,
                                                   std::uint64_t suffix_tokens,
                                                   std::uint64_t vision_items,
                                                   std::uint64_t vision_patches,
                                                   std::uint32_t prefill_chunk) noexcept {
    PrefillWork result;
    // ceil(suffix / prefill_chunk)，但**空后缀给 0 而不是 1**：服务工作量那边对空后缀仍算 1 个
    // 服务单元（"轮到你一次"确实发生了），这里数的是机器工作。一个算 0、一个算 1是有意的。
    result.chunks =
        suffix_tokens == 0 || prefill_chunk == 0 ? 0 : 1U + (suffix_tokens - 1U) / prefill_chunk;
    result.tokens                       = suffix_tokens;
    result.vision_items                 = vision_items;
    result.vision_patches               = vision_patches;
    const unsigned __int128 suffix      = suffix_tokens;
    const unsigned __int128 linear      = static_cast<unsigned __int128>(prefix_tokens) * suffix;
    const unsigned __int128 triangular  = suffix * (suffix + 1U) / 2U;
    constexpr unsigned __int128 maximum = ~static_cast<unsigned __int128>(0);
    const unsigned __int128 attention =
        triangular > maximum - linear ? maximum : linear + triangular;
    result.attention_pairs = attention > std::numeric_limits<std::uint64_t>::max()
                                 ? std::numeric_limits<std::uint64_t>::max()
                                 : static_cast<std::uint64_t>(attention);
    return result;
}
// 三类资源的粒度不同，不能混在一个账上。一个 checkpoint 要求**所有**必需类别都满足覆盖，
// 只够一类不算命中。
enum class ContextResourceClass : std::uint8_t {
    State,      // 完整状态镜像，单位是"份"
    MainKV,     // 主 KV 存储，分页的
    BackendKV,  // 所选投机后端自己的 KV，结构与主 KV 不同
};

// **声明顺序是有意义的**：CoalescedTransferWork 用 std::array<TransferWork, 3> 按方向索引，
// 顺序必须与这里一致。
enum class ContextTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    DeviceToDevice,
};

// 一次**已经发生**的搬运的观测记录。与下面 ContextTransferRequirement（尚未发生，用于规划）
// 是配对的两个视角。
struct ContextTransferObservation {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    // **单位随 resource 而变**，这是最容易误读的一点：State 是镜像份数，其余 KV 是字节数。
    // 拿两个不同 resource 的 units 相加没有意义。
    std::uint64_t units                = 0; // State images for State; bytes for typed KV.
    std::uint32_t page_count           = 0;
    TransferWork work;                 // 字节数与拷贝调用次数，见 core/transfer_work.h
    std::uint64_t elapsed_ns = 0;      // 实测耗时
};

// 一次规划中的搬运需求。它需要 operator== 是因为规划器要判断两个需求是否等价，
// 从而合并相同的搬运；纯规划期结构，观测路径不用。
struct ContextTransferRequirement {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                = 0;  // 单位语义同 ContextTransferObservation
    std::uint32_t page_count           = 0;
    TransferWork work;

    [[nodiscard]] friend constexpr bool operator==(ContextTransferRequirement,
                                                   ContextTransferRequirement) noexcept = default;
};

// 纯观测字段，不参与任何决策，只对外汇报（serve 的 JSONL 按区间增量报告）。
// 但它们是排查"上下文缓存为什么没按预期工作"的主要依据。
struct ContextOperationCounts {
    std::uint64_t state_moves            = 0;  // 状态镜像换 placement
    std::uint64_t state_forks            = 0;  // 状态分叉
    std::uint64_t state_restores         = 0;  // 从 Host 恢复回 Device
    std::uint64_t pressure_spill_pages   = 0;  // 因内存压力换出的 KV 页数
    // 前缀共享时非对齐尾页不能原地冻结（后面还要写），得复制一页给新的写入者。
    // **对齐的 frontier 不产生 KV 拷贝**，所以这个计数长期偏高说明请求长度普遍不对齐。
    std::uint64_t partial_tail_cow_pages = 0;
    std::uint64_t historical_fork_hits   = 0;
};

// 目标当前的就绪程度。四个值其实是**两组不同的问题**，不要当进度条读：
//   Ready / NeedsTransfer          —— "现在能不能落地"，前者能，后者要等搬运或压力处理。
//   TemporarilyBlocked             —— 被活跃请求 / lane / 未完成事务挡住。**只在特定事件发生后**
//                                     才需要重查（lane 释放、FIFO 保护变化、事务关闭、修订号变化），
//                                     普通解码推进不触发重算，否则每个 token 都要重跑规划器。
//   PermanentlyInfeasible          —— 孤立 root 的需求本身就超出当前 Engine 合同，等多久都不行。
// 把可恢复的当成永久失败，会把本可运行的请求误判为 blocked。
enum class Readiness : std::uint8_t {
    Ready,
    NeedsTransfer,
    TemporarilyBlocked,
    PermanentlyInfeasible,
};

// 上下文事务的生命周期。只有 Published 算成功，其余两者调用方都要走各自的收尾路径。
enum class ContextTransactionStatus : std::uint8_t {
    InProgress,  // 已建立、正在推进，资源尚未对外生效
    Published,   // 已采用，资源正式归位
    Aborted,     // 需回滚到建立前的状态
};

// 事务**建立阶段**的结果，与运行期状态分开：建立时失败通常前置条件不满足，还没有副作用要回滚；
// InProgress 之后才 Aborted 是有副作用的，需要真正的回滚。
enum class ContextTransactionReserveStatus : std::uint8_t {
    Reserved,
    Aborted,
};

// 空标记类型，用**类型**而不是枚举值表达"事务已建立、进行中"——
// 于是 std::variant 在类型层面就把这种情况与最终结果区分开，调用方不可能忘记处理。
struct ContextTransactionInProgress {};

// 两种事务，区别在于被保护的资源不同，因此取消标志的宿主也不同
//（见 contract/request.h 里 CancellationFlagView）。
enum class ContextTransactionKind : std::uint8_t {
    Materialization,  // 为请求准备/迁移资源，源仍有效直到 commit 或 abort
    ActiveCapture,    // 为活跃请求捕获当前上下文（用于发布 checkpoint）
};

// 比 ReserveStatus 多一档：把"过期"和"坏了"分开。
// 混同两者会导致两种错误——把过期当崩溃（过度反应），或把不变量破坏当过期（掩盖 bug）。
// StalePolicyState 是**正常**结果，只是需要重新决策。
enum class PreflightStatus : std::uint8_t {
    Ready,
    StalePolicyState,
    InvariantFailure,
};

// checkpoint 的种类 = 这个可恢复点在对话语义上的位置。种类决定 frontier 能不能被下一个请求复用：
//   TurnClosure / ResponseReplay —— 有**硬性位置要求**，frontier 必须落在"下一个请求可能替换掉的
//   assistant 后缀"之前。落在会被改写的那段里就不再是前缀，复用它会得到错误的历史。
//   SessionEndpoint / SharedStablePrefix / LongAnchor —— 最新可继续状态 / 多条历史共用的不可变
//   稳定前缀 / 保留策略选出的较早长上下文恢复点，无位置要求。
enum class CheckpointKind : std::uint8_t {
    SessionEndpoint,
    TurnClosure,
    ResponseReplay,
    SharedStablePrefix,
    LongAnchor,
};

// 共享的 checkpoint 需要额外的发布证据才允许创建。
enum class CheckpointScope : std::uint8_t {
    Private,
    Shared,
};

// 一份副本驻留在哪。描述的是**分布**，不是可用性：valid = Device 或 Host 任一侧有完整副本即可；
// Device-ready = 执行所需的**全部**内容都在 Device 上，才能立刻开跑。所以 HostOnly 的 checkpoint
// 有效但要先做 H2D restore（即 NeedsTransfer）——把"有效"当成"可以立刻执行"是最容易犯的推理错误。
enum class ReplicaResidency : std::uint8_t {
    DeviceOnly,
    HostOnly,
    Both,
};

// 保留等级**只用于估值，不提供淘汰顺序**——它通过 owner prior 影响"值不值得留"，但不能写成
// "先淘汰 Disposable 再淘汰 RecentPrivate"这样的流水线，实际淘汰由完整目标的联合效果决定。
// 私有侧权重 Disposable=1 / RecentPrivate=4 / LiveSession=16；共享侧**没有固定乘数**，
// 取决于它有没有带来过实际命中。
enum class RetentionClass : std::uint8_t {
    SharedStable,
    LiveSession,
    RecentPrivate,
    Disposable,
};

// owner = 一个拥有独立可恢复状态的实体。一个 continuation 内可以有多个 checkpoint，
// 它们引用同一地址空间的不同前缀；但**每个 checkpoint 都有自己的完整 StateImage 身份**，
// 不允许靠"前缀重合"来偷懒共享状态。
enum class LogicalOwnerKind : std::uint8_t {
    PrivateContinuation,
    SharedPrefix,
};

// (kind, id) 合起来才唯一。
struct LogicalOwnerKey {
    LogicalOwnerKind kind = LogicalOwnerKind::PrivateContinuation;
    std::uint64_t id      = 0;

    [[nodiscard]] friend constexpr bool operator==(LogicalOwnerKey,
                                                   LogicalOwnerKey) noexcept = default;
};

// catalog 条目的能力句柄。generation 是防复用的关键：slot 会被回收再分配，
// 只记 slot 就分不清"我说的还是那个条目"和"那个 slot 已经属于别人了"。
// slot 默认取 uint32 最大值，充当无效哨兵。
struct CatalogCapability {
    LogicalOwnerKey owner;
    std::uint32_t slot       = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t generation = 0;

    [[nodiscard]] friend constexpr bool operator==(CatalogCapability,
                                                   CatalogCapability) noexcept = default;
};

// 规划期的临时 owner 身份，只在一次规划会话内有效，用来在 guidance/assessment 里指代 owner。
// 与 CatalogCapability（长期身份，带代次）是两种东西，别混用。
struct PlanningOwnerId {
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] friend constexpr bool operator==(PlanningOwnerId,
                                                   PlanningOwnerId) noexcept = default;
};

// 规划期的候选编号（一个 candidate = 一个可采用的 root 或精确 checkpoint 来源）。
struct PlanningCandidateId {
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] friend constexpr bool operator==(PlanningCandidateId,
                                                   PlanningCandidateId) noexcept = default;
};

// Program 侧资源修订号——**物理事实的版本**。owner/引用拓扑变化、非活跃资源 placement 变化、
// 全局空闲/预留容量变化、allocator 几何变化时推进。
//
// **不推进**的时机（最容易误解的一条）：活跃请求在**自己的预留额度之内**推进它已提交的 frontier。
// 那不发生任何物理事实变化——容量早就留好了。若这种情况也推进，每次解码都会让所有缓存规划失效。
struct ProgramResourceRevision {
    std::uint64_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(ProgramResourceRevision,
                                                   ProgramResourceRevision) noexcept = default;
};

// ResourceManager-owned final schedule policy. Program validates and consumes this view while
// sealing the already assessed physical target; ResourcePlan is immutable after that boundary.
//
// sealing 是一条**不可逆边界**：seal 之后的 ResourcePlan 没有任何修改接口，所以"已评估的物理目标"
// 与"实际执行的物理计划"不可能偏离——一旦偏离，前面所有可行性论证的前提就没了。
// shared_capture_frontiers 是借用视图。
struct FinalScheduleIntent {
    std::span<const std::uint32_t> shared_capture_frontiers;
};

// **ConsumeToActive 是所有权转移，不是淘汰**：资源交给活跃请求继续用，并没有被释放，
// 所以不该计入 eviction 统计。误报成淘汰会让压力统计看起来比实际严重得多。
enum class PrivateSourceMode : std::uint8_t {
    Retain,           // 保留来源，不并入活跃序列
    ConsumeToActive,  // 并入活跃序列，来源本身被消耗掉
};

// 这里 Retained 才是"没动它"，Evicted 是真的整个扔掉。
enum class VictimDisposition : std::uint8_t {
    Retained,
    Evicted,
};

// 请求正常终止时资源的最终去向。Released 是**正当终态**而不是降级方案：如果保留**无法**形成一个
// 完整的 continuation，确定性的做法就是 release，而不是让请求卡在 TerminalPending 上等一个永远
// 凑不齐的 checkpoint。所以"发布不了"是正常结果，不是错误。
enum class FinishDisposition : std::uint8_t {
    Catalogued,  // 成功发布了完整 checkpoint，编目保留供以后复用
    Released,    // 释放了整个活跃延续，没留下可复用状态
};

// 指向"某个 owner 的某个可恢复点"的坐标。frontier 是 token 绝对位置。
//
// frontier 与分页边界**不是一回事**：页边界 64 对齐，而有效 frontier 可以落在页内部。
// 反过来，"页边界上能读"不证明"模型能从那里恢复"——恢复需要完整的 StateImage 和全部
// 必需的 typed KV 覆盖，不只是字节齐了。
struct CheckpointRef {
    CheckpointKind kind    = CheckpointKind::SessionEndpoint;
    std::uint32_t frontier = 0;
    // Singleton checkpoint kinds use zero. LongAnchor uses a nonzero, per-continuation slot.
    // 由 resource_manager 强制：LongAnchor 要求 ordinal != 0、不超上限、且 frontier != 0。
    std::uint32_t ordinal = 0;

    [[nodiscard]] friend constexpr bool operator==(CheckpointRef, CheckpointRef) noexcept = default;
};

// 与 ProgramResourceRevision 是**两个不同的类型**，尽管内部都是 uint64——它们量的是不同的
// 东西，绝不该互相赋值，用类型隔离来防混用。
struct Revision {
    std::uint64_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(Revision, Revision) noexcept = default;
};

// 一个物化投影在物理上成不成立。三档是**有序的严重程度**：
//   Feasible          —— 可做
//   Infeasible        —— 容量不够，**有可能**通过压力降级腾地方变可行
//   StructuralInvalid —— 结构上不合法，再怎么腾地方也没用
// 区分后两者直接关系到要不要继续搜：把 StructuralInvalid 当 Infeasible 会浪费大量搜索，
// 反过来会误判本可运行的请求为永久阻塞。
//
// 默认值是**最严重的那一档**（fail-closed）：忘记赋值的路径会留在"不可用"，
// 而不是被误当成可行。
enum class MaterializationPhysicalStatus : std::uint8_t {
    Feasible,
    Infeasible,
    StructuralInvalid,
};

inline constexpr std::size_t kContextTransferDirectionCount = 3;

// 已按方向归并的搬运工作量。不同数组元素之间是**串行**的，不能相加——它们是三个不同的方向，
// 各走各的路径，加起来毫无物理意义。调用方必须在交给成本模型定价之前把共享字段的活合并好。
using CoalescedTransferWork = std::array<TransferWork, kContextTransferDirectionCount>;

// Exact, unpriced machine work for one complete materialization projection. Program owns this
// physical fact; the common search runner applies the immutable planning cost model exactly once.
// `optimistic_candidate_transfers` is ordering evidence only and never proves feasibility.
//
// **unpriced（未定价）**是关键字：Program 产出物理事实（搬多少字节、几次拷贝、还要预填多少）但
// **拿不到成本模型、也不定价**；公共搜索侧用启动时选定的那套不可变模型定价，而且**只定价一次**。
// 于是定价策略可以独立调整，Program 也没法"自己算便宜点"来操纵搜索。
//
// ⚠️ 名字里的 optimistic 是警告：它**只是排序证据**，永远不能用来证明可行性。
struct MaterializationMachineWork {
    CoalescedTransferWork pressure_transfers;               // 为腾空间而做的搬运
    CoalescedTransferWork candidate_transfers;              // 为本次候选做的搬运
    CoalescedTransferWork optimistic_candidate_transfers;   // 仅供排序，见上
    PrefillWork remaining_prefill_work;                     // 选定前缀之后仍需重建的预填
    std::uint32_t reused_prompt_tokens = 0;

    [[nodiscard]] friend constexpr bool
    operator==(const MaterializationMachineWork&,
               const MaterializationMachineWork&) noexcept = default;
};

// One exact recovery recipe. Program enumerates every supported alternative; pricing policy
// selects the cheapest alternative without changing physical legality or the target graph.
//
// 同一个 checkpoint 常有多种恢复路径（从 Host 拷回来、从别的 placement 搬过来、或从 root 重算）。
// 职责划分：Program **枚举全部它支持的备选**，定价策略**只选最便宜的**，不改变物理合法性也不改变
// 目标图。每个备选都带完整 transfer + prefill 特征——不同路径的代价结构不同（搬运 vs 算力），
// 必须都交给同一个成本模型定价，不能直接比。
struct CheckpointRecoveryAlternativeWork {
    CoalescedTransferWork transfers;
    PrefillWork prefill;

    [[nodiscard]] friend constexpr bool
    operator==(const CheckpointRecoveryAlternativeWork&,
               const CheckpointRecoveryAlternativeWork&) noexcept = default;
};

// "身份目标"的评估（这个候选压力降级图的**根**，所有更激进的降级方案从它出发）：
// 只应用引入这个候选所必需的变化，**其他所有可淘汰的非活跃 owner 的 checkpoint 内容与
// placement 保持不变**。一句话："不动任何人、只做必须做的事，这个候选成立吗？"
struct IdentityMaterializationAssessment {
    MaterializationPhysicalStatus physical_status =
        MaterializationPhysicalStatus::StructuralInvalid;
    PrivateSourceMode source_mode = PrivateSourceMode::ConsumeToActive;
    MaterializationMachineWork machine_work;
    // 施加内存压力**是否可能改变**这份机器工作量。为 true = "光看当前这份数字还不够"，搜索得继续。
    bool pressure_may_change_machine_work = false;
    // 是否还值得继续降级。只要还有候选是 expandable 就必须建立压力域继续搜——**不能因为
    // "估计成本高于现有最优"就跳过**：组合动作可能互相抵消拷贝，估计成本不是可靠的下界。
    bool expandable                       = false;
    // Program 实际做了多少步物理投影。**精确计数**，不是估计。
    std::uint64_t projection_work         = 0;
    // 这份评估的全部决定性事实折叠成的 64 位摘要（FNV-1a）：资源修订号、可复用提示长度、
    // remaining_prefill_work 五字段、每项候选搬运的字节数与拷贝次数、physical_status。
    // 用途：判断"两个看起来一样的目标是不是真的同一个"，从而复用已做的评估。
    std::uint64_t assessment_digest       = 0;

    [[nodiscard]] friend constexpr bool
    operator==(const IdentityMaterializationAssessment&,
               const IdentityMaterializationAssessment&) noexcept = default;
};

// ⚠️ 借用视图。语义：这个 checkpoint 在**目标终态下**还剩哪些恢复办法，为空表示它扛不住这次
// 压力、会消失。
struct PressureCheckpointRecoveryImpact {
    PlanningOwnerId owner;
    CheckpointRef checkpoint;
    std::span<const CheckpointRecoveryAlternativeWork> target_recovery_work;
    bool survives = true;

    [[nodiscard]] friend constexpr bool
    operator==(const PressureCheckpointRecoveryImpact&,
               const PressureCheckpointRecoveryImpact&) noexcept = default;
};

// 上面那个的轻量版：只回答"活不活得下来"，不带恢复成本表，省掉枚举恢复配方的开销。
struct PressureCheckpointOutcome {
    PlanningOwnerId owner;
    CheckpointRef checkpoint;
    bool survives = true;

    [[nodiscard]] friend constexpr bool operator==(PressureCheckpointOutcome,
                                                   PressureCheckpointOutcome) noexcept = default;
};

struct PressureOwnerOutcome {
    PlanningOwnerId owner;
    VictimDisposition disposition     = VictimDisposition::Retained;
    // 施加在这个 owner 身上的**物理降级动作条数**（见 pressure_planner.h）：
    //     淘汰整个 continuation ? 1 : 0
    //   + State 变更数 + Main KV 变更数 + Backend KV 变更数 + 丢弃的 checkpoint 数
    // 它数的是动作条数而不是字节数——衡量"影响面"而非"省了多少空间"。也是确定性的平局判据之一。
    std::uint32_t degradation_units   = 0;
    std::uint32_t dropped_checkpoints = 0;

    [[nodiscard]] friend constexpr bool operator==(const PressureOwnerOutcome&,
                                                   const PressureOwnerOutcome&) noexcept = default;
};

// Cheap, target-neutral ordering evidence for an unassessed pressure target.  This is deliberately
// not a feasibility certificate: only PressureTargetAssessment may admit or seal a target.  The
// Program owns the physical projection and the common planner combines the owner outcomes with its
// retention policy.
//
// ⚠️ 「提示（guidance）不是证明」——本文件最重要的一条约束。这些数字**不能** 1) 标记目标可行；
// 2) 证明当前最优就是最优（不能当剪枝依据）；3) 参与物理就绪判断。
//
// 为什么：**组合动作可能互相抵消拷贝**，单个动作代价之和不等于整体代价，所以预测成本**不是**有效
// 下界，"用下界剪枝"那套推理在这里不成立；另外物理缺口为零**也不能代替** publication slot 检查。
// 正确用法是拿它决定**探索顺序**和额度还值不值得花，再对真正在意的目标走完整精确评估。
struct PressurePhysicalGuidance {
    std::uint32_t unsatisfied_constraints   = 0;  // 还剩几个硬约束没满足
    std::uint32_t estimated_remaining_steps = 0;  // 预计还要多少步才能可行
    // 归一化后的剩余量，Q20 定点。用整数而非浮点是为了让排序**确定性可复现**——
    // 浮点累加的舍入差异会让同一输入得到不同决策。
    std::uint64_t normalized_residual_q20   = 0;
    // Aggregate byte relief cannot resolve an extent/ordered-stage geometry question.
    //
    // 为 true 表示有些约束是**几何/次序**性质的（某个具体 extent 放不放得下、有序阶段峰值是否
    // 超限），问的是形状和顺序而不是总量：总量够但形状不对照样执行不了。
    bool requires_exact_feedback = false;
};

// 与 CoalescedTransferWork 一样**按方向分别记账**，不同方向之间不可相加。
struct PressureOwnerRecoveryGuidance {
    PlanningOwnerId owner;
    CoalescedTransferWork additional_restore;
};

// 构造扫描的可恢复游标。三个代次字段合起来保证"接着上次继续扫"是安全的：
// 代次不匹配说明中间发生过变更（例如 session 被 reset），继续扫没有意义。
struct PressureConstructionOptionId {
    std::uint32_t cursor_generation = 0;  // 游标自身的代次，防止拿一个已被重置的游标继续用
    std::uint32_t scan_generation   = 0;
    std::uint32_t index             = 0;
};

// The spans are borrowed from a PressurePlanningSession scratch generation and remain valid only
// until the next session operation.  The common planner folds them immediately into owning values.
//
// ⚠️ 借用规则对下面三个 span 都成立，必须收到即拷贝。
//
// 这是 PressurePhysicalGuidance 的完整版，但**仍然只是提示**：注意 estimated_machine_work 的
// estimated 前缀，以及 recovery_estimate_complete 为 false 时那份恢复代价表不完整、求和 ≠ 整体代价。
struct PressureTargetGuidance {
    PressurePhysicalGuidance physical;
    MaterializationMachineWork estimated_machine_work;
    std::span<const PressureOwnerOutcome> owner_outcomes;
    PlanningCandidateId candidate;
    // 稳定序号，说明见 PressureTargetAssessment::stable_target_ordinal。
    std::uint32_t stable_target_ordinal = 0;
    std::uint32_t degradation_units     = 0;
    std::uint32_t dropped_checkpoints   = 0;
    PrivateSourceMode source_mode       = PrivateSourceMode::ConsumeToActive;
    std::span<const PressureCheckpointOutcome> checkpoint_changes;
    std::span<const PressureOwnerRecoveryGuidance> recovery_estimates;
    bool recovery_estimate_complete = false;
};

// One resumable scan operation: one owner's successor generation or one option summary.
// Guidance spans expire at the next session call; the ID remains valid until choose/reset.
//
// 两种产物的生命周期不同：span **下一次 session 调用就失效**，option ID 一直有效**直到 choose
// 或 reset**。正确做法是马上把 guidance 拷走，只留 option ID 用于下次续扫。
struct PressureConstructionStep {
    std::optional<PressureTargetGuidance> guidance;  // 空表示这一步没有产出（例如已 exhausted）
    PressureConstructionOptionId option;
    bool exhausted = false;
};

// The spans are borrowed from a PressurePlanningSession scratch generation and remain valid only
// until the next session mutation. The common planner folds them immediately into owning values.
//
// PressureTargetAssessment —— **唯一**有资格让目标被采用（admit）或 seal 的精确评估。
// 与 PressureTargetGuidance 的分工：guidance 是廉价排序依据、estimated_*、**不能**采用目标；
// assessment 是权威可行性结论、精确且已实际投影出来、**只有它能**决定用不用。只要某个候选还有
// expandable 的可能，就必须走到这一步拿精确结论，**不能因为提示里的估计成本不划算就跳过它**。
//
// ⚠️ owner_outcomes / checkpoint_impacts 仍是**借用 span**。注意对比：machine_work 是自有值
//（已折叠的事实）而它们是借用的——这个差别就是在提醒你"要留下的东西必须先拷走"。
struct PressureTargetAssessment {
    MaterializationPhysicalStatus physical_status =
        MaterializationPhysicalStatus::StructuralInvalid;
    PrivateSourceMode source_mode = PrivateSourceMode::ConsumeToActive;
    MaterializationMachineWork machine_work;
    std::span<const PressureOwnerOutcome> owner_outcomes;
    std::span<const PressureCheckpointRecoveryImpact> checkpoint_impacts;
    PlanningCandidateId candidate;
    // 在一次规划会话内**确定性地**标识一个目标：给有界账本做标记避免重复评估，并作为最终平局
    // 判据的**最后一项**。它不是地址也不是索引，而是"身份"——相同输入必须给出相同结果，
    // 否则既无法复现也无法写测试。
    //
    // 平局判据顺序（全相等才轮到下一项）：少影响已命中的 checkpoint → 少 owner 淘汰 / checkpoint
    // 删除 / 拷贝次数 / 搬运字节 → 少剩余 Text/Vision 预填 → 多复用提示 token → 当前会话绑定 →
    // 最后才轮到它。
    std::uint32_t stable_target_ordinal = 0;
    std::uint32_t degradation_units     = 0;
    std::uint32_t dropped_checkpoints   = 0;
    std::uint64_t projection_work       = 0;
    std::uint64_t assessment_digest     = 0;
    // 语义同 IdentityMaterializationAssessment::expandable：只要还有候选可展开，搜索就必须继续。
    bool expandable                     = false;
    // 这份评估是不是那个**根最大目标**：root 候选 + 释放全部未受保护的缓存。它是整个有界搜索的
    // **正确性兜底**——连"什么都不复用、能扔的都扔了"都跑不动，这个请求才算真的跑不动。它保证
    // 启发式失败、目标预算耗尽、墙钟超时这些**搜索层面**的失败**不会把本来能跑的请求误判成阻塞**。
    // 因此它**不计入"普通可行种子"**（会污染搜索统计与预算），而是在没有 identity target 可行时
    // 直接充当初始 incumbent。
    //
    // 注意 active 需求、来源必需的覆盖、写入者、事务 pin **始终留在需求集合里**：
    // "释放全部"只针对未受保护的缓存，不是真的清空一切。
    bool root_maximal                   = false;
};

// Target-produced affine reservation curve for one Main KV physical-capacity axis. The byte
// values come from complete target physical layout plans, not from a model geometry formula in
// the common runtime.
//
// 主 KV 容量 ↔ 预留字节数之间是**仿射**的：真实布局里除了"每组页多少字节"还有一大块与页数无关
// 的固定开销，所以形如 `预留字节 = 固定底价 + 页组数 × 每组增量`。两个函数对应这条直线的正反两向。
//
// ⚠️ 这些字节数**来自 Program 给出的完整物理布局方案**，不是公共运行时按模型几何公式算的：Program
// 探测两个相邻布局实例、从差值反推斜率（断言为正），再断言整体确实仿射——"仿射"是被**验证**的性质
// 而不是被假设的，破坏它会直接抛错而不是给出错误容量。
//
// page group 是 allocation / reservation / reference / transfer 四种操作共同的最小 Device 单位，
// 页大小固定 64 token（kPagedKVPageSize）。main_page_groups 就是这池共享主 KV 的自由度。
struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;  // 一个页组装多少 token
    std::uint32_t minimum_main_page_groups           = 0;  // 合法页组数下界
    std::uint32_t maximum_main_page_groups           = 0;  // 合法页组数上界
    std::size_t minimum_device_reservation_bytes     = 0;  // 直线截距：与页数无关的固定开销
    std::size_t bytes_per_additional_main_page_group = 0;  // 直线斜率

    // 页组数 → 预留字节数：min_bytes + (groups - min_groups) * stride，中间做溢出检查。
    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    // 页组数 → 可用 token 数：groups * main_page_tokens。
    // 两个函数都会校验页组数落在 [minimum, maximum] 内并检查曲线自身合法性：越界抛
    // std::invalid_argument，结果超 uint32 抛 overflow_error——传越界值得到的是异常，不是被夹住的结果。
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

// KV 容量的解析结果——启动阶段这整段决策的最终快照。两种模式的取舍不同：
//   Explicit  —— 用户指定 token 数、向上取整到页边界。**不会**被 maximum 夹住（用户说了算），但会
//                校验最终预留不超可用容量；此模式不允许携带自动 headroom。
//   Automatic —— 加载权重、量出剩余显存，直接选**最大的合法页组数**，同时留出
//                automatic_headroom_bytes 不分配（默认 1 GiB）。它**不会**在请求期探测分配或缩放
//                池子——容量在这一刻就定死了。
struct KvCapacityResolution {
    KvCapacityMode mode                              = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;  // 实际选定的页组数
    std::uint32_t maximum_main_page_groups           = 0;  // 曲线上界（供对照，看有没有顶到）
    std::uint32_t resolved_tokens                    = 0;  // = resolved_tokens(main_page_groups)
    std::size_t minimum_runtime_reservation_bytes    = 0;  // 曲线的固定底价
    std::size_t bytes_per_additional_main_page_group = 0;  // 曲线的斜率
    std::size_t runtime_reservation_bytes            = 0;  // 最终预留
    // 权重驻留之后量到的空闲 Device 字节数，是自动模式的输入，也是排查"显存都去哪了"先看的数。
    std::size_t available_after_weights_bytes        = 0;
    // 启动**全部**工作（workspace、CUDA Graph 等）完成之后的剩余量，与上一条的差就是启动开销。
    std::size_t available_after_startup_bytes        = 0;
    std::size_t automatic_headroom_bytes             = 0;
    // 计划内松弛量 = 可用量 - 预留量。**不是**浪费：留一点余量让后续上下文资源调度有腾挪空间，
    // 而不是每次都必须精确凑满。
    std::size_t planned_slack_bytes                  = 0;
};

} // namespace ninfer::runtime
