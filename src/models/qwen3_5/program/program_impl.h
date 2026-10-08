#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "core/host_kv_arena.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/storage/draft_context.h"
#include "models/qwen3_5/program/storage/host_kv_store.h"
#include "models/qwen3_5/program/storage/kv_store.h"
#include "models/qwen3_5/program/storage/state_store.h"
#include "models/qwen3_5/program/prefix_identity.h"
#include "models/qwen3_5/program/planning/resource_projection.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_prefill.h"

#include <algorithm>
#include <cstdint>
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

// ============================================================================
// ProgramImpl —— Qwen3.5 Program 的内部状态与规划入口
//
// 与 program.h 的分工：program.h 给的是对外契约（不透明句柄 + 能力接口），真正的字段全在这里，
// 读接口要两边对着看。本文件被所有实现单元包含（planning/ transactions/ storage/ speculative/
// 顶层 decode/prefill/graphs/context_work），是它们唯一的共享状态定义。
//
// 内容自上而下分四段（下面几处重复打开的命名空间只是分段，不是不同域）：
//   1) 捕获身份与几个小枚举：一次 capture 的"身份"、前端标出的捕获点、投机桥接位置；
//   2) 压力决策词汇表与**物理规划记录**（RequestBasePlan / ResourceCandidateState / 两个候选）：
//      它们是封印的对象，属于某个候选，物理版本一变即作废；
//   3) 运行期状态与 ProgramImpl 本体：序列/共享前缀的账本、两个资源事务、lane 与槽位。类内先是对外
//      入口（请求侧 / 定稿 / 事务 / 捕获 / 生成 / 收尾 / 快照），随后是私有实现，按主题编号 1)–12)，
//      编号只为方便对照，不表示调用顺序；
//   4) PressurePlanningSessionImpl：Program 出借的压力搜索会话（对应 program.h 的同名类）。
//
// 贯穿全文件的不变量（违反即抛异常或 terminate，因为那是调用方违约）：
//   * **槽位代次**：ContinuationSlot / SharedPrefixSlot 各带 generation，句柄的 epoch 必须与之
//     相等，否则句柄已作废；
//   * **lane 代次**：作废一个 lane = 递增 lane_epochs[lane]，于是该 lane 上所有句柄与捕获票据
//     同时失效（换代即一次性作废，不需要逐句柄注销）；
//   * **resource_revision**：物理世界的版本号，任何物理变化递增一次；封印前的 plan 与压力会话里
//     的 target 都绑定在某个版本上，版本一变即全部作废；
//   * 至多一个未结算的上下文事务（context_transaction_ 为 monostate 即没有），至多一个未结算的
//     pending 解码事务；
//   * 捕获候选与请求候选在**类型上**分开（AdmissionCandidateImpl / CapturePressureCandidateImpl），
//     压力搜索因此永远不可能把一次 capture 当成请求封印进调度。
// ============================================================================
namespace ninfer::models::qwen3_5::detail {

// 短名重导出，供本文件的各实现单元使用。RewriteCheckpointKind / ReusePath /
// runtime::CheckpointKind 是三套各说各话的枚举（前端改写语义 / 模型复用路径 / 对外契约大类），
// 下面三个小函数是它们之间唯一的翻译点，别在别处再写一遍映射。
using PreparedPromptData    = qwen3_5::PreparedPromptData;
using RewriteCheckpointKind = qwen3_5::RewriteCheckpointKind;
using RewriteCheckpointSpec = qwen3_5::RewriteCheckpointSpec;

using ReusePath = ninfer::PrefixReusePath;

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ReusePath path) noexcept {
    return path == ReusePath::PrivateTurnClosure || path == ReusePath::PrivateResponseReplay;
}

[[nodiscard]] constexpr ReusePath restore_path(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? ReusePath::PrivateTurnClosure
                                                      : ReusePath::PrivateResponseReplay;
}

[[nodiscard]] constexpr runtime::CheckpointKind
checkpoint_kind(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? runtime::CheckpointKind::TurnClosure
                                                      : runtime::CheckpointKind::ResponseReplay;
}

// 目标里已存在的改写检查点（rewrite checkpoint）该怎么处理：保持原样 / 在已提交前沿上替换 /
// 它只是可选的，丢掉。
enum class RewriteCheckpointDisposition : std::uint8_t {
    RetainExisting,
    ReplaceAtCommittedFrontier,
    DropOptional,
};

// 一次 capture 的"身份"本体。用 const + shared_ptr 的原因：捕获一旦发布成共享前缀，它的身份必须
// 活得比产生它的那个请求更久（请求结束、lane 换代都不影响它），所以内容按不可变来组织，共享而非拷贝。
struct PreparedCaptureBacking {
    std::vector<TokenId> ledger;
    qwen3_5::detail::ResidentPrefixIdentity prefix_identity;
};

struct PreparedCaptureIdentity {
    std::shared_ptr<const PreparedCaptureBacking> backing;
    qwen3_5::PrefixShortlistKey shortlist_key;
    runtime::PrefillWork rebuild_work;

    // 只暴露 frontier 之前的那段前缀；没有 backing 时返回空。空身份是合法状态，但匹配不了任何东西
    // ——prefix_equals 要求两边都有身份。
    [[nodiscard]] std::span<const TokenId> ledger() const noexcept {
        if (!backing || shortlist_key.frontier > backing->ledger.size()) { return {}; }
        return std::span<const TokenId>(backing->ledger).first(shortlist_key.frontier);
    }

    [[nodiscard]] const qwen3_5::detail::ResidentPrefixIdentity* prefix_identity() const noexcept {
        return backing ? &backing->prefix_identity : nullptr;
    }

    [[nodiscard]] bool prefix_equals(const PreparedCaptureIdentity& other) const {
        const std::span<const TokenId> left  = ledger();
        const std::span<const TokenId> right = other.ledger();
        const auto* left_identity            = prefix_identity();
        const auto* right_identity           = other.prefix_identity();
        return left_identity != nullptr && right_identity != nullptr &&
               left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin()) &&
               left_identity->prefix_equals(*right_identity, left.size());
    }
};

// 前端在一个 prompt 上标出的一个候选捕获点：在 frontier 处切一刀。input_order 保留前端给出的顺序
// （外部按它决定优先级），shared 表示它是否要成为共享前缀候选；long_anchor 与 shared_evidence 是
// "值不值得共享"的证据，由前端收集、Program 只负责消费。
struct CaptureGroup {
    std::shared_ptr<const PreparedCaptureIdentity> identity;
    std::optional<RewriteCheckpointKind> rewrite;
    std::uint32_t frontier                  = 0;
    std::uint32_t input_order               = 0;
    bool shared                             = false;
    bool long_anchor                        = false;
    SharedCandidateEvidence shared_evidence = SharedCandidateEvidence::None;
};

// 预填充阶段给投机后端"搭桥"（把 MTP 状态对齐到复用点，之后才谈得上下一步起草）的时机：
// 不打桥 / 在补后缀之前打 / 精确命中时在命中点之后就打。
enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

// 压力决策的词汇表：压力只被允许用这些动作腾地方，不允许凭空"取消某人的状态"。对每一类所有者
// （endpoint / rewrite / shared）都只有同一副三步阶梯——先删设备上的重复副本，再降级到 Host，
// 最后才删 Host 上的副本（越靠后越伤：删 Host 副本意味着那份历史只剩设备上这一份）。
enum class PressureStateDecision : std::uint8_t {
    None,
    DropEndpointDeviceDuplicate,
    DemoteEndpointToHost,
    DropEndpointHostDuplicate,
    DropRewriteDeviceDuplicate,
    DemoteRewriteToHost,
    DropRewriteHostDuplicate,
    DropSharedDeviceDuplicate,
    DemoteSharedToHost,
    DropSharedHostDuplicate,
};

// KV 上的牺牲以"页区间"为单位：分页存储里只有连续区间才是一次有意义的动作，而同一阶梯
// （删设备重复 / 降级到 Host / 删 Host 重复）对 KV 的含义与对状态镜像相同。
enum class PressureKVDecisionKind : std::uint8_t {
    None,
    DropDeviceDuplicate,
    DemoteToHost,
    DropHostDuplicate,
};

struct PressureKVDecision {
    std::uint32_t begin_page    = 0;
    std::uint32_t page_count    = 0;
    PressureKVDecisionKind kind = PressureKVDecisionKind::None;

    [[nodiscard]] friend constexpr bool operator==(PressureKVDecision,
                                                   PressureKVDecision) noexcept = default;
};

// 一个 owner 的完整结局：把 State、主 KV、后端 KV、checkpoint 四类改动合成一份**不可分**的清单，
// 公共调度永远看不到这些物理决策。
// 要点：effect 是这份结局的物理净效果，checkpoint_drop_effect 单独记账——它改变的不是当前占用，
// 而是"以后还能回滚到哪"；transfer_requirements 是这份结局自身需要的搬运（降级到 Host 也要搬）。
struct PressureDecision {
    std::uint64_t id = 0;
    std::vector<PressureStateDecision> state_changes;
    std::vector<PressureKVDecision> main_kv_changes;
    std::vector<PressureKVDecision> backend_kv_changes;
    std::vector<runtime::CheckpointRef> dropped_checkpoints;
    PhysicalDelta checkpoint_drop_effect;
    PhysicalDelta effect;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::uint32_t checkpoint_drops = 0;
    bool evicts_continuation       = false;
    bool shared_owner              = false;

    [[nodiscard]] friend bool operator==(const PressureDecision&,
                                         const PressureDecision&) noexcept = default;
};

// 某个 checkpoint 被牺牲后，它的所有者还剩哪些备选恢复路径（alternative_offset 指向一份共享的
// 工作清单）；survives 说的是这个所有者自己还在不在——两者要分开看：路径没了不等于所有者没了。
struct PressureCheckpointRecoveryProjection {
    runtime::PlanningOwnerId owner;
    runtime::CheckpointRef checkpoint;
    std::uint32_t alternative_offset = 0;
    std::uint32_t alternative_count  = 0;
    bool survives                    = true;
};

// 一次 capture 的评估只需三个数：这次捕获需要多少、会让活跃额度变化多少、能从预备容量里回收多少。
struct CaptureAssessmentImpl {
    PhysicalDemand demand;
    PhysicalDelta active_entitlement_delta;
    PhysicalResources capacity_preparation_removed;
};

// 与目标 lane 无关的那部分请求计划：一个 prompt 要从哪重建、怎么采样、两种 KV 上各占多少页、
// 有哪些捕获候选、前缀摘要是什么。它属于**请求**而不属于 lane，所以同一份 base plan 可以反复拿去
// inspect 不同的 lane；真正落到某个 lane 上的是 AdmissionCandidate。
struct RequestBasePlanImpl {
    runtime::RequestPlanSummary summary;
    detail::PhysicalDemand root_demand;
    runtime::PrefillWork root_rebuild_work;
    std::uint32_t root_rebuild_tail_begin = 0;
    qwen3_5::PreparedContextCache context_cache;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_5::VisionControlPlan> vision_control_plan;
    std::optional<qwen3_5::RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<CaptureGroup> capture_groups;
    std::vector<CaptureGroup> shared_candidates;
    qwen3_5::detail::PrefixShortlistDigests prefix_digests;
    std::uint32_t prefix_identity_tag = 0;
    bool allow_prefix_reuse           = false;
};

// 物化与活跃捕获**共用**的物理规划状态。请求调度字段永远不进这条记录，捕获也不会因此变成一种
// 准入类型。（Program 自己的规划状态，不是 Runtime 的候选摘要。）
struct ResourceCandidateState {
    runtime::RequestPlanSummary summary;
    runtime::IdentityMaterializationAssessment identity_assessment;
    // 这份候选是在哪个物理版本上算出来的。压力会话构造时逐条校验；对不上即"过期候选"，直接拒绝。
    runtime::ProgramResourceRevision planning_revision;
    // 压力结局按候选的"恒等（不施压）峰值"规范化。组合阶段会把 demand 改写成施压后的峰值，若再拿
    // 改写后的 demand 重新生成结局，封印时就是在用另一个问题比对同一份结局。
    detail::PhysicalResources identity_pressure_deficit;
    // 结构上成立的 target 仍可能被 Host extent 的几何形状卡住（哪怕空闲字节总量够）。把被卡住的分配量
    // 显式留着，子 target 才知道该去释放 Host 副本，而不是被误判成"结构不可行"的节点。
    std::size_t blocked_host_allocation_bytes = 0;
    detail::PhysicalDemand demand;
    // 只算"单独消费这个私有 owner 自己"能释放的资源；共享别名刻意不算进来——完整的压力 target 会
    // 单独结算它们共同的引用图。
    detail::PhysicalResources source_resources;
    // 复用路径与它的锚点：从哪份来源接着算、接到哪个前沿为止。
    ReusePath reuse                                  = ReusePath::Root;
    std::uint32_t reuse_base                         = 0;
    RewriteCheckpointDisposition rewrite_disposition = RewriteCheckpointDisposition::DropOptional;
    bool has_source                                  = false;
    bool has_shared_source                           = false;
    std::optional<runtime::CheckpointRef> selected_checkpoint;
    std::uint32_t source_index             = 0;
    std::uint64_t source_generation        = 0;
    std::uint32_t shared_source_index      = 0;
    std::uint64_t shared_source_generation = 0;
    runtime::PrefillWork remaining_prefill_work;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    runtime::PrivateSourceMode source_mode = runtime::PrivateSourceMode::ConsumeToActive;
    detail::PhysicalResources active_optional_resources;
    bool state_fork_required          = false;
    bool text_prefix_fork_required    = false;
    bool backend_prefix_fork_required = false;
    bool needs_transfer               = false;
    // 四组压力选项是**平行数组**（私有/共享各一组 options / owner_ids / indices / generations）：
    // 第 i 个决策必须对上第 i 个 owner id、下标与代次，代次对不上即视为已失效。这是把"当时的决策"
    // 重新对回"当时的 owner"的唯一凭据——owner 可能已经换代，光有下标是不够的。
    std::vector<qwen3_5::detail::PressureDecision> pressure_options;
    std::vector<runtime::PlanningOwnerId> pressure_owner_ids;
    std::vector<std::uint32_t> pressure_indices;
    std::vector<std::uint64_t> pressure_generations;
    std::vector<qwen3_5::detail::PressureDecision> shared_pressure_options;
    std::vector<runtime::PlanningOwnerId> shared_pressure_owner_ids;
    std::vector<std::uint32_t> shared_pressure_indices;
    std::vector<std::uint64_t> shared_pressure_generations;
};

// 落到具体目标 lane 上的候选：在共用骨架之外多了投机桥接、视觉预填充、采样配置、两种 KV 的页额度，
// 以及 destination 的 lane 与代次（于是"这份候选属于哪个 lane 世代"也可校验）。
struct AdmissionCandidateImpl : ResourceCandidateState {
    MtpBridgeMode mtp_bridge = MtpBridgeMode::None;
    bool prepare_mtp         = false;
    std::optional<VisionPrefillPlan> vision;
    std::vector<CaptureGroup> capture_groups;
    std::vector<CaptureGroup> shared_candidates;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    runtime::LaneId destination{};
    std::uint64_t destination_epoch = 0;
    runtime::PrefillWork root_rebuild_work;
    std::uint32_t root_rebuild_tail_begin = 0;
    bool text_retained_tail_release       = false;
    bool backend_retained_tail_release    = false;
};

// 空派生不是啰嗦，是类型隔离：压力会话绑定候选时必须"要么是 admission、要么是 capture"，二者互斥，
// 于是捕获永远不可能被当成请求摘要去检查、或被封印成一次准入混进调度。
struct CapturePressureCandidateImpl : ResourceCandidateState {};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

// 本段把后面各实现单元要用到的名字集中列一遍（其中 CapturePressureCandidate 来自上层命名空间）。
using AdmissionCandidateImpl       = qwen3_5::detail::AdmissionCandidateImpl;
using CapturePressureCandidateImpl = qwen3_5::detail::CapturePressureCandidateImpl;
using ResourceCandidateState       = qwen3_5::detail::ResourceCandidateState;
using RequestBasePlanImpl          = qwen3_5::detail::RequestBasePlanImpl;
using CapturePressureCandidate     = qwen3_5::CapturePressureCandidate;

// 未结算解码事务里一行的性质：Begin 是序列的第一步（还没有已成立的锚点），Ordinary 是普通解码，
// Speculative 是投机轮——它决定这一行的采样结果该怎么解释（普通轮直接接受，投机轮要先验证草稿）。
enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

// base_E / base_S 是本轮开始时的执行前沿与账本前沿。结算时以它们为基准推出新前沿，而不是当场读现值
// ——中间可能已经过去若干次推进，现值未必还是本轮出发时的样子。
struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
};

// 一个请求槽（lane）的生命周期。Empty = 该 lane 空着；Prefilling = 还在补前缀；Active = 可以接下一
// 轮；Pending = 已经下发了一轮、事务尚未结算（此时不允许再提交这一行）；Finishable = 已到终点、只等
// 收尾（finish 才会决定是释放还是把续跑发布出去）。
enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Finishable,
};

// 续跑槽位的状态机：Free → ReservedMaterialization（某个物化事务已经认领它，还没落定）→ Active
// （被某个 lane 占用）→ Catalogued（已发布，Runtime 的目录里有条目，可被后续请求复用）→ Free。
// ReservedMaterialization 是**事务期**状态：事务中止就要把槽位还原，不能留在"半认领"上。
enum class ContinuationSlotRole : std::uint8_t {
    Free,
    ReservedMaterialization,
    Active,
    Catalogued,
};

// 槽位身份 = 下标 + 代次。ContinuationHandle 的 epoch 必须与这里的 generation 相等才算有效，
// 于是"槽位被回收再分配给另一个序列"这件事会让旧句柄自动失效，不需要逐个去注销。
struct ContinuationSlot {
    ContinuationSlotRole role = ContinuationSlotRole::Free;
    std::uint64_t generation  = 1;
};

// 改写检查点：由前端声明、在某个前沿上留下的"可以回到这里重写"的点（回合收束 / 回复重放两种）。
struct RewriteCheckpoint {
    bool valid                 = false;
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
    runtime::PrefillWork rebuild_work;
};

// 长上下文里的恢复锚点：与改写检查点不同，它只是"留存下来的一处历史状态 + 回到它还差多少重算"，
// ordinal 是它在这个序列锚点序列里的次序（释放时要按序校验，见 validate_long_anchor_ordinals）。
struct LongAnchorCheckpoint {
    StateImageHandle state;
    std::uint32_t frontier = 0;
    std::uint32_t ordinal  = 0;
    runtime::PrefillWork rebuild_work;
};

// 一个序列用到的两种 KV 地址空间：主 KV 必有，投机后端自己的 KV 可选（后端不同，结构也不同）。
struct SequenceKVBundle {
    KVAddressSpaceHandle text;
    std::optional<KVAddressSpaceHandle> backend;
};

// 解码图按"形状"组织：profile 描述一段可用的 frontier 区间与 batch，topology 是同一个拓扑类下
// 真正实例化出来的图。installed_profile 记录当前装的是哪一个——换 profile 是要动图的，不是纯查询。
struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

// 一条逻辑序列的**目标模型续跑状态**：产生它的那个请求结束之后，它依然有意义，所以它被刻意与
// 请求生命周期、输出、采样、轮次控制状态分开存放（后者在 RequestControl 里）。
// 物理句柄与账本混住在这一条记录里，是这里最容易踩的地方：**账本可以随便改，物理句柄的使用必须成对**
// ——凡是拿着 state / kv / tail_hidden 这些句柄去释放的地方，都要按 strict 与 best-effort 两套语义之一
// 走，而不是"顺手置空"。
struct SequenceState {
    // 物理句柄：这份续跑占着哪些 KV 地址空间、哪份状态镜像、以及可以留作检查点的隐藏态视图。
    // reserved_state 是"已经为某次 fork 预留、但 fork 还没落定"的目的地（见 settle_state_fork）。
    std::optional<SequenceKVBundle> kv;
    ActiveStateBinding state;
    std::optional<StateImageHandle> rewrite_state;
    std::optional<StateImageHandle> reserved_state;
    Tensor tail_hidden;
    Tensor rewrite_checkpoint_hidden;
    std::uint32_t lane = 0;

    // 这两条前沿是整条记录里最重要的一对不变量，别混用：
    //   * execution_frontier = 设备上**真正算到**哪（KV 与状态实际推进到的位置）；
    //   * ledger_frontier    = 逻辑上**已知并记账**到哪（ledger / prefix_identity / prefix_digests
    //     三个数组的长度都等于它，校验点见 transactions/commit.cpp）。
    // 正常节奏下账本比执行前沿多一个 token（多出来的是已经定下、设备还没算的锚点），但真正做决定时
    // 只认 execution_frontier：检查点、捕获、恢复都以它为准，因为只有它是物理事实。
    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_5::detail::ResidentPrefixIdentity prefix_identity;
    qwen3_5::detail::PrefixShortlistDigests prefix_digests;
    std::int32_t rope_delta               = 0;
    // 三个后端各自的 KV 实际推进度：主 KV、投机后端 KV、以及 DFlash 的上下文前沿。它们可以落后于
    // execution_frontier（本轮刚追加的锚点还没算），但绝不能超过它。
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    // 上一轮起草出来的草稿（下一轮要拿去验证/接受）；endpoint_valid 表示"这个序列有一个可发布的端点"。
    std::array<TokenId, qwen3_5::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool endpoint_valid           = false;
    // 可回滚点：改写检查点一个，长上下文锚点若干个（释放顺序有讲究）。
    RewriteCheckpoint rewrite_checkpoint;
    std::vector<LongAnchorCheckpoint> long_anchors;
    // 这份续跑引用了哪些共享前缀槽。它是共享前缀引用计数的依据：共享槽只有在没有人引用时才能回收。
    std::vector<std::uint32_t> shared_prefix_references;
    // 若这份状态是从某个检查点接回来的，还需要重算多少 token（rebuild_tail_begin 是重算段的起点）。
    runtime::PrefillWork rebuild_work;
    std::uint32_t rebuild_tail_begin = 0;
};

// 一份已发布的共享前缀的物理状态。结构与 SequenceState 类似，但语义不同：它是**公共缓存**而不是谁的
// 续跑，所以没有请求侧字段，只有"被引用多少次"。identity 是它的身份，用来在后续请求里判定命中。
struct SharedPrefixState {
    std::optional<SequenceKVBundle> kv;
    StateImageHandle state;
    std::shared_ptr<const PreparedCaptureIdentity> identity;
    std::uint32_t frontier         = 0;
    std::uint32_t backend_frontier = 0;
    std::int32_t rope_delta        = 0;
    bool tail_hidden_valid         = false;
    runtime::PrefillWork rebuild_work;
    std::uint32_t active_references = 0;
};

// 共享前缀槽的状态机，与续跑槽同构，但多一种事务期状态：ReservedReplacement 表示"这次捕获不是新增
// 一份共享前缀，而是**顶替**已有的那一份"——旧槽位已经被认领、等事务落定后才真正换掉。
enum class SharedPrefixSlotRole : std::uint8_t {
    Free,
    ReservedCapture,
    ReservedReplacement,
    Catalogued,
};

struct SharedPrefixSlot {
    SharedPrefixSlotRole role = SharedPrefixSlotRole::Free;
    std::uint64_t generation  = 1;
};

// 请求/轮次控制状态。它**不随可复用的 SequenceState 保留**：续跑能活过请求，请求控制不能。
// 每个被占用的 request 槽（lane）各有一份实例。
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;
    // 这个请求当前登记的两本资源账：active 是它实打实占着的活跃额度，optional 是它可选占用的部分
    // （捕获带来的预留等）。释放与借用都按这两本账归还，压力侧也靠它们算"这个 lane 欠多少"。
    detail::PhysicalResources active_resources;
    detail::PhysicalResources optional_resources;
    // 请求结束时是否把续跑发布出去（false 表示这次只是借用，收尾时直接释放，不留目录条目）。
    bool publish_continuation = true;

    // 预填充阶段的控制状态：候选捕获点是**逐个**递给外部决策的（pending_capture_offer 就是当前那一张
    // 票据的 id，next_capture 指向下一个待递的捕获点，cursor 是当前的推进位置），因此一次捕获的取舍
    // 不会打断其他捕获点的判断。
    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<execution::VisionPrefillSession> vision;
        std::vector<CaptureGroup> capture_groups;
        std::size_t next_capture            = 0;
        std::uint64_t pending_capture_offer = 0;
        std::uint32_t base                  = 0;
        std::uint32_t cursor                = 0;
        std::uint32_t prompt_tokens         = 0;
        std::uint32_t initial_mtp_extent    = 0;
        double elapsed_seconds              = 0.0;
        bool prepare_mtp                    = false;
        ReusePath reuse                     = ReusePath::Root;
        MtpBridgeMode mtp_bridge            = MtpBridgeMode::None;
    };

    std::optional<Prefill> prefill;
};

// Program 的内部实现体：把①启动期定死的配置②设备资源（arena / 池 / 存储 / 解码器）③运行期账本
// （lane、续跑槽、共享前缀槽、两个事务）④规划入口收在一个对象里。它才是"物理权威"的所在地；
// Runtime 只看得到句柄与快照，看不到这里的字段。
//
// 对外方法的形状（为什么是这几个、而不是更细的）：
//   * 请求侧：plan_request（与 lane 无关的 base plan）→ inspect_admission（落到某条 lane 上）→
//     seal_materialization（组合压力、定稿）→ reserve_materialization（开事务）→
//     progress_context_transaction（推事务）→ finalize_context_transaction（收尾）；
//   * 生成侧：advance_prefill / decode / append_forced_tokens → commit / abort_pending；
//   * 捕获侧：inspect_capture → reserve_active_capture(_with_pressure) → 同样走 progress；
//   * 收尾侧：finish / abort / release_continuation / release_shared_prefix / fail_all_cleanup。
// 单 worker 使用，内部不加锁；"同时只有一个事务"这件事靠状态检查而不是互斥量保证。
class ProgramImpl {
public:
    // 压力恢复投影用的暂存：一次评估里反复用到的三张表（状态镜像落在哪、owner 的投影、checkpoint
    // 的存活情况）放在这里复用，避免每次评估都重新分配。
    struct PressureRecoveryScratch {
        struct StatePlacement {
            StateImageHandle state;
            bool device = false;
            bool host   = false;
        };

        struct OwnerProjection {
            const SequenceState* sequence                     = nullptr;
            const SharedPrefixState* shared                   = nullptr;
            const qwen3_5::detail::PressureDecision* decision = nullptr;
            runtime::PlanningOwnerId owner;
        };

        struct CheckpointProjection {
            qwen3_5::CheckpointSummary checkpoint;
            StateImageHandle state;
            bool survives = true;
        };

        std::vector<StatePlacement> state_placements;
        std::vector<OwnerProjection> owners;
        std::vector<CheckpointProjection> checkpoints;
        std::vector<std::optional<runtime::CheckpointRecoveryAlternativeWork>> direct_work;
        qwen3_5::ContinuationSummary continuation_summary;
    };

    ProgramImpl(const execution::Parameters& parameters, const SequencePlanImpl& plan,
                DeviceContext& device, const StartupObserver& startup_observer);
    ~ProgramImpl() noexcept;

    // —— 请求侧：从"与 lane 无关的计划"到"落到某条 lane 上的候选" ——
    [[nodiscard]] RequestBasePlan plan_request(const PreparedPromptData& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    // 离线段（CausalScoring）：与生成共用同一份 Program 资源，但不占 lane、不走调度，因此单独一个入口。
    [[nodiscard]] std::vector<float> causal_score(PreparedPromptData&& prompt,
                                                  std::uint32_t first_target);
    [[nodiscard]] std::optional<AdmissionCandidate> inspect_admission(
        const PreparedPromptData& prompt, const RequestBasePlan& base, runtime::LaneId destination,
        const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
        std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source);
    // 定稿：把候选与它选中的牺牲者一起组合成一份最终方案（容器容量、压力结局、来源都在这一步定型）。
    // 它是"封印"而不是"再查一遍"——封印之后候选就不该再被改动。
    [[nodiscard]] std::optional<AdmissionCandidate> seal_materialization(
        const AdmissionCandidate& admission, const PreparedPromptData& prompt,
        std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const runtime::PlanningOwnerId> pressure_owner_ids,
        std::span<const qwen3_5::detail::PressureDecision* const> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const runtime::PlanningOwnerId> shared_pressure_owner_ids,
        std::span<const qwen3_5::detail::PressureDecision* const> shared_pressure_options);
    [[nodiscard]] std::unique_ptr<CapturePressureCandidateImpl>
    make_capture_physical_candidate(const CaptureAssessment& assessment) const;
    void select_shared_captures(AdmissionCandidate& candidate, const PreparedPromptData& prompt,
                                std::span<const std::uint32_t> frontiers);
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                      const PreparedPromptData& prompt,
                                      std::span<const std::uint32_t> frontiers);
    // 封印的 plan 带着 resource_revision；真正开事务前要在同一状态上再验一次物理前提（来源还在、
    // 牺牲者还在、页数还够）。这一步不能省：封印与开工之间允许发生别的物理变化。
    [[nodiscard]] runtime::PreflightStatus
    revalidate_materialization(const AdmissionCandidate& plan,
                               const PreparedPromptData& prompt) const;
    // —— 事务侧：开、推、收 ——
    // reserve 会先冻结逻辑账本，再让 Program 动手；Program 拒绝就整体回滚，不留半成品。
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_materialization(AdmissionCandidate&& plan, PreparedPromptData&& prompt,
                            runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool
    persistent_backfill_safe(const RequestBasePlan& blocked_head,
                             const AdmissionCandidate& candidate,
                             std::span<const SequenceHandle> persistent_borrowers) const;
    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    // —— 捕获侧：评估 → 开事务；捕获绝不危及已提交的请求，因此它总是可以退化成"跳过" ——
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                    const SharedPrefixHandle* replacement,
                    std::optional<runtime::CheckpointRef> private_replacement,
                    bool permit_shared_publication) const;
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const ContinuationHandle& owner,
                             runtime::CheckpointRef checkpoint) const;
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const SharedPrefixHandle& owner,
                             runtime::CheckpointRef checkpoint) const;
    [[nodiscard]] bool shared_capture_matches(const CaptureOffer& offer,
                                              const SharedPrefixHandle& shared) const;
    void skip_capture(CaptureOffer&& offer);
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
        CapturePressureCandidate&& pressure, runtime::CancellationFlagView cancellation);
    // —— 生成侧：一轮 = 一个 pending 事务 ——
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing);
    // 结算本轮：commit 把结果落进账本（observation 只决定回报哪些行，不改变状态变化本身），
    // abort_pending 则把这一轮整体丢弃。两者都以 PendingBatch 为凭据，消费掉它。
    [[nodiscard]] CommitResult commit(PendingBatch&& pending,
                                      std::span<const runtime::CommitDecision> decisions,
                                      runtime::CommitObservation observation,
                                      runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    // —— 收尾：结束一个请求、释放一份已发布的缓存、以及设备级失败后的兜底 ——
    // finish 只在 Finishable 上有意义：它决定这份续跑是发布出去还是直接释放；
    // release_continuation / release_shared_prefix 是外部主动放弃一份缓存（消费句柄）。
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    [[nodiscard]] ReleaseResult release_continuation(ContinuationHandle&& continuation) noexcept;
    [[nodiscard]] ReleaseResult release_shared_prefix(SharedPrefixHandle&& shared) noexcept;
    void fail_all_cleanup() noexcept;
    // —— 对外快照：给 ResourceManager 算准入用的容量与可行性、物理占用、以及内存统计 ——
    [[nodiscard]] detail::PhysicalResources admission_capacity() const noexcept;
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;

    // 物理世界的版本号：任何一次物理变化都会推进它。封印出的 plan、压力会话里的 target 都绑定在
    // 某个版本上，因此"版本不同"就是"你手里的东西过期了"的统一判据。
    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return resource_revision_;
    }

    [[nodiscard]] qwen3_5::PhysicalUsageSnapshot physical_usage() const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() noexcept;

    // 压力会话要读私有暂存与版本号，是这里唯一的例外好友（能力句柄那一侧的好友是 program.h 里的
    // RuntimeContractAccess）。
    friend struct qwen3_5::detail::PressurePlanningSessionImpl;

    // —— 启动期定死的配置：都是 const，因为 Program 只会在规划定稿之后出生 ——
    const execution::Parameters& parameters;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const ContextCacheOptions context_cache;
    const std::uint32_t continuation_capacity;
    const std::uint32_t shared_prefix_capacity;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend speculative_backend;
    const KvCacheStorage kv_storage;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const bool causal_scoring;
    const std::size_t kv_payload_bytes;
    const std::size_t graph_allowance_bytes;
    const WorkspacePlan workspace_plan;

    // —— 设备资源 ——
    // persistent 是一次性按 PersistentLayout 切好的常驻分配（解码器状态、状态镜像池、GDN replay 记录、
    // DFlash 常驻态、round state、prefill_hidden 等都在里面，所以下面的 tensor 是它的视图）；
    // workspace_storage + work 是每轮临时用、轮末整体 reset 的工作区。
    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_5::DecoderState> decoder;
    std::unique_ptr<HostKVArena> host_kv_arena;
    std::unique_ptr<LogicalKVPageStore> text_kv_pages;
    // 存储两层：*_pages 管物理页的分配与引用，*_addresses 管"逻辑序列 → 页"的地址空间（绑定、
    // 前缀 fork、快照都作用在地址空间上）；host_kv_extents 是 Host 侧的区间账。
    std::unique_ptr<KVAddressSpaceStore> text_kv_addresses;
    std::unique_ptr<LogicalKVPageStore> backend_kv_pages;
    std::unique_ptr<KVAddressSpaceStore> backend_kv_addresses;
    std::unique_ptr<HostKVExtentStore> host_kv_extents;
    std::size_t text_host_kv_page_stride    = 0;
    std::size_t backend_host_kv_page_stride = 0;
    std::unique_ptr<qwen3_5::StateImageDevicePool> state_images;
    std::unique_ptr<qwen3_5::HostStatePool> host_state_images;
    std::unique_ptr<StateImageStore> state_store;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<ops::GdnReplayFoldPlan> replay_fold;
    std::optional<DFlashPersistentState> dflash;
    qwen3_5::RoundState io;
    Tensor prefill_hidden;
    std::optional<Tensor> score_hidden;
    Tensor sampling_config;
    Tensor token_counts;

    // —— 运行期账本 ——
    // states 与 slots 一一对应：states[i] 是内容，slots[i] 是身份（角色 + 代次）。句柄认的是后者。
    std::vector<SequenceState> continuation_states;
    std::vector<ContinuationSlot> continuation_slots;
    std::vector<SharedPrefixState> shared_prefix_states;
    std::vector<SharedPrefixSlot> shared_prefix_slots;
    // lane → 续跑下标；等于 continuation_capacity 表示这条 lane 当前没有绑定续跑（哨兵值，不是 0）。
    std::array<std::uint32_t, kMaximumConcurrency> active_continuations{};
    std::array<RequestControl, kMaximumConcurrency> requests;
    // lane 的代次。作废一条 lane 就是把它加一，于是该 lane 上所有句柄与捕获票据一次性失效。
    std::array<std::uint64_t, kMaximumConcurrency> lane_epochs{};

    // 三种解码图家族各一份（普通 / MTP / DFlash）。图是 Program 私有资源，与请求无关，只是被开关按
    // 形状选取。
    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;

    // 主机侧钉住的缓冲区：设备图只认设备地址，而采样出的 token、要读回的 logprob 得先落在页锁定内
    // 存上才能被异步拷回来。每种解码路径（普通 / MTP / DFlash）各有一套 ingress/egress 视图，指向
    // 同一块缓冲的不同区域——ingress 是"这一轮喂进去什么"，egress 是"这一轮回什么"。
    std::optional<PinnedHostBuffer> round_host;
    std::optional<PinnedHostBuffer> score_logprobs_host;
    TokenId* host_tokens = nullptr;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_5::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_5::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_5::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_5::MtpDecodeEgress* mtp_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_5::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3_5::DFlashDecodeEgress* dflash_host_egress   = nullptr;

    // 两个峰值计数：workspace_logical_peak_bytes 是本 Program 见过的工作区逻辑峰值（用于核对工作区
    // 规划够不够），vision_handoff_peak_bytes 是多模态交接那一段的峰值——它不属于常规解码路径，单独记，
    // 免得把一次性开销混进解码的账里。
    std::size_t workspace_logical_peak_bytes = 0;
    std::size_t vision_handoff_peak_bytes    = 0;

private:
    void advance_resource_revision() noexcept {
        if (++resource_revision_.value == 0) { ++resource_revision_.value; }
    }

    // resource_revision_ 跳过 0：0 留给"未初始化/无效"的理解，免得有人拿默认构造的值当有效版本比对。
    runtime::ProgramResourceRevision resource_revision_{.value = 1};
    // 压力会话同一时刻至多一个；generation 让旧会话的 target / 游标在会话结束后自动失效。
    std::uint32_t pressure_planning_generation_ = 0;
    bool pressure_planning_active_              = false;

    // —— 压力评估的暂存（mutable 是有意义的：评估是 const 的观测动作，但需要缓存） ——
    // 每次开始新一轮压力规划时清一次页暂存（begin_pressure_page_scratch），因此这里记的是"本轮规划里
    // 我已经为这一页算过什么"，不是长期状态。PressurePageScratchSlot::generation 用来识别陈旧条目。
    // scratch 里同时留着"选中的页/状态"，方便一次评估内反复引用而不重算。
    struct PressurePageScratchSlot {
        std::uint32_t generation     = 0;
        std::uint32_t selected_index = std::numeric_limits<std::uint32_t>::max();
        std::uint64_t host_group     = 0;
        bool projected               = false;
        bool device                  = false;
        bool host                    = false;
        bool pressure_targeted       = false;
    };

    struct PressureSelectedPage {
        LogicalKVPageHandle page;
        std::uint32_t references = 0;
    };

    struct PressureSelectedState {
        StateImageHandle state;
        bool device = false;
        bool host   = false;
    };

    mutable std::uint32_t pressure_page_scratch_generation_ = 0;
    mutable std::vector<PressurePageScratchSlot> pressure_text_page_scratch_;
    mutable std::vector<PressurePageScratchSlot> pressure_backend_page_scratch_;
    mutable std::vector<PressureSelectedPage> pressure_text_selected_pages_;
    mutable std::vector<PressureSelectedPage> pressure_backend_selected_pages_;
    mutable std::vector<std::uint8_t> pressure_private_owner_scratch_;
    mutable std::vector<std::uint8_t> pressure_shared_owner_scratch_;
    mutable std::vector<std::vector<runtime::CheckpointRef>> pressure_private_drop_scratch_;
    mutable std::vector<PressureSelectedState> pressure_state_scratch_;

    void begin_pressure_page_scratch() const noexcept;
    [[nodiscard]] PressurePageScratchSlot& pressure_page_scratch(const LogicalKVPageStore& store,
                                                                 LogicalKVPageHandle page) const;
    [[nodiscard]] const PressurePageScratchSlot*
    find_pressure_page_scratch(const LogicalKVPageStore& store, LogicalKVPageHandle page) const;

    // 物化期间"不能被动机器"的清单：自己的来源（私有/共享）、要消费掉的那份状态、以及一旦释放就会
    // 破坏这次物化的那部分 KV 前缀页。压力搜索拿它排除"牺牲者正好是自己来源"的方案——这正是
    // 压力会话构造时把 source 对应的 owner 从受害候选里剔除的依据。
    struct MaterializationSourceProtection {
        struct StateOwnershipCandidate {
            StateImageHandle state;
            std::uint32_t source_checkpoint_references = 0;
        };

        std::optional<std::uint32_t> private_source_index;
        bool consumed_private_source = false;
        std::optional<StateImageHandle> state;
        std::uint32_t consumed_state_references = 0;
        bool state_fork_required                = false;
        std::vector<StateOwnershipCandidate> state_ownership_candidates;
        std::optional<KVAddressSpaceHandle> text;
        std::uint32_t text_pages          = 0;
        std::uint32_t text_transfer_pages = 0;
        bool text_prefix_fork_required    = false;
        std::optional<KVAddressSpaceHandle> backend;
        std::uint32_t backend_pages          = 0;
        std::uint32_t backend_transfer_pages = 0;
        bool backend_prefix_fork_required    = false;
    };

    // 一次解码轮次的事务身份。lanes 与 epochs 同时记下，是为了让"这一行还是不是当初那一行"可校验：
    // 只要中间有 lane 被作废过（epoch 变了），pending 就整体作废，不能拿旧结果去结算。
    struct PendingTransaction {
        std::uint64_t id = 0;
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<std::uint64_t, kMaximumConcurrency> epochs{};
        std::size_t size = 0;
    };

    std::optional<PendingTransaction> pending_transaction_;
    std::uint64_t next_transaction_id_ = 1;

    // 压力动作的阶段机：先释放 Host 副本腾出空间，再准备搬运、提交、发布，最后才算落地。
    // 之所以要分阶段并记下来，是因为压力动作会跨多次 progress 调用（拷贝要等完成事件），中途不能
    // 装作"已经做完"——每一步都要能从上次停下的地方接着走。
    enum class PressureTransitionPhase : std::uint8_t {
        HostReleases,
        CopyPreparation,
        CopiesInFlight,
        CopyPublication,
        Committed,
    };

    // 三个搬运槽按 ContextResourceClass 索引（State / MainKV / BackendKV），timer_mask 的每个 bit 表示
    // "这一路开了计时"——只给真正发生的搬运计时才需要它。
    struct PressureTransition {
        PressureTransitionPhase phase = PressureTransitionPhase::HostReleases;
        std::array<TransferWork, 3> transfer_work{};
        std::array<std::uint32_t, 3> transfer_pages{};
        std::uint64_t state_images = 0;
        std::uint8_t timer_mask    = 0;
    };

    // 一次物化的全过程状态。分组读：身份与来源 → 牺牲者账 → 压力工作的执行进度 → 物理预订 →
    // 搬运/恢复的暂存 → 阶段标志。整条记录的生命期就是"开事务到结算"，结算后不再有意义。
    struct MaterializationTransaction {
        struct KVRestorePage {
            LogicalKVPageHandle logical;
            HostKVExtentCapability extent;
            std::uint32_t extent_page = 0;
        };

        // 一个牺牲者要执行的压力动作：按四类资源分别记账（状态镜像 / 主 KV / 后端 KV / checkpoint），
        // 每一步都有"做到哪了"的标志——压力动作是分阶段推进的，中途可以停、也可以重来一遍检查。
        struct PressureWork {
            struct StateChangeWork {
                std::optional<StateImageTransfer> transfer;
                bool host_released = false;
            };

            struct KVChangeWork {
                std::vector<LogicalKVPageHandle> pages;
                std::vector<DeviceKVPageHandle> sources;
                std::optional<HostKVExtentReservation> backup;
                bool host_released = false;
            };

            qwen3_5::detail::PressureDecision option;
            std::uint32_t continuation_index      = 0;
            std::uint64_t continuation_generation = 0;
            bool shared_owner                     = false;
            std::vector<StateChangeWork> state_changes;
            std::vector<KVChangeWork> main_kv_changes;
            std::vector<KVChangeWork> backend_kv_changes;
            detail::PhysicalDelta committed_delta;
            bool submitted                 = false;
            bool completed                 = false;
            bool checkpoint_drop_published = false;
            bool mutation_published        = false;
            std::uint64_t spill_pages      = 0;
        };

        std::uint64_t id = 0;
        runtime::LaneId destination;
        bool has_source                        = false;
        bool has_shared_source                 = false;
        runtime::PrivateSourceMode source_mode = runtime::PrivateSourceMode::ConsumeToActive;
        std::uint32_t source_index             = 0;
        std::uint64_t source_generation        = 0;
        std::uint32_t shared_source_index      = 0;
        std::uint64_t shared_source_generation = 0;
        std::optional<MaterializationSourceResult> source_result;
        std::optional<MaterializationSharedSourceResult> shared_source_result;
        // 牺牲者账（私有 / 共享各一套）：下标与代次**成对**保存，因为释放前必须确认"要牺牲的还是不是
        // 当初那一个"；victim_released 记录已经释放了哪些，中途失败时才能接着做完，而不是重做一遍。
        std::vector<std::uint32_t> victim_indices;
        std::vector<std::uint64_t> victim_generations;
        std::vector<bool> victim_released;
        std::vector<PressureWork> pressure;
        std::vector<MaterializationVictimResult> pressure_results;
        std::size_t pressure_cursor = 0;
        std::size_t victim_count    = 0;
        std::vector<std::uint32_t> shared_victim_indices;
        std::vector<std::uint64_t> shared_victim_generations;
        std::vector<bool> shared_victim_released;
        std::vector<MaterializationSharedVictimResult> shared_pressure_results;
        std::vector<PressureWork> shared_pressure;
        std::size_t shared_pressure_cursor = 0;
        std::size_t shared_victim_count    = 0;
        PressureTransition pressure_transition;
        // 打开这个事务时用的那份封印计划（自包含：即使外部把候选改坏了，事务仍按封印时的样子执行）。
        std::optional<AdmissionCandidate> plan;
        std::optional<std::uint32_t> root_continuation_index;
        bool root_waiting_for_victim = false;
        // 物理预订：都是"先占下、再使用、失败要还回去"的资源；reserved_states 最多两个（来源一份、
        // 目的地一份），每一项都必须在结算或中止时归位。
        std::array<StateImageHandle, 2> reserved_states{};
        std::size_t reserved_state_count = 0;
        std::optional<StateImageHandle> state_fork_destination;
        std::optional<KVAddressSpaceHandle> root_text_address;
        std::optional<KVAddressSpaceHandle> root_backend_address;
        std::optional<KVActivationReservation> text_activation;
        std::optional<KVActivationReservation> backend_activation;
        std::optional<DeviceKVPageReservation> text_source_restore_reservation;
        std::optional<DeviceKVPageReservation> backend_source_restore_reservation;
        std::optional<KVPrefixForkReservation> text_prefix_fork;
        std::optional<KVPrefixForkReservation> backend_prefix_fork;
        std::optional<LogicalKVPageHandle> text_retained_tail;
        std::optional<LogicalKVPageHandle> backend_retained_tail;
        std::optional<HostKVExtentReservation> text_retained_tail_backup;
        std::optional<HostKVExtentReservation> backend_retained_tail_backup;
        std::optional<std::uint32_t> text_activation_frontier;
        std::optional<std::uint32_t> backend_activation_frontier;
        std::optional<StateImageTransfer> state_restore;
        bool split_state_identity = false;
        std::vector<KVRestorePage> text_restores;
        std::vector<DeviceKVPageHandle> text_restore_destinations;
        std::vector<KVRestorePage> backend_restores;
        std::vector<DeviceKVPageHandle> backend_restore_destinations;
        // 搬运观测与计数：这些是会回给 Runtime 的账面数字（搬了多少页、多少次恢复/移动/继承），
        // 与物理动作本身分开记——物理做完了不代表账已经报出去。
        std::vector<runtime::ContextTransferObservation> transfer_observations;
        runtime::ContextOperationCounts operations;
        // 阶段标志：prepared 表示"这一轮该准备的都准备好了"，terminal 表示事务已终结（成功或失败），
        // cancel_pending 表示外部已请求取消，剩下几个是各段流水线的"已提交/已就绪"。
        bool state_restored                 = false;
        bool transfer_submitted             = false;
        std::uint8_t transfer_timer_mask    = 0;
        bool prefix_tail_submitted          = false;
        bool retained_tail_backup_submitted = false;
        bool prefix_forks_ready             = false;
        bool source_prepared                = false;
        bool cancel_pending                 = false;
        bool prepared                       = false;
        bool terminal                       = false;
    };

    // 物化专用的两件事：一份"新序列的账本/前缀身份"工作副本（在事务里逐步长出来，落定后才交给
    // SequenceState），以及两个完成事件（来源就绪、整条链完成）。它们属于 Program 而不是事务，因为
    // 事件要在事务之间复用。
    std::uint64_t next_materialization_id_ = 1;
    CudaCompletionEvent context_source_ready_;
    CudaCompletionEvent context_completion_;
    std::vector<TokenId> materialization_ledger_;
    qwen3_5::detail::ResidentPrefixIdentity materialization_identity_;
    qwen3_5::detail::PrefixShortlistDigests materialization_prefix_digests_;

    // 活跃捕获事务：结构与物化事务同构（同样的牺牲者账与压力流水线），但对象是"给正在跑的请求拍快照"。
    // 两个要点：lane_epoch 让"拍到的还是不是当初那条请求"可校验；publish_private / publish_shared /
    // replaces_shared 是这次捕获的**发布意图**——捕获可以只留私有检查点、也可以顺手发布一份共享前缀。
    struct ActiveCaptureTransaction {
        std::uint64_t id         = 0;
        std::uint32_t lane       = 0;
        std::uint64_t lane_epoch = 0;
        CaptureGroup group;
        bool publish_private = false;
        bool publish_shared  = false;
        bool replaces_shared = false;
        std::optional<runtime::CheckpointRef> private_replacement;
        std::optional<std::uint32_t> shared_index;
        std::uint64_t replacement_generation = 0;
        StateImageHandle source_state;
        StateImageHandle destination_state;
        qwen3_5::CaptureStatePlacement state_placement = qwen3_5::CaptureStatePlacement::DeviceFork;
        std::optional<StateImageTransfer> state_snapshot;
        std::optional<KVAddressSpaceHandle> active_text_destination;
        std::optional<KVAddressSpaceHandle> active_backend_destination;
        std::optional<KVActiveSnapshotReservation> text_snapshot;
        std::optional<KVActiveSnapshotReservation> backend_snapshot;
        detail::PhysicalDelta resource_delta;
        detail::PhysicalDelta active_entitlement_delta;
        detail::PhysicalResources capacity_preparation_removed;
        ContinuationSummary active_summary;
        std::vector<runtime::ContextTransferRequirement> transfer_requirements;
        std::vector<runtime::ContextTransferObservation> transfer_observations;
        runtime::ContextOperationCounts operations;
        std::vector<std::uint32_t> victim_indices;
        std::vector<std::uint64_t> victim_generations;
        std::vector<MaterializationTransaction::PressureWork> pressure;
        std::vector<MaterializationVictimResult> pressure_results;
        std::vector<std::uint32_t> shared_victim_indices;
        std::vector<std::uint64_t> shared_victim_generations;
        std::vector<MaterializationTransaction::PressureWork> shared_pressure;
        std::vector<MaterializationSharedVictimResult> shared_pressure_results;
        PressureTransition pressure_transition;
        // recycles_private_state 说的是"这份快照要回收活跃请求自己的状态镜像"（而不是另开一份拷贝），
        // recycled_state_epoch 是那次回收的代次凭据；published 表示结果已经回给 Runtime。
        bool recycles_private_state        = false;
        bool replacement_removed           = false;
        bool prepared                      = false;
        std::uint64_t recycled_state_epoch = 0;
        bool transfer_enqueue_pending      = false;
        bool transfer_submitted            = false;
        std::uint8_t transfer_timer_mask   = 0;
        bool published                     = false;
    };

    std::uint64_t next_capture_offer_id_ = 1;

    // Program 同一时刻至多有一个上下文事务；monostate 就是"没有"。事务类型是互斥的：一次物化与一次
    // 捕获不可能同时在跑（这也是压力会话构造时的前置条件之一）。
    using ContextTransaction =
        std::variant<std::monostate, MaterializationTransaction, ActiveCaptureTransaction>;
    ContextTransaction context_transaction_;

    [[nodiscard]] MaterializationResult
    progress_materialization_transaction(runtime::CancellationFlagView cancellation);
    [[nodiscard]] ActiveCaptureResult
    progress_active_capture_transaction(runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture_impl(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                                const SharedPrefixHandle* replacement,
                                std::optional<runtime::CheckpointRef> private_replacement,
                                bool permit_shared_publication,
                                std::optional<CapturePressureCandidate> pressure,
                                runtime::CancellationFlagView cancellation);

    std::array<CudaEventTimer, 3> context_transfer_timers_;

    // —— 私有实现（按主题分组，与上面的公开入口一一对应）——
    // 1) 落位与物化流水线：inspect_lane 是 inspect_admission 的单 lane 版本；_raw 系列是真正碰设备
    //    的那一层（上面几个入口都是它的包装）；以 prepare_ / enqueue_ / publish_ / abort_ 开头的函数
    //    是同一段流水线的四段，成对出现——有 prepare 就有对应的 abort。
    [[nodiscard]] std::optional<AdmissionCandidate>
    inspect_lane(std::uint32_t lane, const PreparedPromptData& prompt, const RequestBasePlan& base,
                 const SequenceState* source, const SharedPrefixState* shared_source,
                 std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source);
    [[nodiscard]] StartResult start_request(MaterializationTransaction& transaction);
    void prepare_materialization(MaterializationTransaction& transaction);
    void enqueue_materialization_transfers(MaterializationTransaction& transaction);
    void record_materialization_transfer_observations(MaterializationTransaction& transaction);
    void publish_materialization_transfers(MaterializationTransaction& transaction);
    void prepare_prefix_forks(MaterializationTransaction& transaction);
    void prepare_consumed_source(MaterializationTransaction& transaction);
    void abort_materialization_transfers(MaterializationTransaction& transaction) noexcept;
    void prepare_pressure_bookkeeping(MaterializationTransaction::PressureWork& work);
    void prepare_pressure_work(MaterializationTransaction::PressureWork& work,
                               runtime::ContextResourceClass resource);
    void publish_pressure_host_releases(MaterializationTransaction::PressureWork& work);
    void publish_pressure_work(MaterializationTransaction::PressureWork& work) noexcept;
    void abort_pressure_work(MaterializationTransaction::PressureWork& work) noexcept;
    void start_context_transfer_timer(runtime::ContextResourceClass resource);
    void stop_context_transfer_timer(runtime::ContextResourceClass resource);
    [[nodiscard]] runtime::ContextTransferObservation context_transfer_observation(
        runtime::ContextResourceClass resource, runtime::ContextTransferDirection direction,
        TransferWork work, std::uint32_t page_count = 0, std::uint64_t state_images = 1) const;

    struct PhysicalReleaseResult {
        runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
        detail::PhysicalDelta delta;
    };

    [[nodiscard]] PhysicalReleaseResult
    release_materialization_victim(MaterializationTransaction& transaction, std::size_t position);
    void start_sequence(std::uint32_t lane, SequenceState& sequence,
                        MaterializationTransaction& transaction);
    void release_materialization_staging(MaterializationTransaction& transaction) noexcept;
    [[nodiscard]] runtime::PrefillStepResult
    advance_prefill_raw(std::uint32_t lane, runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_raw(std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget> budgets,
               runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming
    resolve_prefill_raw(std::uint32_t lane, bool terminal, runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming resolve_pending_raw(
        std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
        std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> cancelled,
        std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
        runtime::ExecutionTiming* failed_timing);
    // 2) 校验谓词：全部按"句柄 → 所有权 + 代次 + 角色/生命周期"三连检查，不用断言式的"应该没问题"。
    //    materialization_pins 比较特别：它回答的是"这个槽位正被某个进行中的物化掐着吗"，供 Runtime
    //    判断某个空闲槽位此刻是否真的可以被别人拿走。
    [[nodiscard]] bool valid_sequence(SequenceHandle handle) const noexcept;
    [[nodiscard]] bool valid_continuation(const ContinuationHandle& handle) const noexcept;
    [[nodiscard]] bool valid_shared_prefix(const SharedPrefixHandle& handle) const noexcept;
    [[nodiscard]] bool valid_capture_offer(const CaptureOffer& offer) const noexcept;
    [[nodiscard]] bool materialization_pins(std::uint32_t index,
                                            std::uint64_t generation) const noexcept;
    [[nodiscard]] bool has_unsettled_state_fork() const noexcept;
    [[nodiscard]] bool valid_pending(const PendingBatch& pending) const noexcept;
    // 3) 记账：逐 owner 的资源与全局占用是**两套**账。owner 那一套只算"移除它会释放的"资源——别名
    //    共享的分配只有在拆掉这个 owner 真能释放它时才计入，否则会把别人的东西算成自己的。
    [[nodiscard]] detail::PhysicalResources
    owner_exclusive_resources(const SequenceState& sequence) const;
    [[nodiscard]] detail::PhysicalResources
    owner_exclusive_resources(const SharedPrefixState& shared) const;
    [[nodiscard]] detail::PhysicalResources physical_occupancy() const noexcept;
    [[nodiscard]] bool physical_peak_fits(detail::PhysicalResources peak) const noexcept;
    // 4) 来源选择："复用哪一份状态、要不要 fork、能省下多少引用"这三件事必须一起算——选了不改的状态
    //    就要 fork，靠共享省下的引用数决定了资源账能不能对上。
    [[nodiscard]] StateImageHandle
    selected_state(const SequenceState& sequence, ReusePath reuse,
                   std::optional<runtime::CheckpointRef> checkpoint) const;
    [[nodiscard]] std::uint32_t
    selected_state_consumed_references(const SequenceState& sequence, ReusePath reuse,
                                       RewriteCheckpointDisposition rewrite_disposition,
                                       std::optional<runtime::CheckpointRef> checkpoint,
                                       std::uint32_t reuse_base) const;
    [[nodiscard]] bool
    selected_state_requires_fork(const SequenceState& sequence, ReusePath reuse,
                                 RewriteCheckpointDisposition rewrite_disposition,
                                 std::optional<runtime::CheckpointRef> checkpoint,
                                 std::uint32_t reuse_base) const;
    [[nodiscard]] bool can_retain_rewrite_checkpoint(const PreparedPromptData& prompt,
                                                     const RewriteCheckpointSpec& desired,
                                                     const SequenceState& sequence, ReusePath reuse,
                                                     std::uint32_t reuse_base) const;
    // 5) KV 前缀的页数/字节数几个数法（设备上、共享的、缺多少、尾部要不要 CoW），以及由此派生的
    //    checkpoint / 续跑 / 共享前缀摘要——对外报的摘要与内部账本必须同源，否则外部会按错的账做决定。
    [[nodiscard]] std::uint32_t device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                       KVAddressSpaceHandle address,
                                                       std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t shared_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                       KVAddressSpaceHandle address,
                                                       std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                              KVAddressSpaceHandle address,
                                                              std::uint32_t frontier) const;
    [[nodiscard]] bool partial_tail_cow_required(const KVAddressSpaceStore& addresses,
                                                 KVAddressSpaceHandle address,
                                                 std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t
    missing_shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                          KVAddressSpaceHandle address,
                                          std::uint32_t frontier) const;
    [[nodiscard]] std::size_t host_kv_prefix_bytes(const KVAddressSpaceStore& addresses,
                                                   KVAddressSpaceHandle address,
                                                   std::uint32_t frontier) const noexcept;
    [[nodiscard]] qwen3_5::CheckpointSummary
    checkpoint_summary(const SequenceState& sequence, runtime::CheckpointRef checkpoint,
                       StateImageHandle state, runtime::PrefillWork rebuild_work) const;
    [[nodiscard]] qwen3_5::ContinuationSummary
    continuation_summary(const SequenceState& sequence) const;
    void populate_continuation_summary(const SequenceState& sequence,
                                       qwen3_5::ContinuationSummary& summary) const;
    [[nodiscard]] qwen3_5::SharedPrefixSummary
    shared_prefix_summary(const SharedPrefixState& shared) const;
    // 6) 缺口与保护：deficit 有两套数法——裸缺口与"在引导压力生效后的缺口"（guided_），后者用于把
    //    压力结局折算回候选的恒等峰值上比对，二者不能混用。
    [[nodiscard]] std::optional<MaterializationSourceProtection>
    materialization_source_protection(const ResourceCandidateState& candidate) const;
    [[nodiscard]] detail::PhysicalResources
    materialization_deficit(const ResourceCandidateState& candidate) const;
    [[nodiscard]] detail::PhysicalResources
    guided_materialization_deficit(const ResourceCandidateState& candidate,
                                   const detail::PhysicalDelta& pressure) const;
    [[nodiscard]] bool
    protected_materialization_page(const MaterializationSourceProtection* protection,
                                   const KVAddressSpaceStore& addresses, std::uint32_t page_offset,
                                   LogicalKVPageHandle page, bool backend) const;
    // 7) 压力检视：对某个 owner 枚举"在给定缺口下可行的动作"，以及从当前动作继续往下推的后继。
    //    这些函数只**枚举**，不做选择、不改账——选择留给 Runtime 的压力搜索，改账留到事务里。
    [[nodiscard]] std::optional<qwen3_5::detail::PressureDecision>
    inspect_pressure_option(const SequenceState& sequence, detail::PhysicalResources deficit,
                            const MaterializationSourceProtection* protection           = nullptr,
                            const qwen3_5::TargetKVRequirement* retained_requirement    = nullptr,
                            std::span<const runtime::CheckpointRef> dropped_checkpoints = {},
                            std::span<const StateImageHandle> released_states           = {},
                            const qwen3_5::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::vector<qwen3_5::detail::PressureDecision>
    inspect_pressure_successors(const SequenceState& sequence, detail::PhysicalResources residual,
                                const MaterializationSourceProtection* protection,
                                const qwen3_5::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::vector<qwen3_5::detail::PressureDecision> inspect_shared_pressure_successors(
        const SharedPrefixState& shared, detail::PhysicalResources residual,
        const MaterializationSourceProtection* protection,
        const qwen3_5::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::optional<qwen3_5::detail::PressureDecision> inspect_shared_pressure_option(
        const SharedPrefixState& shared, detail::PhysicalResources deficit,
        const MaterializationSourceProtection* protection = nullptr,
        const qwen3_5::detail::PressureDecision* current  = nullptr) const;
    [[nodiscard]] std::vector<qwen3_5::detail::PressureDecision> inspect_shared_pressure_options(
        const SharedPrefixState& shared, detail::PhysicalResources deficit,
        const MaterializationSourceProtection* protection = nullptr,
        const qwen3_5::detail::PressureDecision* current  = nullptr) const;
    [[nodiscard]] qwen3_5::detail::PressureDecision
    inspect_eviction_option(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_5::detail::PressureDecision
    inspect_shared_eviction_option(const SharedPrefixState& shared) const;
    [[nodiscard]] std::optional<qwen3_5::detail::PressureDecision>
    inspect_checkpoint_drop_option(const SequenceState& sequence,
                                   std::span<const runtime::CheckpointRef> checkpoints) const;
    [[nodiscard]] bool
    pressure_decision_valid(const SequenceState& sequence,
                            const qwen3_5::detail::PressureDecision& decision,
                            const MaterializationSourceProtection* protection) const;
    [[nodiscard]] bool
    shared_pressure_decision_valid(const SharedPrefixState& shared,
                                   const qwen3_5::detail::PressureDecision& decision,
                                   const MaterializationSourceProtection* protection) const;
    [[nodiscard]] std::vector<runtime::ContextTransferRequirement>
    checkpoint_restore_requirements(const SequenceKVBundle& kv,
                                    const qwen3_5::TargetKVRequirement& requirement,
                                    StateImageHandle state) const;
    [[nodiscard]] bool pressure_checkpoint_recovery_impacts(
        const ResourceCandidateState& candidate,
        std::span<const ContinuationHandle* const> private_owners,
        std::span<const qwen3_5::detail::PressureDecision* const> private_decisions,
        std::span<const runtime::PlanningOwnerId> private_owner_ids,
        std::span<const SharedPrefixHandle* const> shared_owners,
        std::span<const qwen3_5::detail::PressureDecision* const> shared_decisions,
        std::span<const runtime::PlanningOwnerId> shared_owner_ids,
        std::vector<qwen3_5::detail::PressureCheckpointRecoveryProjection>& output,
        std::vector<runtime::CheckpointRecoveryAlternativeWork>& alternatives,
        PressureRecoveryScratch& scratch, std::uint64_t& projection_work) const;
    void publish_checkpoint_drop(SequenceState& sequence, runtime::CheckpointRef checkpoint);
    // 8) lane 与槽位簿记：invalidate_lane 是"整条 lane 作废"的那个闸门——它一次把该 lane 的代次推进，
    //    于是所有指向这条 lane 上序列/续跑/共享前缀的句柄与 offer 同时失效，不需要逐个去改。清 lane 也
    //    分 strict / best_effort 两档：前者要求"放手后一切干净"（拿不到就失败），后者只求"别崩、别漏"，
    //    用在错误路径上。wrap_ 系列把裸的执行结果翻译成对外可见的进度（并顺手把 pending 记上）。
    [[nodiscard]] PrefillProgress wrap_prefill(std::uint32_t lane, runtime::PrefillStepResult step);
    [[nodiscard]] PendingBatch wrap_pending(std::span<const std::uint32_t> lanes,
                                            const runtime::BatchedGeneratedRound& round);
    void invalidate_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] SequenceState& active_sequence(std::uint32_t lane);
    [[nodiscard]] const SequenceState& active_sequence(std::uint32_t lane) const;
    [[nodiscard]] std::optional<std::uint32_t> allocate_continuation_slot() noexcept;
    [[nodiscard]] bool can_release_continuation_slot_strict(std::uint32_t index) const;
    void release_continuation_slot_strict(std::uint32_t index) noexcept;
    void release_continuation_slot_best_effort(std::uint32_t index) noexcept;
    void retire_continuation_slot(std::uint32_t index) noexcept;
    void clear_execution_failure_lanes(std::span<const std::uint32_t> lanes) noexcept;
    [[nodiscard]] bool can_clear_lane_strict(const SequenceState& sequence) const;
    [[nodiscard]] bool clear_lane_strict(SequenceState& sequence, RequestControl& request) noexcept;
    void clear_lane_best_effort(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    [[nodiscard]] StateImageSelectors state_selectors(const SequenceState& sequence) const;
    [[nodiscard]] detail::PhysicalResources
    sequence_exclusive_state_resources(const SequenceState& sequence) const;
    [[nodiscard]] std::uint32_t owned_checkpoint_references(const SequenceState& sequence,
                                                            StateImageHandle state) const noexcept;
    [[nodiscard]] bool state_exclusive_to_sequence(const SequenceState& sequence,
                                                   StateImageHandle state) const noexcept;
    // 9) 压力候选的组装与评估：compose_ 把"选中的若干个 owner 动作"打包成一个候选状态的快照（含它的
    //    恒等峰值）；evaluate_ 再把这份候选折算成压力目标投影，回答"按这个方案，压力能不能落到位"。
    //    两者都不改真实账本，真正的落地由随后创建的事务负责。
    [[nodiscard]] bool compose_pressure_candidate(
        ResourceCandidateState& candidate,
        std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const runtime::PlanningOwnerId> pressure_owner_ids,
        std::span<const qwen3_5::detail::PressureDecision* const> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const runtime::PlanningOwnerId> shared_pressure_owner_ids,
        std::span<const qwen3_5::detail::PressureDecision* const> shared_pressure_options);
    [[nodiscard]] std::optional<detail::PressureTargetProjection> evaluate_pressure_target(
        const MaterializationSourceProtection* protection,
        std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const qwen3_5::detail::PressureDecision> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const qwen3_5::detail::PressureDecision> shared_pressure_options,
        std::vector<HostKVPageReplicaRelease>* released_host_pages) const;
    void refresh_state_views(SequenceState& sequence);
    void reserve_state_entitlement(SequenceState& sequence, std::uint32_t slots);
    void settle_state_fork(SequenceState& sequence);
    [[nodiscard]] detail::PhysicalResources
    release_checkpoint_reference(StateImageHandle checkpoint) noexcept;
    // 10) 共享前缀与捕获：捕获事务同样是 prepare → enqueue → (publish | abort) 一条流水线；释放侧则
    //     一律成对出现——strict 是"释放并如实报告释放了多少"，best_effort 是错误路径上的兜底（只保证
    //     账本不残留，不承诺结果可用）。
    [[nodiscard]] bool can_release_shared_prefix_state(std::uint32_t index,
                                                       SharedPrefixSlotRole expected_role) const;
    [[nodiscard]] detail::PhysicalResources
    release_shared_prefix_state_strict(std::uint32_t index,
                                       SharedPrefixSlotRole expected_role) noexcept;
    [[nodiscard]] detail::PhysicalResources
    install_private_capture(SequenceState& sequence, const CaptureGroup& group,
                            StateImageHandle checkpoint,
                            std::optional<runtime::CheckpointRef> replacement);
    void prepare_active_capture(ActiveCaptureTransaction& transaction);
    void enqueue_active_capture_transfers(ActiveCaptureTransaction& transaction);
    void abort_active_capture(ActiveCaptureTransaction& transaction) noexcept;
    [[nodiscard]] ActiveCaptureResult publish_active_capture(ActiveCaptureTransaction& transaction);
    void release_active_shared_references_strict(SequenceState& sequence) noexcept;
    void release_active_shared_references(SequenceState& sequence) noexcept;
    void release_active_sequence_state_strict(SequenceState& sequence) noexcept;
    void release_sequence_state_strict(SequenceState& sequence) noexcept;
    void release_sequence_state(SequenceState& sequence) noexcept;
    // 11) 执行本体：图的准备与安装只做一次；copy_ / set_device_i32 是同一条流水线上反复出现的小动作
    //     （把尾部 token 拷到新位置、把标量写进设备张量——都是计算图之外、但必须发生在正确时刻的边角）。
    //     commit_generated_prefix_identity 是本段的关键：它把"这一轮实际算出来的 token"落成前缀身份
    //     （ledger/前缀摘要），也就是把设备事实写进逻辑账本的那一步。
    void prepare_graphs();
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void
    commit_generated_prefix_identity(SequenceState& sequence, std::uint32_t base_ledger_frontier,
                                     std::span<const TokenId> accepted_tokens,
                                     std::optional<std::uint32_t> prefix_execution_split_after);
    [[nodiscard]] runtime::ExecutionTiming
    resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                    std::uint32_t accepted_tokens, bool terminal,
                                    std::optional<std::uint32_t> prefix_execution_split_after,
                                    runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::PrefillStepResult
    advance_prefill(SequenceState& sequence, RequestControl& request,
                    runtime::ExecutionTiming* failed_timing);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets,
                          runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets,
                     runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed_timing);
    // 12) KV 映射：entitlement（我占了多少页的额度）与 mapping（逻辑地址真落到哪些物理页）是两件事，
    //     所以 resize / bind / unbind / trim 分得很细。resize_ 属于前瞻性预留（还没写数据就先把额度占
    //     上，免得写到一半发现不够），commit_ 是执行后确认，release_ 系列则是三档放手。
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void ensure_sequence_kv_mapped(SequenceState& sequence, std::uint32_t main_tokens,
                                   std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    void release_active_sequence_kv_strict(SequenceState& sequence) noexcept;
    void release_sequence_kv_strict(SequenceState& sequence) noexcept;
    void release_sequence_kv(SequenceState& sequence) noexcept;
    void commit_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                            std::uint32_t backend_tokens = 0);
    // backend_* 指的是"文本 KV 之外"的那套后端 KV（MTP / DFlash 各自的上下文缓存），它跟文本 KV 走
    // 不同的进度与生命周期，所以视图要分开取。
    [[nodiscard]] qwen3_5::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_5::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_5::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_5::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

// ============================================================================
// PressurePlanningSessionImpl —— 一次压力规划的会话状态
// 对外契约在 program.h 的 PressurePlanningSession 里（那边已写明：只有 assess 过的 target 才能 seal，
// seal 会把它消费掉；扩展走 prepare → commit/discard 两段；guidance 只是排序提示，不是硬约束）。
// 这里说明内部怎么撑起那套契约：
//   · 会话是"临时账本"：它在真实世界不动的前提下，试着把若干 owner 的压力动作组合起来，算出残余缺口。
//     真正落地由程序侧随后创建的事务完成——会话本身不改任何真实资源。
//   · target 用"候选 + 每个 owner 选了哪个动作"的向量来标识，向量先 intern 进 arena，再压成句柄；
//     同一个标识只对应一个 TargetNode（见 find_target / intern_target / target_hash_table）。
//   · 各种 *_slots 都是租借式缓冲：调用方拿走一个 slot 用，用完归还（generation 用来识别"这个 slot 还
//     是不是我当初租的那一个"）。容量是写死的，租满即失败（抛异常），不扩容。评估结果 slot 尤其严：会话
//     析构时若仍有租借中的评估 slot，直接终止进程——那意味着还有调用方攥着指向会话内部 storage 的视图。
//   · 会话全程绑定 resource_revision：期间真实资源账一变，valid() 就会失败，会话必须重建（见 valid）。
// 本结构是 ProgramImpl 之外唯一的友元，两边的状态相互依赖，改任一侧都要看另一侧。
// ============================================================================
struct PressurePlanningSessionImpl {
    using Core                         = ProgramImpl;
    using AdmissionCandidate           = qwen3_5::AdmissionCandidate;
    using CapturePressureCandidate     = qwen3_5::CapturePressureCandidate;
    using AdmissionCandidateImpl       = qwen3_5::detail::AdmissionCandidateImpl;
    using CapturePressureCandidateImpl = qwen3_5::detail::CapturePressureCandidateImpl;
    using CandidateState               = qwen3_5::detail::ResourceCandidateState;
    using ContinuationHandle           = qwen3_5::ContinuationHandle;
    using SharedPrefixHandle           = qwen3_5::SharedPrefixHandle;

    // 被施压的一方：私有的续跑或共享前缀，两者只填其一（shared 标明是哪种）。
    struct Owner {
        const ContinuationHandle* private_handle = nullptr;
        const SharedPrefixHandle* shared_handle  = nullptr;
        runtime::PlanningOwnerId id;
        bool shared = false;
    };

    // 候选的三份身份证：资源账（谁占了多少）、准入候选（请求侧那次物化）、捕获候选（捕获侧那次捕获）。
    // 同一个 ResourceCandidateState 在两种入口下只会填其中一份，指针为空即是"不是那种候选"。
    struct PhysicalCandidateBinding {
        const CandidateState* state                 = nullptr;
        const AdmissionCandidateImpl* admission     = nullptr;
        const CapturePressureCandidateImpl* capture = nullptr;
    };

    // 一张压力方案（target）的内部记录。victim_choice_* 指向 target_choice_arena 里的一段，就是"每个
    // owner 各选了第几个动作"；assessed_residual 记录评估后的残余缺口；next_expansion_owner 是扩展示
    // 探的进度；stable_ordinal 给同层 target 一个稳定顺序，供对外枚举时排序用；root_maximal 标记"这是
    // 某个根候选的最大方案"（对外的 root_maximal_target）。
    struct TargetNode {
        std::uint32_t candidate_index      = 0;
        std::uint32_t victim_choice_offset = 0;
        std::uint32_t victim_choice_count  = 0;
        std::optional<detail::PhysicalResources> assessed_residual;
        std::uint32_t next_expansion_owner = 0;
        std::uint32_t stable_ordinal       = 0;
        bool root_maximal                  = false;
    };

    // 每个 victim 可选的动作集合，外加一个"逐出"选项的下标。这是构造/枚举时的候选池。
    struct CandidateVictimOptions {
        std::uint32_t owner_index = 0;
        std::vector<PressureDecision> decisions;
        std::uint16_t eviction_choice = 0;
    };

    struct CandidateOptions {
        std::vector<CandidateVictimOptions> victims;
        bool populated = false;
    };

    // 构造过程中"每个 owner 已定下哪个动作"的冻结快照。
    struct PreparedOwnerDecision {
        std::uint32_t candidate_index = 0;
        std::uint32_t victim_index    = 0;
        std::uint16_t choice          = 0;
        PressureDecision decision;
    };

    // 评估结果存放在这里：owner 层面的收效、checkpoint 层面的影响、以及"别的恢复路子要多少额外工作"。
    // 因为评估结果以视图的形式交出去（调用方拿着它枚举），所以 slot 有租借与代次：租出去期间不能被复用，
    // 会话析构时若仍被租借就是泄漏了视图，直接终止。
    struct AssessmentSlot {
        std::vector<runtime::PressureOwnerOutcome> owner_outcomes;
        std::vector<runtime::PressureCheckpointRecoveryImpact> checkpoint_impacts;
        std::vector<runtime::CheckpointRecoveryAlternativeWork> recovery_alternatives;
        std::uint32_t generation = 1;
        bool leased              = false;
    };

    // 构造是"逐个 owner 挑动作"的推进过程。identity 表示"这个选项就是保持原样"，它在枚举里必须存在，
    // 否则"什么都不做"的方案就没法表达。
    struct ConstructionOption {
        std::size_t victim = 0;
        PressureDecision decision;
        bool identity = false;
    };

    // 构造游标指向的现场：choices 是已定下的选择，options 是当前这个 owner 的可选动作，next_owner /
    // next_option 是枚举进度，residual 是这一步之后的剩余缺口，restore 表示这次构造是"恢复"语义
    // （用于 checkpoint 的恢复路径）。scan_generation 让"重新扫一遍"不必清空数组。
    struct ConstructionSlot {
        std::vector<std::uint16_t> choices;
        std::vector<ConstructionOption> options;
        std::size_t next_owner        = 0;
        std::size_t next_option       = 0;
        std::uint32_t candidate_index = 0;
        std::uint32_t generation      = 0;
        std::uint32_t scan_generation = 1;
        detail::PhysicalResources residual;
        bool restore = false;
        bool leased  = false;
    };

    // 构造时就固定了参与者：候选、私有 owner、共享 owner 三组列表，之后不再增删。会话的有效性只取决于
    // Program 的资源版本有没有变（见 valid）。
    PressurePlanningSessionImpl(
        Core& owner, std::span<const PhysicalCandidateBinding> physical_candidates,
        std::span<const runtime::PlanningCandidateId> admission_candidate_ids,
        std::span<const ContinuationHandle* const> private_owners,
        std::span<const runtime::PlanningOwnerId> private_owner_ids,
        std::span<const SharedPrefixHandle* const> shared_owners,
        std::span<const runtime::PlanningOwnerId> shared_owner_ids);
    ~PressurePlanningSessionImpl() noexcept;

    // —— target 的三种起点 ——
    // identity：谁都不动，用它来量化"这个候选本来缺多少"；maximal：某个候选下把能用的压力全用上的方案；
    // root_maximal：某个根候选的整体最优方案（供 Runtime 先看最乐观的情况）。
    [[nodiscard]] qwen3_5::PressureTargetHandle
    identity_target(runtime::PlanningCandidateId candidate) const;
    [[nodiscard]] qwen3_5::PressureTargetHandle
    root_maximal_target(runtime::PlanningCandidateId root_candidate);
    [[nodiscard]] qwen3_5::PressureTargetHandle
    maximal_target(runtime::PlanningCandidateId candidate);
    // —— 构造流水线 ——
    // begin_construction 起一个游标，next_construction_option 逐个吐选项，choose_construction 定一个，
    // 反复直到所有 owner 都定完（residual 收敛）或调用方放弃。construction_target 取游标当前对应的
    // target（后续可拿去 assess / seal），construction_residual 在不改动游标的前提下试算残余。
    [[nodiscard]] qwen3_5::PressureConstructionCursor
    begin_construction(qwen3_5::PressureTargetHandle target, bool restore = false);
    [[nodiscard]] runtime::PressureConstructionStep
    next_construction_option(qwen3_5::PressureConstructionCursor& cursor);
    void choose_construction(qwen3_5::PressureConstructionCursor& cursor,
                             runtime::PressureConstructionOptionId option);
    [[nodiscard]] std::optional<qwen3_5::PressureTargetHandle>
    construction_target(const qwen3_5::PressureConstructionCursor& cursor);
    [[nodiscard]] ConstructionSlot&
    construction_slot(const qwen3_5::PressureConstructionCursor& cursor);
    static void release_construction(const void*, std::uint32_t, std::uint32_t) noexcept;
    // guidance 只表达"下一步怎么走更划算"，Runtime 可以不采纳；它内部会临时试算，但不落任何状态。
    [[nodiscard]] runtime::PressureTargetGuidance
    guidance_choices(std::uint32_t candidate_index, std::span<const std::uint16_t> choices,
                     std::uint32_t ordinal,
                     std::optional<std::size_t> override_owner = std::nullopt,
                     const PressureDecision* override_decision = nullptr);
    [[nodiscard]] detail::PhysicalResources
    construction_residual(std::uint32_t candidate_index,
                          std::span<const std::uint16_t> choices) const;
    [[nodiscard]] runtime::PressureTargetGuidance guidance(qwen3_5::PressureTargetHandle target);
    // assess 是"把这张方案真正算一遍"：算出残余缺口、每个 owner 的收效、以及它对 checkpoint 恢复的
    // 影响。结果放进 assessment_slots 并租给调用方。seal 则把已评估的方案定稿成一个准入候选 / 捕获
    // 候选（这时才算"这次压力规划有产出"）；没评估过的 target 不能 seal。
    [[nodiscard]] qwen3_5::AssessedPressureTarget assess(qwen3_5::PressureTargetHandle target);
    // 扩展（往下多考虑一个 owner）分两段：prepare 期间探测并算好，commit 才把子节点真正并进 targets；
    // 中途放弃就走 discard，不留痕迹。这是"试探不污染会话状态"的实现方式。
    [[nodiscard]] qwen3_5::PreparedPressureExpansion
    prepare_expansion(qwen3_5::PressureTargetHandle parent,
                      std::uint32_t maximum_owners = std::numeric_limits<std::uint32_t>::max());
    [[nodiscard]] qwen3_5::PressureExpansionView
    commit_expansion(qwen3_5::PreparedPressureExpansion&& prepared);
    void discard_expansion(qwen3_5::PreparedPressureExpansion&& prepared) noexcept;
    // 共享前缀被捕获切开之后，重建前缀要多少预填充工作量——这个数会跟着候选一起报给 Runtime。
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const qwen3_5::AssessedPressureTarget& assessed,
                                      const PreparedPromptData& prompt,
                                      std::span<const std::uint32_t> frontiers) const;
    [[nodiscard]] std::optional<AdmissionCandidate> seal(qwen3_5::AssessedPressureTarget&& assessed,
                                                         const PreparedPromptData& prompt,
                                                         runtime::FinalScheduleIntent intent);
    [[nodiscard]] std::optional<CapturePressureCandidate>
    seal_capture(qwen3_5::AssessedPressureTarget&& assessed);

    // —— 会话内部辅助 ——
    // valid 除了检查句柄本身，还会回头核对 program->resource_revision()：真实资源账一变，之前算出来的
    // 方案就不作数了，会话必须重建。
    [[nodiscard]] bool valid(qwen3_5::PressureTargetHandle target) const noexcept;
    [[nodiscard]] std::uint32_t candidate_index(runtime::PlanningCandidateId candidate) const;
    [[nodiscard]] std::span<const std::uint16_t> victim_choices(const TargetNode& target) const;
    [[nodiscard]] TargetNode* find_target(std::uint32_t candidate_index,
                                          std::span<const std::uint16_t> choices) noexcept;
    [[nodiscard]] const TargetNode*
    find_target(std::uint32_t candidate_index,
                std::span<const std::uint16_t> choices) const noexcept;
    // target 按"候选 + 选择向量"去重：intern_target 找到就复用、找不到才新建节点（对应 handle 里的
    // target 下标）；choices 存进 target_choice_arena，finder 用 target_hash_table 定位。
    [[nodiscard]] std::uint32_t intern_target(std::uint32_t candidate_index,
                                              std::span<const std::uint16_t> choices,
                                              bool root_maximal = false);
    void index_target(std::uint32_t target_index);
    void populate_options(std::uint32_t candidate_index);
    [[nodiscard]] std::vector<PressureDecision>
    pressure_successors(const CandidateVictimOptions& victim_options,
                        const detail::PhysicalResources& residual,
                        const Core::MaterializationSourceProtection& protection,
                        const PressureDecision* current) const;
    // 评估 slot 的租借与归还。释放是静态函数，因为销毁 assessment 视图时手里只有会话与 slot 号，不保证
    // 还有可用的会话对象生命周期——这让 Runtime 侧的视图析构不必依赖会话还活着。
    [[nodiscard]] std::uint32_t acquire_assessment_slot();
    static void release_assessment_slot(const void* owner, std::uint32_t slot,
                                        std::uint32_t generation) noexcept;

    // program 用裸指针：会话由 Program 创建并管理，生命周期包含在 Program 之内。
    Core* program = nullptr;
    // resource_revision 是会话建立时抓下的真实资源版本，之后每次 valid() 都拿它跟 Program 当前的比。
    // generation / scratch_generation 分别是本会话自己的版本号与临时区版本号（后者用于"临时算完再决定
    // 要不要留下"的场景，见 prepared_* 一组字段）。
    runtime::ProgramResourceRevision resource_revision;
    std::uint32_t generation         = 1;
    std::uint32_t scratch_generation = 1;
    // —— 会话参与者（构造时固定，之后只读）——
    std::vector<PhysicalCandidateBinding> candidates;
    std::vector<runtime::PlanningCandidateId> candidate_ids;
    std::vector<Owner> owners;
    // —— target 图 ——
    // candidate_options 是每个候选可用的动作池；targets 是已 intern 的节点，向量下标即句柄里的下标；
    // target_choice_arena 按段存放各节点的选择向量（节点里只记 offset/count）；choice_scratch 是拼接新
    // 选择向量的临时区；target_hash_table 提供定位；expansion_scratch / committed_children 供扩展两段
    // 式流程暂存子节点，prepared_owner_decisions 记录探测过程中已定的 owner 决策。
    std::vector<CandidateOptions> candidate_options;
    std::vector<TargetNode> targets;
    std::vector<std::uint16_t> target_choice_arena;
    std::vector<std::uint16_t> choice_scratch;
    std::vector<std::uint32_t> target_hash_table;
    std::vector<TargetNode> expansion_scratch;
    std::vector<PreparedOwnerDecision> prepared_owner_decisions;
    std::vector<qwen3_5::PressureTargetHandle> committed_children;
    // —— 三个平行的选择视图（私有 / 共享各一套）——
    // 这三条数组必须等长、且按下标一一对应：句柄、owner id、选定的压力动作。它们正是传给 Program 侧
    // compose_pressure_candidate / pressure_checkpoint_recovery_impacts 的入参形态，也是"一次压力方案"
    // 的对外表达。recovery_* 那套是同样结构，但用在恢复语义（restore）下。
    std::vector<const ContinuationHandle*> selected_private_owners;
    std::vector<runtime::PlanningOwnerId> selected_private_owner_ids;
    std::vector<const PressureDecision*> selected_private_decisions;
    std::vector<const SharedPrefixHandle*> selected_shared_owners;
    std::vector<runtime::PlanningOwnerId> selected_shared_owner_ids;
    std::vector<const PressureDecision*> selected_shared_decisions;
    std::vector<const ContinuationHandle*> recovery_private_owners;
    std::vector<const PressureDecision*> recovery_private_decisions;
    std::vector<runtime::PlanningOwnerId> recovery_private_owner_ids;
    std::vector<const SharedPrefixHandle*> recovery_shared_owners;
    std::vector<const PressureDecision*> recovery_shared_decisions;
    std::vector<runtime::PlanningOwnerId> recovery_shared_owner_ids;
    // —— 评估与 guidance 的暂存 ——
    // assessment_* 这组是评估时的中间结果，最终由 acquire_assessment_slot() 拷进 slot 再租出去；
    // guidance_* 是 guidance 的同类暂存。recovery_scratch 是 Program 侧那套恢复计算的复用缓冲（会话
    // 借它跑，避免每次分配）。
    std::vector<const PressureDecision*> projected_owner_decisions;
    std::vector<runtime::PressureOwnerOutcome> assessment_outcomes;
    std::vector<PressureCheckpointRecoveryProjection> assessment_impact_projections;
    std::vector<runtime::CheckpointRecoveryAlternativeWork> assessment_recovery_alternatives;
    Core::PressureRecoveryScratch recovery_scratch;
    std::vector<runtime::PressureOwnerOutcome> guidance_outcomes;
    std::vector<runtime::PressureCheckpointOutcome> guidance_checkpoint_changes;
    std::vector<runtime::PressureOwnerRecoveryGuidance> guidance_recovery;
    // 固定容量的两套 slot：4 个构造游标、2 个评估结果。数量写死是因为对外契约不允许同时有更多——租满时
    // 申请抛异常而不是扩容，调用方必须按契约及时释放。prepared_* / scratch_* 是构造期间的进度游标。
    std::array<ConstructionSlot, 4> construction_slots;
    std::uint32_t construction_generation = 0;
    std::array<AssessmentSlot, 2> assessment_slots;
    std::uint32_t prepared_new_count = 0;
    std::uint32_t prepared_owner_end = 0;
    std::size_t scratch_choice_mark  = 0;
    bool scratch_live                = false;
};



} // namespace ninfer::models::qwen3_5::detail
