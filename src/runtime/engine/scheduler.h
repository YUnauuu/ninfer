#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/engine/admission_policy.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// runtime/engine/scheduler.h —— 准入与排班的裁判（模型无关）
// ============================================================================
//
// Scheduler 是 EngineCore 的"何时让谁运行"的那一半脑子：一半管**准入**（排队的人谁能进 active），
// 一半管**排班**（这一轮执行单位跑什么）。它夹在两个都掌握事实的角色之间：
//
//     EngineCore（队列 / active lane / 执行循环）
//          │ ① 告知：这一轮的队首是谁
//          ▼
//      Scheduler ──② 发准入证（grant）──▶ EngineCore 拿证去让 ResourceManager 真正预留
//          ▲                                        │
//          └────── ③ 兑现或作废（commit_admission）◀─┘   物理可行性由 Program 的 proof 负责
//
// 关键方向：**Scheduler 不持有任何物理事实**。它不碰 KV、不碰 allocator、不知道还剩多少显存，
// 只记三件事——当前队首是谁、队首身上开着哪一份保护、有没有请求占着 prefill 归属。容量与可行性
// 一律由资源侧证明（proof 上带一个 ProgramResourceRevision），Scheduler 只负责把**调度身份**与
// 那份证明绑在一起，并在兑现时重验它没有被推翻。
//
// 它要解决的核心张力是 FIFO 顺序与利用率之间的冲突：队首因为资源不足进不来时，干等会浪费卡，随便
// 放人进来又会让 FIFO 失去意义。解法是"**保护期 + 冻结 donor + 回填资格**"：
//
//   * 队首被挡住时，冻结一份"当前 active 集合"的身份快照（donor），含义是：**只有这批人的释放才
//     可能让队首可跑**。此后拓扑怎么变，这份名单与保护期编号（epoch）都不变；换队首才换期。
//   * 于是回填不是插队。资源侧必须先证明"放这个候选进来不会让队首等更久"，Scheduler 再核对这份
//     proof 的物理版本与保护期冻结时是否一致，一致才给候选发证。同一保护期内进来的回填者不会被
//     补进 donor 名单，所以反复回填也不会挪动"队首在等谁"这个结论。
//
// 契约纪律（读 admission 那一半最有用的读法）：所有"这件事本该成立"的前提都用 logic_error 表达，
// 而不是容错。因为队首是 EngineCore 通过 observe_fifo_head **告知**的、不是 Scheduler 自己去看
// 队列的，所以"你说的队首"与"我记的队首"不一致只可能是调用方用错了，不可能是竞态。相对地，**正常
// 的"这次不行"用 nullopt / false 表达**（例如回填资格不足），两类刻意分开、不许混。
//
// 它没有自己的同步原语：所有方法都只在 worker 线程内、在 EngineCore 的执行边界上被调用；队列由
// EngineCore 带着 queue_mutex_ 读完，以快照的形式递进来（见 FifoSnapshot）。

namespace ninfer::runtime {

template <class Request>
class Scheduler {
public:
    using RequestPtr     = std::shared_ptr<Request>;
    // 模型侧的序列句柄。Scheduler 只把它当不透明身份搬运（成员表按 lane 与它对齐），不解释内容。
    using SequenceHandle = typename Request::SequenceHandle;

    // 一个执行单位跑什么。Prefill 与 control 都是单请求形态，decode 是紧凑批量；Wait 表示这一轮
    // 什么都不跑（worker 会短暂让出），它是正常状态而不是错误。
    enum class ExecutionAction : std::uint8_t {
        Prefill,
        Decode,
        Wait,
    };

    // 一张**一次性的准入通行证**：由 grant_head / qualify_backfill 签发，EngineCore 拿着它让资源
    // 侧真正预留，最后在 commit_admission 里兑现。兑现即作废，所以它只能移动、不能复制。
    //
    // 它携带的是"签发当时的事实"而不是"现在的事实"：保护期编号、候选类别、以及签发所依据的物理
    // 版本。中间任何一步让这些事实变了（换了队首、epoch 前进、revision 推进），这张证就不再成立
    // ——validate_grant 判 false，EngineCore 必须丢掉它重新走一遍准入。
    class AdmissionGrant {
    public:
        AdmissionGrant(AdmissionGrant&&) noexcept            = default;
        AdmissionGrant& operator=(AdmissionGrant&&) noexcept = default;
        AdmissionGrant(const AdmissionGrant&)                = delete;
        AdmissionGrant& operator=(const AdmissionGrant&)     = delete;

        // 给谁：这条请求的身份。
        [[nodiscard]] std::uint64_t request_id() const noexcept { return request_id_; }

        // 它是怎么进来的：None = 队首本尊（按 FIFO 顺序正常准入），Persistent = 保护期内的回填者。
        // 这个类别决定下面两项有没有意义——队首本尊的两项恒为 0。
        [[nodiscard]] BackfillClass backfill_class() const noexcept { return backfill_class_; }

        // 它是在哪一期保护下进来的（见 protect_blocked_head）。请求会把它抄进自己的 backfill_epoch
        // 并从此不再改，供后来的 donor 判定使用。
        [[nodiscard]] std::uint64_t protection_epoch() const noexcept { return protection_epoch_; }

        // 签发时 Program 的物理版本。ProgramResourceRevision 只在**物理事实变化**时推进（活跃请求
        // 在自己预留额度之内推进 frontier 不算），所以"revision 没变"等价于"物理前提没变"——这正是
        // 这张证可以被重验的原因。
        [[nodiscard]] ProgramResourceRevision resource_revision() const noexcept {
            return resource_revision_;
        }

        // 这次准入授予多少**调度服务份额**（service work）：EngineCore 会把它记进请求的
        // remaining_service_work，之后按实际执行扣减。口径是"调度步"而不是算力——prefill 的一个
        // 分块算一份、每采纳一个输出 token 算一份；资源侧的 PrefillWork 是另一套记账（按 token 与
        // attention 对数量），两者别混。
        [[nodiscard]] std::uint64_t service_work_quanta() const noexcept {
            return service_work_quanta_;
        }

    private:
        AdmissionGrant(std::uint64_t request_id, BackfillClass backfill_class,
                       std::uint64_t protection_epoch, ProgramResourceRevision resource_revision,
                       std::uint64_t service_work_quanta) noexcept
            : request_id_(request_id), backfill_class_(backfill_class),
              protection_epoch_(protection_epoch), resource_revision_(resource_revision),
              service_work_quanta_(service_work_quanta) {}

        std::uint64_t request_id_       = 0;
        BackfillClass backfill_class_   = BackfillClass::None;
        std::uint64_t protection_epoch_ = 0;
        ProgramResourceRevision resource_revision_;
        std::uint64_t service_work_quanta_ = 0;

        friend class Scheduler;
    };

    // 队列的**时点快照**。真正的 pending_ 由 EngineCore 的另一把锁保护，Scheduler 不能在锁外直接
    // 看它，于是准入流程在同一次加锁里拷出一份。本次决策只针对快照里的这批人——即使期间调用方又
    // 提交了新请求，那些人也只能等下一轮。
    class FifoSnapshot {
    public:
        [[nodiscard]] bool empty() const noexcept { return requests_.empty(); }

        // 队首。空快照没有队首——那是调用方的契约错误（应当先问 empty()）。
        [[nodiscard]] const RequestPtr& head() const {
            if (requests_.empty()) { throw std::logic_error("FIFO snapshot has no head"); }
            return requests_.front();
        }

        // 回填候选：**除队首之外**的其余排队者，保持 FIFO 顺序。这个定义本身就是"回填不多于排队"
        // 的前提——候选集合里根本没有队首，也不存在任何"跳到前面"的入口。
        [[nodiscard]] std::span<const RequestPtr> backfill_candidates() const noexcept {
            if (requests_.size() <= 1) { return {}; }
            return {requests_.data() + 1, requests_.size() - 1};
        }

    private:
        explicit FifoSnapshot(const std::deque<RequestPtr>& pending)
            : requests_(pending.begin(), pending.end()) {}

        std::vector<RequestPtr> requests_;

        friend class Scheduler;
    };

    // 一轮 decode 的成员表：哪些 lane 一起跑，各自用哪个 sequence、还剩多少**模型侧**额度。
    // Scheduler 在这里做的是筛选与不变量校验，不决定要不要跑 decode（那是 choose_execution 的事）。
    struct RoundMembership {
        // 三张表按行对齐（lanes[i] 配 sequences[i]、取 budgets[i]），这是紧凑批量的形状要求。
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<SequenceHandle, kMaximumConcurrency> sequences{};
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        std::size_t size = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return {sequences.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    // control 批量的成员表：把若干请求各自要注入的**规范 token** 拼成一块紧凑 target span。它与
    // decode 分成两种执行单位，因为那些 token 是 Engine 定为既成事实喂进去的，不经过采样。
    struct ControlMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<SequenceHandle, kMaximumConcurrency> sequences{};
        // 各行 token 按 lanes 顺序首尾相接，row_stride 是每行的固定宽度（见 build_control_membership）。
        std::vector<TokenId> tokens;
        std::uint32_t row_stride = 0;
        std::size_t size         = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return {sequences.data(), size};
        }
    };

    // 当前 active 请求的**身份投影**：只带 id / epoch / class 三项，让保护逻辑判断"谁是 donor、
    // 谁是被本保护期放进来的人"。注意它是全量 active（只跳过空位）：正在做 capture 的请求也在里面，
    // 因为它仍然占着 lane 与物理资源，仍然是可能的 donor。
    struct ActiveAdmissionSet {
        std::array<ActiveAdmissionSnapshot, kMaximumConcurrency> requests{};
        std::size_t size = 0;

        [[nodiscard]] std::span<const ActiveAdmissionSnapshot> span() const noexcept {
            return {requests.data(), size};
        }
    };

    // 队列快照的唯一构造入口（EngineCore 持锁调用）。
    [[nodiscard]] static FifoSnapshot fifo_snapshot(const std::deque<RequestPtr>& pending) {
        return FifoSnapshot(pending);
    }

    // ---- 排班：这一轮跑什么 ----

    // 编一轮 decode。进表的条件都是不变量级的：必须已经 DecodeReady、没有在做 capture；
    // capture_pending 的请求被显式跳过——它的状态正被发布成 checkpoint，本轮不能参与。
    //
    // budgets 里装的是**模型自发 token** 的剩余额度（思考预算会封顶，待注入控制 token 时归零），
    // 它既不是服务份额也不是总预算。额度为 0 却还留在 decode 表里的请求，此刻本该是 ControlReady，
    // 所以那三条 throw 讲的是同一件事：不变量破了，而不是"跳过它就没事"。
    template <class Slots>
    [[nodiscard]] RoundMembership build_round_membership(const Slots& slots,
                                                         std::uint32_t max_concurrency) const {
        RoundMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr || !request->is_decode_ready() || request->capture_pending) {
                continue;
            }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            if (!request->sequence) {
                throw std::logic_error("decode-ready request has no sequence handle");
            }
            membership.lanes[membership.size]     = lane;
            membership.sequences[membership.size] = *request->sequence;
            membership.budgets[membership.size]   = RoundBudget{
                  .generated_tokens_remaining =
                    request->output.model_token_budget_remaining(request->budget->remaining()),
            };
            if (membership.budgets[membership.size].generated_tokens_remaining == 0) {
                throw std::logic_error("decode-ready request has no licensed model tokens");
            }
            ++membership.size;
        }
        return membership;
    }

    // 编一个 control 批量。行宽（row_stride）由第一个成员定下，之后每个成员必须一致：整批要压成
    // 一块连续的 target span，宽度不等就没法成型，所以那是 bug 而不是"跳过这一行"。同理，要注入的
    // control token 多于剩余预算也是 bug——Engine 自己定的注入量不该超过自己给的预算。
    template <class Slots>
    [[nodiscard]] ControlMembership build_control_membership(const Slots& slots,
                                                             std::uint32_t max_concurrency) const {
        ControlMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr || !request->is_control_ready() || request->capture_pending) {
                continue;
            }
            if (!request->budget || !request->sequence) {
                throw std::logic_error("control-ready request has no generation state");
            }
            const std::span<const TokenId> control = request->output.pending_control_tokens();
            if (control.empty() || control.size() > request->budget->remaining()) {
                throw std::logic_error("control-ready request has no admissible control span");
            }
            if (membership.row_stride == 0) {
                membership.row_stride = static_cast<std::uint32_t>(control.size());
                membership.tokens.reserve(static_cast<std::size_t>(membership.row_stride) *
                                          max_concurrency);
            } else if (control.size() != membership.row_stride) {
                throw std::logic_error("compact control membership has ragged target spans");
            }
            membership.lanes[membership.size]     = lane;
            membership.sequences[membership.size] = *request->sequence;
            membership.tokens.insert(membership.tokens.end(), control.begin(), control.end());
            ++membership.size;
        }
        return membership;
    }

    // 把 active lane 投影成身份三元组，供准入保护使用。服务份额为 0 的 active 请求是 bug：准入授予
    // 的份额按实际执行扣减，还没走完的请求必然还剩着。
    template <class Slots>
    [[nodiscard]] ActiveAdmissionSet active_admission_set(const Slots& slots,
                                                          std::uint32_t max_concurrency) const {
        ActiveAdmissionSet active;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr) { continue; }
            if (request->remaining_service_work == 0) {
                throw std::logic_error("active request has no admission accounting");
            }
            active.requests[active.size++] = ActiveAdmissionSnapshot{
                .request_id     = request->id,
                .backfill_epoch = request->backfill_epoch,
                .backfill_class = request->backfill_class,
            };
        }
        return active;
    }

    // 扣减一次准入授予的服务份额。粒度按单位而不是按时间：prefill 一步一份、control 每 token 一份、
    // decode 每**采纳**一个 token 一份（投机一轮采纳了多个就扣多份，所以这里的入参可以是 accepted
    // 数）。超支意味着计划给的份额不足以兑现实际执行，那是不变量破了，不是"少扣一点"。
    static void consume_service_work(Request& request, std::uint64_t work) {
        if (work == 0 || work > request.remaining_service_work) {
            throw std::logic_error("request service projection consumed " + std::to_string(work) +
                                   " quanta with " +
                                   std::to_string(request.remaining_service_work) + " remaining");
        }
        request.remaining_service_work -= work;
    }

    // 这一轮要不要试着准入。**准入不是每轮都做的**，它被四道门控拦着：有排队者、有"准入可见变化"
    // 待处理、当前没有资源事务在跑、没有请求占着 prefill 归属；最后一条还要在"有 decode 可跑且上
    // 一单位不是 decode"时让路——让已经开头的那串 decode 先跑顺，准入等信息更清楚的下一轮再去碰。
    // 准入要付出资源规划的代价，这些条件合起来表达的就是"别把它塞进热路径"。
    [[nodiscard]] bool should_attempt_admission(bool have_pending, bool admission_check_pending,
                                                bool have_decode, bool previous_unit_was_decode,
                                                bool context_transaction) const noexcept {
        return have_pending && admission_check_pending && !context_transaction && !prefill_lane_ &&
               (!have_decode || previous_unit_was_decode);
    }

    // 这一轮跑什么，三选一。prefill 的优先级高于 decode，但**不是抢占**：只有当上一单位不是
    // decode 时 prefill 才插得进来，也就是说规则刻意让同形态的执行单位连成一"波"，抑制 prefill
    // 与 decode 逐轮交错（两者在 Program 侧的 kernel 形态与 capture 方式都不同）。
    // 没有 prefill 可跑就跑 decode，两者都没有才 Wait。
    [[nodiscard]] ExecutionAction choose_execution(bool have_decode, bool prefill_runnable,
                                                   bool previous_unit_was_decode) const noexcept {
        if (prefill_runnable) {
            return have_decode && !previous_unit_was_decode ? ExecutionAction::Decode
                                                            : ExecutionAction::Prefill;
        }
        return have_decode ? ExecutionAction::Decode : ExecutionAction::Wait;
    }

    // prefill 的**归属记录**：谁拥有当前这段 prefill。它必须存在，是因为 prefill 是跨执行单位、
    // 分块推进的（中途还可能插入 capture），Program 不接受"没有归属的 prefill"。同一时刻至多一个
    // 持有者，set / clear 都做配对检查——重复 set 或清掉别人的 lane 都是所有权乱了。
    [[nodiscard]] std::optional<std::uint32_t> prefill_lane() const noexcept {
        return prefill_lane_;
    }

    // 当前保护期的编号；没有保护期时为 nullopt。请求把发证时的编号抄进自己的 backfill_epoch，于是
    // "本期回填者"这个身份此后可以脱离保护期本身独立判断。
    [[nodiscard]] std::optional<std::uint64_t> protection_epoch() const noexcept {
        return protection_ ? std::optional<std::uint64_t>(protection_->epoch_id) : std::nullopt;
    }

    void set_prefill_lane(std::uint32_t lane) {
        if (prefill_lane_) { throw std::logic_error("multiple requests own staged prefill"); }
        prefill_lane_ = lane;
    }

    void clear_prefill_lane(std::uint32_t lane) {
        if (!prefill_lane_ || *prefill_lane_ != lane) {
            throw std::logic_error("request does not own staged prefill");
        }
        prefill_lane_.reset();
    }

    // 告知"这一轮的队首是谁"（nullopt = 队列空了）。这是保护期的唯一驱动：队首一变，上一期保护连同
    // 它的 donor 名单立刻作废——那份名单描述的是"谁挡着上一个队首"，对新队首没有意义。
    void observe_fifo_head(std::optional<std::uint64_t> request_id) noexcept {
        if (fifo_head_id_ == request_id) { return; }
        fifo_head_id_ = request_id;
        protection_.reset();
    }

    // 队列里有人被摘掉（取消、超时、了结）时通知。只有摘掉的正好是队首才清：队首已经不在队列里了，
    // 再挂着它的保护期就是记着一件不存在的事。
    void on_waiting_removed(std::uint64_t request_id) noexcept {
        if (fifo_head_id_ && *fifo_head_id_ == request_id) {
            fifo_head_id_.reset();
            protection_.reset();
        }
    }

    // 给队首发证。注意它属于 None 类、epoch 与 revision 都是 0：队首本尊的资格**绑在"我观测到的
    // 队首"上**，不绑保护期——它按 FIFO 顺序本来就该进来，不需要任何额外的物理承诺。
    [[nodiscard]] AdmissionGrant grant_head(std::uint64_t request_id,
                                            std::uint64_t service_work_quanta) const {
        if (!fifo_head_id_ || *fifo_head_id_ != request_id || request_id == 0 ||
            service_work_quanta == 0) {
            throw std::logic_error("head admission is not bound to the observed FIFO head");
        }
        return AdmissionGrant(request_id, BackfillClass::None, 0, {}, service_work_quanta);
    }

    // 队首被挡住：给它开一个保护期，或刷新已经开着的那个。第一次调用把当前 active 全体冻成 donor、
    // 记下当时的物理版本；之后只做 rebind——重新校验 active 的分区与身份、更新物理版本，但 epoch
    // 与 donor 名单冻结不动。**同一期内进来的回填者绝不会被补进 donor 名单**，否则反复回填就能不断
    // 改写"队首在等谁"这个结论，队首可能永远等不到。
    //
    // 返回的是"这份保护还有没有活着的 donor"。没有 donor 就没人能救队首，EngineCore 会据此直接放弃
    // 本轮——那是正常结果，不是错误。
    [[nodiscard]] bool protect_blocked_head(std::uint64_t request_id,
                                            std::span<const ActiveAdmissionSnapshot> active,
                                            ProgramResourceRevision resource_revision) {
        if (!fifo_head_id_ || *fifo_head_id_ != request_id) {
            throw std::logic_error("blocked admission does not match the observed FIFO head");
        }
        if (!protection_) {
            protection_.emplace(make_admission_protection(next_protection_epoch_++, request_id,
                                                          resource_revision, active));
        } else if (protection_->head_request_id != request_id) {
            throw std::logic_error("protected head changed without a FIFO transition");
        } else {
            rebind_admission_protection(*protection_, active, resource_revision);
        }
        return protection_has_live_donor(*protection_, active);
    }

    // 给队列内部的候选发回填证。三个前提来路不同：保护期必须开着且对应当前队首（Scheduler 自己记
    // 的事实）、候选身份必须合法（不能是队首、不能是 0）、以及 Program 的 proof 必须与保护期冻结的
    // 物理版本一致（证明所依据的世界没变过）。
    //
    // 资格不足返回 nullopt 而不是抛异常——这是**正常的"这次不行"**，与上面那些 logic_error（调用方
    // 用错了）刻意分开。候选可以继续留在队列里等下一轮。
    [[nodiscard]] std::optional<AdmissionGrant>
    qualify_backfill(std::uint64_t request_id, std::uint64_t service_work_quanta,
                     std::span<const ActiveAdmissionSnapshot> active,
                     ProgramResourceRevision program_proof_revision) const {
        if (!fifo_head_id_ || !protection_ || protection_->head_request_id != *fifo_head_id_) {
            throw std::logic_error("backfill qualification has no open protected head");
        }
        if (request_id == 0 || request_id == *fifo_head_id_ || service_work_quanta == 0) {
            throw std::logic_error("backfill candidate has invalid scheduling identity");
        }
        if (persistent_backfill_is_authorized(*protection_, request_id, active,
                                              program_proof_revision)) {
            return AdmissionGrant(request_id, BackfillClass::Persistent, protection_->epoch_id,
                                  program_proof_revision, service_work_quanta);
        }
        return std::nullopt;
    }

    // 重验一张证是否还成立。两类证有两套条件：队首本尊要求 epoch 与 revision 都是 0（它不依赖保护
    // 期），且**当前观测到的队首仍然是它**；回填者要求保护期还开着、队首没换、epoch 与 revision 都
    // 仍等于签发时的值，并且自己不是队首。任一条不成立，就说明签发与兑现之间世界变了，EngineCore
    // 必须把这张证丢掉重来。
    [[nodiscard]] bool validate_grant(const AdmissionGrant& grant) const noexcept {
        if (grant.request_id_ == 0 || grant.service_work_quanta_ == 0) { return false; }
        if (grant.backfill_class_ == BackfillClass::None) {
            return grant.protection_epoch_ == 0 && grant.resource_revision_.value == 0 &&
                   fifo_head_id_ && *fifo_head_id_ == grant.request_id_;
        }
        if (!fifo_head_id_ || !protection_ || protection_->head_request_id != *fifo_head_id_ ||
            protection_->epoch_id != grant.protection_epoch_ ||
            protection_->resource_revision != grant.resource_revision_ ||
            grant.request_id_ == *fifo_head_id_) {
            return false;
        }
        return grant.backfill_class_ == BackfillClass::Persistent;
    }

    // 兑现，也就是销证。**队首本尊进场意味着队首已经离开队列，本期保护随之结束**——fifo_head_id_
    // 与 protection_ 一起清掉，下一轮 observe_fifo_head 会看到新队首、重开一期。回填者进场则不动
    // 保护期：队首还在等，它的 donor 名单也仍然成立。
    void commit_admission(AdmissionGrant&& grant) {
        if (!validate_grant(grant)) {
            throw std::logic_error("admission grant is stale or inconsistent");
        }
        if (grant.backfill_class_ == BackfillClass::None) {
            fifo_head_id_.reset();
            protection_.reset();
            grant.request_id_          = 0;
            grant.service_work_quanta_ = 0;
            return;
        }
        grant.request_id_          = 0;
        grant.service_work_quanta_ = 0;
    }

    // 停机 / 引擎级失败时清空：这些记录描述的都是"当前关系"，引擎都不跑了就无所谓关系。
    void reset() noexcept {
        prefill_lane_.reset();
        fifo_head_id_.reset();
        protection_.reset();
    }

private:
    // prefill 归属者：跨执行单位的那段 prefill 属于哪个 lane。
    std::optional<std::uint32_t> prefill_lane_;
    // 最后一次被告知的队首。它是**观测记忆**而不是真相——真相在 EngineCore 的队列里，不一致就意味着
    // 调用方用错了（所以各发证接口都拿它做契约检查）。
    std::optional<std::uint64_t> fifo_head_id_;
    // 队首的保护期（冻结的 donor 名单 + 物理版本）。只在队首被挡住、且还有 donor 活着时开张。
    std::optional<AdmissionProtection> protection_;
    // 保护期编号只增不减，且从 1 起：0 表示"没有保护期"，被当作无效值使用。
    std::uint64_t next_protection_epoch_ = 1;
};

} // namespace ninfer::runtime
