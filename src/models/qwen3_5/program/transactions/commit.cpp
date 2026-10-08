#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"

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
#include <variant>

namespace ninfer::models::qwen3_5::detail {

// ============================================================================
// 结算面 —— 进程对外交付与生命周期收场的地方
//
// 相邻的三类工作，性质各不相同：
//   1) 结果包装（wrap_pending / wrap_prefill）：把执行层的裸结果翻译成外部能拿走的句柄与票据。凡是对外
//      发出的句柄都带 lane 代次，因此它们只在"发出的那一刻"有效——lane 一作废，所有旧句柄同时失效。
//   2) 一轮生成的结算（commit / abort_pending）：外部对"这一行接受哪些 token"作出裁决之后，由这里把裁决
//      落进账本，并把该释放的行释放掉。
//   3) 生命周期的收场（start_request / finish / abort / release_*）：落位 → 结束（释放或目录化）→ 被回收。
//      每一种收场各走一条路，互不共用。
//
// 两条贯穿全文件的性质：
//   · 收场函数一律 noexcept，失败用"空结果 / 未消费"表示而不抛异常——走到这一步已经没有上层能接住异常，
//     只剩"做成了"和"没做成"两种事实。
//   · 任何改变物理占用或目录可见性的收场都要推进 resource_revision（物理世界版本号）：不推进，之前算出的
//     压力方案与封印计划就会按旧世界继续生效。
// ============================================================================
// 把一轮生成的结果包成对外的 PendingBatch：登记这次结算涉及哪些 lane（连各自的代次一起记下），并给出
// 事务号。Program 同一时刻只允许一个 pending 事务（pending_transaction_），所以这里既在包装也在上锁——
// 在 commit 或 abort_pending 把它消化掉之前，新的解码与追加都不允许开始。
PendingBatch ProgramImpl::wrap_pending(std::span<const std::uint32_t> lanes,
                                       const runtime::BatchedGeneratedRound& round) {
    if (pending_transaction_ || lanes.empty() || lanes.size() > max_concurrency) {
        throw std::logic_error("Program already owns a pending transaction");
    }
    PendingTransaction transaction;
    transaction.id   = next_transaction_id_++;
    transaction.size = lanes.size();
    std::array<SequenceHandle, kMaximumConcurrency> handles{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending) {
            throw std::logic_error("pending transaction membership is invalid");
        }
        transaction.lanes[row]  = lane;
        transaction.epochs[row] = lane_epochs[lane];
        handles[row] =
            ContractAccess::make_sequence(this, runtime::LaneId{lane}, lane_epochs[lane]);
    }
    pending_transaction_ = transaction;
    return ContractAccess::make_pending(
        this, transaction.id, std::span<const SequenceHandle>(handles.data(), lanes.size()),
        round.tokens, round.row_counts, round.row_stride, round.timing);
}

// 预填一步的结果有两个出口，取决于它有没有走完：走完就顺手把这一轮的 token 包成 pending 事务（预填
// 结束那一枚 token 同样要走一次结算）；没走完就把当前挂着的那张捕获票据递出去。
PrefillProgress ProgramImpl::wrap_prefill(std::uint32_t lane, runtime::PrefillStepResult step) {
    PrefillProgress out;
    out.summary                 = step.summary;
    out.processed_prompt_tokens = step.processed_prompt_tokens;
    out.complete                = step.complete;
    out.timing                  = step.timing;
    if (step.complete) {
        const std::array<std::uint32_t, 1> lanes{lane};
        const runtime::BatchedGeneratedRound round{
            .tokens     = step.round.tokens,
            .row_counts = {},
            .row_stride = 1,
        };
        out.pending.emplace(wrap_pending(lanes, round));
    } else if (requests[lane].prefill && requests[lane].prefill->pending_capture_offer != 0) {
        out.capture.emplace(
            ContractAccess::make_capture_offer(this, runtime::LaneId{lane}, lane_epochs[lane],
                                               requests[lane].prefill->pending_capture_offer));
    }
    return out;
}

// 物化事务的**物理发布点**：把此前全是"预订态"的一切转正成一条真正可执行的活跃序列。之所以强调这一点，
// 是因为在这之前的所有阶段都还能整体撤销，而从这里开始它们就落成了 lane 上的事实。正因如此，它自己也
// 必须尽量原子：任何一步失败，catch 里把 lane 与槽位退回到干净形态再抛出。
//
// 落脚点有两种来路：ConsumeToActive 直接接管来源槽位（来源就此消失，新请求继承它的一切）；其余情况用
// 容量预留阶段就占好的那个 root 槽位。两条路的共同要求是"槽位身份未变"——索引 + 代次都要对得上。
StartResult ProgramImpl::start_request(MaterializationTransaction& transaction) {
    std::optional<std::uint32_t> destination = transaction.destination.value;
    std::optional<std::uint32_t> continuation_index;
    try {
        if (!transaction.prepared || !transaction.plan || !destination ||
            *destination >= max_concurrency) {
            throw std::invalid_argument("materialization transaction is not publishable");
        }
        const std::uint32_t lane              = *destination;
        const AdmissionCandidateImpl& details = *transaction.plan->impl_;
        if (details.destination_epoch != lane_epochs[lane] ||
            details.has_source != transaction.has_source ||
            details.has_shared_source != transaction.has_shared_source) {
            throw std::logic_error("admission plan physical epoch is stale");
        }
        if (requests[lane].lifecycle != Lifecycle::Empty ||
            active_continuations[lane] < continuation_capacity) {
            throw std::logic_error("admission destination is not free");
        }
        // 来源被"吃掉"：来源槽位原样转给本 lane，因此这里要复核它确实还在目录里、身份未变，并与计划
        // 描述的来源完全一致。
        if (transaction.has_source &&
            transaction.source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
            if (transaction.source_index >= continuation_capacity ||
                continuation_slots[transaction.source_index].role !=
                    ContinuationSlotRole::Catalogued ||
                continuation_slots[transaction.source_index].generation !=
                    transaction.source_generation ||
                transaction.source_index != details.source_index ||
                transaction.source_generation != details.source_generation) {
                throw std::logic_error("admission source capability is stale");
            }
            continuation_index                           = transaction.source_index;
            continuation_slots[*continuation_index].role = ContinuationSlotRole::Active;
        } else {
            // 其余情况用预留好的 root 槽位。它此刻必须还是"事务已认领但未落定"的状态；若还在等牺牲者
            // 让位，说明压力阶段没真正走完。
            continuation_index = transaction.root_continuation_index;
            if (!continuation_index || transaction.root_waiting_for_victim ||
                continuation_slots[*continuation_index].role !=
                    ContinuationSlotRole::ReservedMaterialization) {
                throw std::logic_error("materialization continuation reservation is unavailable");
            }
            continuation_slots[*continuation_index].role = ContinuationSlotRole::Active;
        }

        // 槽位 → lane：从这一刻起这条序列有了归属。root_continuation_index 随即交还给事务，表示"这个
        // 槽位已经不在事务手上，而在 lane 上"。
        const detail::PhysicalResources active = details.demand.active_entitlement;
        active_continuations[lane]             = *continuation_index;
        SequenceState& sequence                = continuation_states[*continuation_index];
        sequence.lane                          = lane;
        transaction.root_continuation_index.reset();
        start_sequence(lane, sequence, transaction);
        // 落位后立刻按 owner 独占资源对账：序列实际持有的必须与计划核定的活跃额度分毫不差。这里是两边
        // 账本合流的关口——对不上说明前面的准备漏了或多了东西。
        detail::PhysicalResources actual         = owner_exclusive_resources(sequence);
        actual.device.active_lanes               = 1;
        const detail::PhysicalResources expected = active;
        if (actual != expected) {
            throw std::logic_error("materialized sequence does not match its active entitlement");
        }
        // 统计口径：这次复用到底是"从 Host 搬回来"、"fork 出一份新的"，还是"整份挪过来"。三者成本差
        // 得远，所以分别计数；Root（全新序列）不算复用，不计。
        if (details.reuse != ReusePath::Root) {
            if (transaction.state_restored) {
                ++transaction.operations.state_restores;
            } else if (details.source_mode == runtime::PrivateSourceMode::Retain ||
                       transaction.has_shared_source || details.state_fork_required) {
                ++transaction.operations.state_forks;
                ++transaction.operations.historical_fork_hits;
            } else {
                ++transaction.operations.state_moves;
            }
        }
        requests[lane].active_resources   = active;
        requests[lane].optional_resources = details.active_optional_resources;
        // 发布 = 换人：推进 lane 代次，于是之前发给外部的一切（句柄、票据）同时失效，新的句柄从新代次
        // 开始。这一步必须在返回句柄之前。
        invalidate_lane(lane);
        const SequenceHandle handle =
            ContractAccess::make_sequence(this, runtime::LaneId{lane}, lane_epochs[lane]);
        return StartResult{.sequence = handle};
    } catch (...) {
        // 失败时的退场分两档：已经挂到 lane 上的用 best-effort 清（只求别崩、别漏），还停在槽位上的退回
        // 槽位。两者都要推进 lane 代次，避免外部拿着一个"看起来还能用"的句柄。
        if (destination && *destination < max_concurrency) {
            const std::uint32_t lane = *destination;
            if (active_continuations[lane] < continuation_capacity) {
                clear_lane_best_effort(active_sequence(lane), requests[lane]);
            } else if (continuation_index) {
                release_continuation_slot_best_effort(*continuation_index);
            }
            invalidate_lane(*destination);
        }
        throw;
    }
}

// 解码一轮：把对外的句柄翻回内部的 lane，跑一轮前向，再把结果包成待结算票据。执行本体在 decode_raw，
// 这一层管的是成员验收、结果包装，以及失败时把整批行从执行失败态清掉。
PendingBatch ProgramImpl::decode(std::span<const SequenceHandle> members,
                                 std::span<const runtime::RoundBudget> budgets,
                                 runtime::ExecutionTiming* failed_timing) {
    // 同一时刻只允许存在一张待结算票据；预算逐行给出，所以条数必须与成员数对齐。
    if (pending_transaction_ || members.empty() || members.size() > max_concurrency ||
        budgets.size() != members.size()) {
        throw std::invalid_argument("decode membership is invalid");
    }
    // 成员验收：句柄身份要有效（owner 与 lane 代次都对得上），lane 必须处于 Active，且同一批里不允许
    // 出现重复 lane —— 一行只跑一次。
    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    for (std::size_t row = 0; row < members.size(); ++row) {
        if (!valid_sequence(members[row])) {
            throw std::logic_error("decode sequence capability is invalid");
        }
        const std::uint32_t lane = ContractAccess::lane(members[row]).value;
        if (requests[lane].lifecycle != Lifecycle::Active ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("decode membership is duplicate or not active");
        }
        lanes[row] = lane;
    }
    const auto lane_span = std::span<const std::uint32_t>(lanes.data(), members.size());
    try {
        runtime::BatchedGeneratedRound round = decode_raw(lane_span, budgets, failed_timing);
        if (failed_timing != nullptr) { *failed_timing += round.timing; }
        return wrap_pending(lane_span, std::move(round));
    } catch (...) {
        // 执行本体已经出事，票据不能留在外面：先把这批行从执行失败态清掉（登记各自的失败），再清票据。
        // 清场所花的时间计入 post_host_ns —— 它是这次失败实打实的代价。
        const Clock::time_point cleanup_started = Clock::now();
        clear_execution_failure_lanes(lane_span);
        pending_transaction_.reset();
        if (failed_timing != nullptr) {
            failed_timing->post_host_ns += elapsed_ns(cleanup_started);
        }
        throw;
    }
}

// 把设备产出的 token 整理成"前缀身份"：账本、前缀身份、前缀摘要三者的长度与内容在这里对齐到接受范
// 围。采样点（decode / prefill）只写下刚采到的那一枚 token 的临时身份，是这里按裁决补全、并在需要时
// 把它们推倒重来——之后这些身份才会被复用判定读到（前缀匹配靠的就是身份与摘要）。
//
// 进入时有两种形态，由调用方决定，函数自己负责确认形态合法：
//   · Begin 与普通轮：接受范围之前的部分可能已经有临时身份，走到这里补齐到接受范围；
//   · 投机段与强制段：带着自己基点处的身份到来，由这里补齐这一段。
// 两者都是 Program 拥有的待定状态，这里是它们共同的接受前缀身份提交点。
void ProgramImpl::commit_generated_prefix_identity(
    SequenceState& sequence, std::uint32_t base_ledger_frontier,
    std::span<const TokenId> accepted_tokens,
    std::optional<std::uint32_t> prefix_execution_split_after) {
    if (base_ledger_frontier > sequence.ledger.size() ||
        accepted_tokens.size() > sequence.ledger.size() - base_ledger_frontier ||
        sequence.ledger.size() != base_ledger_frontier + accepted_tokens.size() ||
        !std::equal(accepted_tokens.begin(), accepted_tokens.end(),
                    sequence.ledger.begin() + static_cast<std::ptrdiff_t>(base_ledger_frontier)) ||
        (prefix_execution_split_after &&
         (*prefix_execution_split_after == 0 ||
          *prefix_execution_split_after > accepted_tokens.size()))) {
        throw std::logic_error("committed generated-prefix identity has an invalid span");
    }
    // 只有两种合法起点：已经补到账本末尾（重复调用），或者停在基点（待补）。长度落在别处说明前面的
    // 流程已经错位，宁可拒绝也不能在错误的长度上做截断追加。
    const bool already_appended = sequence.prefix_identity.size() == sequence.ledger.size() &&
                                  sequence.prefix_digests.size() == sequence.ledger.size();
    const bool awaits_append = sequence.prefix_identity.size() == base_ledger_frontier &&
                               sequence.prefix_digests.size() == base_ledger_frontier;
    if (!already_appended && !awaits_append) {
        throw std::logic_error("generated-prefix identity is not at its base or committed extent");
    }
    // 已经补过且这次没有执行切分要记，才可以空手返回：切分边界是一份必须留下的记录，即使 token 早已
    // 补齐，它也不能被这次"重复调用"顺手丢掉。
    if (already_appended && !prefix_execution_split_after) { return; }
    // 截断到基点再统一追加，重复调用因此得到同一结果（幂等靠的是"先退回到已知状态"）。
    sequence.prefix_identity.truncate(base_ledger_frontier);
    sequence.prefix_digests.truncate(base_ledger_frontier);
    sequence.prefix_identity.append_generated(accepted_tokens.size(), sequence.rope_delta,
                                              prefix_execution_split_after);
    sequence.prefix_digests.append_generated(accepted_tokens, sequence.rope_delta,
                                             prefix_execution_split_after);
    // 收尾复核三者等长：身份与摘要都只允许与账本一样长。多一少一都会让后续的复用判定读到错位的内容。
    if (sequence.prefix_identity.size() != sequence.ledger.size() ||
        sequence.prefix_digests.size() != sequence.ledger.size()) {
        throw std::logic_error("committed generated-prefix identity changed the ledger shape");
    }
}

// 强制 token 的灌入通道：这些 token 不由采样产生，而是调用方直接指定（比如服务侧要求"必须续上这几
// 个词"）。与解码不同，这里没有待结算票据——token 是外部给的，不存在"接受哪些"的裁决。
//
// 整批先整体验收再整体动手：任意一行不合格就整批拒绝，避免出现"半批已生效"的中间态。验收通过后逐行
// 推进——按 chunk 走预填、记录前缀身份、推进重建工作、把 KV 修剪到新前沿。
runtime::ExecutionTiming ProgramImpl::append_forced_tokens(
    std::span<const SequenceHandle> members, std::span<const TokenId> row_major_tokens,
    std::uint32_t row_stride, std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    // token 以行距排布（row_stride 是行距不是行长），因此缓冲区大小必须是行距 × 行数。
    if (pending_transaction_ || members.empty() || members.size() > max_concurrency ||
        row_stride == 0 || prefix_execution_splits.size() != members.size() ||
        row_major_tokens.size() != static_cast<std::size_t>(row_stride) * members.size()) {
        throw std::invalid_argument("forced-token membership is invalid");
    }

    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    for (std::size_t row = 0; row < members.size(); ++row) {
        if (!valid_sequence(members[row])) {
            throw std::logic_error("forced-token sequence capability is invalid");
        }
        const std::uint32_t lane = ContractAccess::lane(members[row]).value;
        if (requests[lane].lifecycle != Lifecycle::Active ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("forced-token membership is duplicate or not active");
        }
        // 前沿不变式：账本永远比执行前沿多一格——那一格是"下一个 token 的位置"，也是这条序列还能接
        // 受新 token 的前提。身份、摘要、KV 有效长度都必须与前沿一致；MTP 后端还要 draft KV 对齐，
        // DFlash 后端的 context 不允许跑到执行前沿之前。
        const SequenceState& sequence = active_sequence(lane);
        if (sequence.execution_frontier == std::numeric_limits<std::uint32_t>::max() ||
            sequence.ledger_frontier != sequence.execution_frontier + 1U ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != sequence.execution_frontier) ||
            (is_masked_draft_backend(speculative_backend) &&
             sequence.dflash_context_frontier > sequence.execution_frontier) ||
            static_cast<std::uint64_t>(sequence.execution_frontier) + row_stride > capacity) {
            throw std::logic_error("forced-token sequence frontier is invalid");
        }
        validate_licensed_tokens(row_major_tokens.subspan(row * row_stride, row_stride));
        if (prefix_execution_splits[row] &&
            (*prefix_execution_splits[row] == 0 || *prefix_execution_splits[row] > row_stride)) {
            throw std::logic_error("forced-token execution split is outside its row");
        }
        lanes[row] = lane;
    }

    // token 计数是可选观测，不是语义：只有真有请求开着计数，才值得为它单独把 token 搬一趟设备。计数
    // 表本身在设备上（采样路径也往同一张表累加），所以只能就地用内核累加，Host 侧没有它的副本。
    const bool count_forced_tokens = std::any_of(
        lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(members.size()),
        [&](std::uint32_t lane) { return requests[lane].sampling_host.token_counts != nullptr; });
    if (count_forced_tokens) {
        work.reset();
        Tensor forced_ids =
            work.alloc(DType::I32, {checked_i32(static_cast<std::uint32_t>(row_major_tokens.size()),
                                                "forced-token batch exceeds int32")});
        CUDA_CHECK(cudaMemcpyAsync(forced_ids.data, row_major_tokens.data(), forced_ids.bytes(),
                                   cudaMemcpyHostToDevice, device.stream));
        for (std::size_t row = 0; row < members.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (requests[lane].sampling_host.token_counts == nullptr) { continue; }
            Tensor ids    = forced_ids.slice(0, static_cast<std::int32_t>(row * row_stride),
                                             static_cast<std::int32_t>(row_stride));
            Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(lane), 1)
                                .view({dimension(parameters.model.resources().public_token_count)});
            ops::increment_token_counts(ids, counts, device.stream);
        }
        work.reset();
    }

    // 逐行推进：从各行当前的前沿出发，把给出的 token 当作一次预填真正跑过模型——只有算过 KV，它们才
    // 算真的接进了序列。
    try {
        for (std::size_t row = 0; row < members.size(); ++row) {
            timing.resume_submit();
            const std::uint32_t lane = lanes[row];
            SequenceState& sequence  = active_sequence(lane);
            RequestControl& request  = requests[lane];
            const std::span<const TokenId> forced =
                row_major_tokens.subspan(row * row_stride, row_stride);
            const std::uint32_t base_ledger_frontier = sequence.ledger_frontier;
            const std::uint32_t base                 = sequence.execution_frontier;
            const std::uint32_t end                  = base + row_stride;
            const auto started                       = Clock::now();

            // DFlash 后端另有一份 draft 用的 context 缓存，它可能落在执行前沿之后：先把中间那一段补进去
            // 并提交 KV，两边的进度才算对齐，下面才能从这个基点往上算。
            if (is_masked_draft_backend(speculative_backend) &&
                sequence.dflash_context_frontier < base) {
                const std::array<std::uint32_t, 1> append_lanes{lane};
                const std::array<std::uint32_t, 1> append_starts{sequence.dflash_context_frontier};
                const std::array<std::uint32_t, 1> append_counts{base -
                                                                 sequence.dflash_context_frontier};
                enqueue_dflash_context_append(append_lanes, append_starts, append_counts);
                timing.begin_wait();
                device.synchronize();
                timing.end_wait();
                sequence.dflash_context_frontier = base;
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
                work.reset();
                timing.resume_submit();
            }

            // KV 的地址空间表述（映射）与页是两件事：这里只是把"这条序列现在用到 [0, end)"这个事实告
            // 诉地址空间，页该分配的时候自然会分配。
            ensure_sequence_kv_mapped(sequence, end, backend_kv_cache() ? end : 0U);

            // token 先进账本，模型随后为它们算 KV。插入后账本必须正好是 end + 1 格：多出来的那一格
            // 是"下一个 token 的位置"，也是前面那条前沿不变式的延续。
            sequence.ledger.insert(sequence.ledger.end(), forced.begin(), forced.end());
            if (sequence.ledger.size() != static_cast<std::size_t>(end) + 1U) {
                throw std::logic_error("forced-token continuation ledger has an invalid shape");
            }

            // DFlash 的解码入口要一份 Host 侧 ingress：这一趟跑哪些 lane、状态镜像的读写槽位、以及该
            // lane 在后端 KV 表里占的行。填好整份拷到设备，设备侧内核照它取数。
            if (is_masked_draft_backend(speculative_backend)) {
                if (!dflash || !io.dflash_decode || !sequence.kv ||
                    (backend_kv_cache() && !sequence.kv->backend)) {
                    throw std::logic_error("DFlash forced continuation state is incomplete");
                }
                *dflash_host_ingress                            = {};
                dflash_host_ingress->active_lanes[0]            = static_cast<std::int32_t>(lane);
                const StateImageSelectors selectors             = state_selectors(sequence);
                dflash_host_ingress->state_source_slots[0]      = selectors.source;
                dflash_host_ingress->state_destination_slots[0] = selectors.destination;
                dflash_host_ingress->dflash_kv_table_rows[0] =
                    sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend)
                                         : 0;
                CUDA_CHECK(cudaMemcpyAsync(io.dflash_decode->ingress.data, dflash_host_ingress,
                                           sizeof(qwen3_5::DFlashDecodeIngress),
                                           cudaMemcpyHostToDevice, device.stream));
            }

            // 分块预填：一次最多推进 prefill_chunk 个 token，因为一次前向能吃下的长度有限。每块算完
            // 立刻提交 KV、结算状态镜像的 fork、把尾部 hidden 拷回——这三件事是"这一块的 KV 已经可用"
            // 的凭据，也是下一块能接着算的前提。强制路径不允许顺手采样，所以 finalized 必须为假（它只
            // 会在这次预填把序列跑完并产出采样结果时为真）。
            std::uint32_t cursor = base;
            while (cursor < end) {
                const std::uint32_t count           = std::min(prefill_chunk, end - cursor);
                const StateImageSelectors selectors = state_selectors(sequence);
                execution::PrefillContext schedule_state{
                    {device, parameters, work, state_images->linear(),
                     replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
                     proposal_head},
                    text_kv_view(sequence),
                    mtp_kv_view(sequence),
                    decoder->text_kv,
                    decoder->mtp_cache(),
                    dflash ? &*dflash : nullptr,
                    cursor,
                    nullptr,
                    nullptr,
                    selectors.source,
                    selectors.destination,
                    0,
                    dflash_host_ingress};
                mark_workspace_usage(speculative_backend == SpeculativeBackend::Mtp
                                         ? workspace_plan.mtp_prefill
                                         : workspace_plan.text_prefill);
                if (is_masked_draft_backend(speculative_backend)) {
                    mark_workspace_usage(workspace_plan.dflash_context);
                }
                const execution::PrefillChunkResult result = execution::prefill_text_chunk(
                    schedule_state, sequence.ledger, count, std::nullopt, false);
                if (result.finalized || result.processed_tokens == 0 ||
                    result.processed_tokens > count) {
                    throw std::logic_error("forced-token prefill made invalid progress");
                }
                cursor += result.processed_tokens;
                sequence.text_kv_valid = cursor;
                if (speculative_backend == SpeculativeBackend::Mtp) {
                    sequence.mtp_kv_valid = cursor;
                } else if (is_masked_draft_backend(speculative_backend)) {
                    sequence.dflash_context_frontier = cursor;
                }
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
                settle_state_fork(sequence);
                copy_tail(sequence,
                          prefill_hidden.slice(
                              1, static_cast<std::int32_t>(result.processed_tokens) - 1, 1));
            }
            timing.begin_wait();
            device.synchronize();
            timing.end_wait();
            work.reset();

            // 设备侧真的落定之后才允许改身份与前沿：身份一旦写下就会被复用判定读到，不能建立在还没算完
            // 的 KV 上。重建工作随后按新前沿续上。
            commit_generated_prefix_identity(sequence, base_ledger_frontier, forced,
                                             prefix_execution_splits[row]);
            advance_rebuild_work(sequence, end, prefill_chunk);
            // 前沿推进到 end：账本前沿 = end + 1（那条"多一格"的不变式）；draft 计数清零、尾部 hidden
            // 记为有效，都是"序列已经干净地停在 end"的一部分。
            sequence.execution_frontier = end;
            sequence.ledger_frontier    = end + 1U;
            sequence.mtp_draft_count    = 0;
            sequence.tail_hidden_valid  = true;
            // 推进之后再复核一遍形状，确认这次强制追加真的把序列留在了合法前沿上。
            if (sequence.ledger.size() != sequence.ledger_frontier ||
                sequence.prefix_identity.size() != sequence.ledger_frontier ||
                sequence.prefix_digests.size() != sequence.ledger_frontier ||
                sequence.ledger.back() != forced.back()) {
                throw std::logic_error("forced-token commit did not establish a valid frontier");
            }
            // 分块预填可能把 KV 映射撑到比实际需要更远，这里收回正好的长度。
            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            // 这一批强制 token 的开销记在 decode 账上：对外它和一次解码属于同一类工作。
            request.timings.decode_seconds +=
                std::chrono::duration<double>(Clock::now() - started).count();
        }
        return timing.finish();
    } catch (...) {
        // 走到这里只能整批弃掉：前面的行可能已经推进过了，"半批生效"没法收拾（不像验收阶段那样可以
        // 整体拒绝）。先把设备停下来——同步本身失败要吞掉，因为真正要报的是引发清场的那个异常——再清
        // 工作区，最后把涉及的行标为执行失败。
        timing.begin_wait();
        try {
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        work.reset();
        clear_execution_failure_lanes(std::span<const std::uint32_t>(lanes.data(), members.size()));
        throw;
    }
}

// 一轮生成的结算：外部（Engine）已经对每一行作出裁决——接受几个 token、是否终止、是否取消——这里把
// 裁决落进账本，并把该释放的行释放掉。设备侧产出的 token 按外部裁决落进逻辑账本，走的就是这里。
//
// 裁决的合法性在这里独立复核，规则是：取消必须"零接受 + 终止"；非取消至少接受一个、不超过产出，并遵
// 守"要么全收继续跑，要么只收一个前缀但必须就此终止"。
//
// 票据本身是一次性的：拿到手先判断它是否有效，然后无论结果如何都立刻消费掉——后面就不必再操心"票据还
// 挂在外面"这件事。
CommitResult ProgramImpl::commit(PendingBatch&& pending,
                                 std::span<const runtime::CommitDecision> decisions,
                                 runtime::CommitObservation observation,
                                 runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    std::array<SequenceHandle, kMaximumConcurrency> members{};
    const auto input_rows       = ContractAccess::rows(pending);
    const std::size_t row_count = input_rows.size();
    for (std::size_t row = 0; row < row_count; ++row) { members[row] = input_rows[row]; }
    const bool valid = valid_pending(pending);
    ContractAccess::consume(pending);

    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    std::array<GenerationTimings, kMaximumConcurrency> timings{};
    std::array<SpeculativeStats, kMaximumConcurrency> speculative{};
    std::array<PendingKind, kMaximumConcurrency> pending_kinds{};
    // 失败兜底：票据已经消费掉了，但这次结算没做成。只能把涉及的行按执行失败清掉，并解锁。
    const auto release_members = [&]() noexcept {
        std::array<std::uint32_t, kMaximumConcurrency> failed_lanes{};
        std::size_t failed_count = 0;
        for (std::size_t row = 0; row < row_count; ++row) {
            if (ContractAccess::owner(members[row]) != this) { continue; }
            const std::uint32_t lane = ContractAccess::lane(members[row]).value;
            if (lane >= max_concurrency) { continue; }
            failed_lanes[failed_count++] = lane;
        }
        clear_execution_failure_lanes(
            std::span<const std::uint32_t>(failed_lanes.data(), failed_count));
        pending_transaction_.reset();
    };

    try {
        if (!valid || row_count == 0 || row_count > max_concurrency ||
            decisions.size() != row_count) {
            throw std::logic_error("pending transaction capability or decision shape is invalid");
        }
        std::array<std::uint32_t, kMaximumConcurrency> accepted{};
        std::array<std::uint8_t, kMaximumConcurrency> terminal{};
        std::array<std::uint8_t, kMaximumConcurrency> cancelled{};
        std::array<std::optional<std::uint32_t>, kMaximumConcurrency> prefix_execution_splits{};
        // 逐行复核裁决。三处硬约束：
        //   · 取消 = 零接受 + 终止（取消的行一个 token 都不落地，整行交还外部）；
        //   · 非取消 = 至少接受一个、不超过产出，且"全收才能继续跑，否则必须带着终止一起收前缀"；
        //   · 执行切分边界必须落在本次接受的范围之内，否则前缀身份会记到不属于它的位置上。
        // 另外取消不允许与全局 context 事务重叠：那张事务正握着设备侧状态，取消要放的资源在它手上。
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::uint32_t lane                = ContractAccess::lane(members[row]).value;
            lanes[row]                              = lane;
            const PendingCandidate& candidate       = requests[lane].pending;
            pending_kinds[row]                      = candidate.kind;
            const runtime::CommitDecision& decision = decisions[row];
            if (decision.cancelled && has_context_transaction()) {
                throw std::logic_error(
                    "active cancellation overlaps the global context transaction");
            }
            if ((decision.cancelled && (decision.accepted_tokens != 0 || !decision.terminal)) ||
                (!decision.cancelled &&
                 (decision.accepted_tokens == 0 || decision.accepted_tokens > candidate.produced ||
                  (!decision.terminal && decision.accepted_tokens != candidate.produced))) ||
                (decision.prefix_execution_split_after &&
                 (decision.cancelled || *decision.prefix_execution_split_after == 0 ||
                  *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                throw std::logic_error("pending transaction decision is invalid");
            }
            accepted[row]                = decision.accepted_tokens;
            terminal[row]                = decision.terminal ? 1U : 0U;
            cancelled[row]               = decision.cancelled ? 1U : 0U;
            prefix_execution_splits[row] = decision.prefix_execution_split_after;
            // 取消的行等一下就会被释放、统计字段随之清空，所以先抄下来——这是它们的最后一次机会。
            if (decision.cancelled) {
                timings[row]     = requests[lane].timings;
                speculative[row] = std::move(requests[lane].speculative_stats);
            }
        }

        // 真正的落账交给 resolve_pending_raw：逐行把接受范围写进账本、决定去向。它的耗时不重复计入
        // recorder 的固有段落，而是单独测量后"并入"（include），仍算在这张票据的 Post 阶段里。
        timing.pause();
        timing.include(
            resolve_pending_raw(std::span<const std::uint32_t>(lanes.data(), row_count),
                                std::span<const std::uint32_t>(accepted.data(), row_count),
                                std::span<const std::uint8_t>(terminal.data(), row_count),
                                std::span<const std::uint8_t>(cancelled.data(), row_count),
                                std::span<const std::optional<std::uint32_t>>(
                                    prefix_execution_splits.data(), row_count),
                                failed_timing));
        timing.resume_post();
        // 事务在这里解锁：从这一刻起又可以开始新的解码或追加。
        pending_transaction_.reset();

        // 逐行给出对外结果。取消的行是唯一会被当场物理释放的一种，所以一旦出现，物理世界版本号必须前推。
        CommitResult out;
        out.row_count          = row_count;
        bool released_resource = false;
        for (std::size_t row = 0; row < row_count; ++row) {
            if (decisions[row].cancelled) {
                // 取消 = 立即释放：lane 作废（外部手里关于它的旧句柄同时失效），资源回到池子。
                invalidate_lane(lanes[row]);
                released_resource = true;
                out.rows[row]     = CommitRowResult{
                        .disposition = runtime::CommitDisposition::CancelledReleased,
                        .timings     = timings[row],
                        .speculative = std::move(speculative[row]),
                };
            } else if (decisions[row].terminal) {
                // 终止但未取消：这一行不释放，转成 Finishable —— 它还要等 finish() 才真正收场（可能被
                // 目录化成可复用的续接点）。
                out.rows[row].disposition = runtime::CommitDisposition::Finishable;
                // 统计快照只在"还没有释放"的行上按需回传；ReleasedRowsOnly 是产品路径的选择，它不需要
                // 为即将结束的行再抄一份统计。
                if (observation == runtime::CommitObservation::AllRows) {
                    out.rows[row].timings     = requests[lanes[row]].timings;
                    out.rows[row].speculative = requests[lanes[row]].speculative_stats;
                }
            } else {
                out.rows[row].disposition = runtime::CommitDisposition::Active;
                if (observation == runtime::CommitObservation::AllRows) {
                    out.rows[row].timings     = requests[lanes[row]].timings;
                    out.rows[row].speculative = requests[lanes[row]].speculative_stats;
                }
            }

            // Begin 行额外担一件事：如果这次预填恰好把 prompt 走完，就把"prompt 前沿捕获"的票据发给外
            // 部——它要拿这张票据去取这一点的状态与 KV。只有 Begin 行、且没被取消/终止的行才有这份责任。
            if (pending_kinds[row] != PendingKind::Begin || decisions[row].cancelled) { continue; }
            RequestControl& request = requests[lanes[row]];
            if (decisions[row].terminal) {
                request.prefill.reset();
                continue;
            }
            if (!request.prefill) { continue; }
            RequestControl::Prefill& prefill = *request.prefill;
            if (prefill.cursor != prefill.prompt_tokens ||
                prefill.next_capture >= prefill.capture_groups.size() ||
                prefill.capture_groups[prefill.next_capture].frontier != prefill.prompt_tokens ||
                prefill.pending_capture_offer != 0) {
                throw std::logic_error("prompt-frontier capture carrier is inconsistent");
            }
            // 票据号 0 是"没有挂着的票据"的保留值，绕回时要跳过它。
            if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
            prefill.pending_capture_offer = next_capture_offer_id_;
            out.captures[row].emplace(ContractAccess::make_capture_offer(
                this, runtime::LaneId{lanes[row]}, lane_epochs[lanes[row]],
                prefill.pending_capture_offer));
        }
        // 只有真的释放过资源才推进物理世界版本号：它是"压力方案与封印计划该重算了"的信号。
        if (released_resource) { advance_resource_revision(); }
        out.timing = timing.finish();
        return out;
    } catch (...) {
        // 失败在这里等于"这批裁决没有生效"：票据早已作废，只能把涉及的行按执行失败清掉，并把事务解锁。
        timing.resume_post();
        release_members();
        throw;
    }
}

// 把一张待结算票据直接作废，不落账：上层放弃这一轮时走这里。没有裁决可复核——只有"这一批不要了"。
// 票据同样是一次性的，先消费再处理。
DiscardResult ProgramImpl::abort_pending(PendingBatch&& pending) noexcept {
    // 注意 DiscardResult 的 status 默认是"未消费"，因此每条成功的出口都必须显式写 Consumed：漏写就
    // 会退化成"没做成"。
    DiscardResult out;
    const auto rows  = ContractAccess::rows(pending);
    const bool valid = valid_pending(pending);
    out.row_count    = std::min<std::size_t>(rows.size(), kMaximumConcurrency);
    std::array<SequenceHandle, kMaximumConcurrency> members{};
    for (std::size_t row = 0; row < out.row_count; ++row) { members[row] = rows[row]; }
    ContractAccess::consume(pending);
    if (!valid) { return out; }
    std::array<std::uint32_t, kMaximumConcurrency> failed_lanes{};
    for (std::size_t row = 0; row < out.row_count; ++row) {
        failed_lanes[row] = ContractAccess::lane(members[row]).value;
    }
    // 全局 context 事务在场时，清场要推迟：那张事务正握着设备侧状态，最终由 fail_all_cleanup 统一收。
    // 于是这里连物理世界版本号都不推，返回的 status 停在默认的"未消费"——等于告诉上层"这条路径没有把
    // 这一批结清"。
    const bool deferred_to_fail_all = has_context_transaction();
    clear_execution_failure_lanes(
        std::span<const std::uint32_t>(failed_lanes.data(), out.row_count));
    pending_transaction_.reset();
    if (deferred_to_fail_all) { return out; }
    // 清掉的行归还了资源，物理世界变了。
    if (out.row_count != 0) { advance_resource_revision(); }
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

// 一条已终止（Finishable）序列的收场。两条出路互不共用：
//   · Released：直接拆掉（clear_lane_strict），资源回池子；
//   · Catalogued：把序列冻结成不可变的续接点（continuation）挂进目录，供后续请求复用——这是前缀复用
//     的来源，也是"结束"最值钱的一种。
// 走哪条路由请求自己声明（publish_continuation）。全程 noexcept：前置条件不满足就返回"未消费"。
FinishResult ProgramImpl::finish(SequenceHandle sequence) noexcept {
    FinishResult out;
    // 前置门：全局 context 事务或待结算票据在场时都不收场——它们可能正引用着这条序列的资源。
    if (has_context_transaction() || pending_transaction_ || !valid_sequence(sequence)) {
        return out;
    }
    const std::uint32_t lane               = ContractAccess::lane(sequence).value;
    RequestControl& request                = requests[lane];
    SequenceState& state                   = active_sequence(lane);
    const std::uint32_t continuation_index = active_continuations[lane];
    if (request.lifecycle != Lifecycle::Finishable) { return out; }
    if (!request.publish_continuation) {
        // 不目录化：直接拆掉。clear_lane_strict 会逐项确认资源确实还在自己手上，确认不了就不动手。
        if (!clear_lane_strict(state, request)) { return out; }
        out.disposition = runtime::FinishDisposition::Released;
        out.timings     = request.timings;
        out.speculative = std::move(request.speculative_stats);
        invalidate_lane(lane);
        advance_resource_revision();
        out.status = runtime::ConsumeStatus::Consumed;
        return out;
    }
    // 任何合法的终止路径都会先把借来的状态镜像 fork 结算掉。借来的来源不能当成本续接点的直接端点：若
    // 这条发布不变式没有成立，就退回到"终止即丢弃"（返回未消费，交给上层走另一条路）。
    if (state.state.fork_pending && state.state.borrows_read()) { return out; }
    // 先把摘要要用的内存腾出来：一旦开始改动状态就不该再抛异常，所以容量必须在动手之前备好。
    try {
        out.summary.long_anchors.reserve(state.long_anchors.size());
    } catch (...) { return out; }
    try {
        if (state.state.fork_pending) {
            // 撤掉借来的写副本。主动捕获来的读副本仍是这条序列的主生命期：把它发布成端点等于保留那份直
            // 接所有权；只要还有别的检查点引用存在，它就依旧无法被独占归属与释放。
            const StateImageHandle source      = state.state.read;
            const StateImageHandle destination = state.state.write;
            state_store->abort_fork(source, destination);
            if (!state_store->release(destination)) { return out; }
            state.state = ActiveStateBinding{.read = source, .write = source};
        }
        if (state.reserved_state) {
            // 预留了却没用上的状态镜像：归还。
            if (!state_store->release(*state.reserved_state)) { return out; }
            state.reserved_state.reset();
        }
        if (state.rewrite_state && *state.rewrite_state == state.state.read) {
            // 重写检查点若正好就是端点，要先退掉它自己持的那一份检查点引用；若连一份引用都没有，说明账
            // 本已经不一致，宁可退场也不能瞎退。
            if (state_store->checkpoint_references(*state.rewrite_state) == 0) { return out; }
            state_store->release_checkpoint_reference(*state.rewrite_state);
            state.rewrite_state.reset();
            state.rewrite_checkpoint = {};
        }
        // 目录化的前提是端点不可变：活跃可变状态要冻结成不可变检查点；已经是检查点就正好；其它角色说明
        // 状态不对，退场。
        if (state_store->role(state.state.read) == StateImageRole::ActiveMutable) {
            state_store->freeze(state.state.read);
        } else if (state_store->role(state.state.read) != StateImageRole::CheckpointImmutable) {
            return out;
        }
        state.endpoint_valid = true;
        refresh_state_views(state);
        // KV 也要一起钉住：目录化出去的续接点必须能取到 execution_frontier 那么长的 KV，所以把"至少保留
        // 到这个长度"的要求登记给地址空间。
        text_kv_addresses->set_checkpoint_requirement(state.kv->text, state.execution_frontier);
        if (state.kv->backend) {
            backend_kv_addresses->set_checkpoint_requirement(*state.kv->backend,
                                                             backend_kv_valid(state));
        }
        populate_continuation_summary(state, out.summary);
        // 目录化之后它不再算活跃引用：后面的释放判定看的是检查点引用，不是活跃引用。
        out.summary.active_references = 0;
    } catch (...) { return out; }
    // 状态已经定型，收尾只剩"交还"：活跃共享引用、增长额度、KV 绑定依次释放，请求回到 Empty。走到这里
    // 之前任何一步失败都还是"没做成"，从这里开始只可能做成。
    release_active_shared_references(state);
    release_sequence_growth_entitlement(state);
    unbind_sequence_kv(state);
    request.active_resources                    = {};
    request.optional_resources                  = {};
    request.lifecycle                           = Lifecycle::Empty;
    request.pending                             = {};
    continuation_slots[continuation_index].role = ContinuationSlotRole::Catalogued;
    active_continuations[lane]                  = continuation_capacity;
    invalidate_lane(lane);
    // 交出去的句柄带槽位代次：槽位日后被回收重用，旧句柄自然失效。
    out.continuation.emplace(ContractAccess::make_continuation(
        this, continuation_index, continuation_slots[continuation_index].generation));
    out.timings     = request.timings;
    out.speculative = std::move(request.speculative_stats);
    out.disposition = runtime::FinishDisposition::Catalogued;
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

// 半路作废：与 finish 的 Released 支路是同一条拆除路径，区别只在用途——它用在还没有终止的序列上
// （Active / Prefilling / Finishable 都能拆）。Pending 与 Empty 不接受：前者手里有票据，该走
// commit / abort_pending；后者本来就是空的。
AbortResult ProgramImpl::abort(SequenceHandle sequence) noexcept {
    AbortResult out;
    if (has_context_transaction() || pending_transaction_ || !valid_sequence(sequence)) {
        return out;
    }
    const std::uint32_t lane = ContractAccess::lane(sequence).value;
    RequestControl& request  = requests[lane];
    if (request.lifecycle == Lifecycle::Pending || request.lifecycle == Lifecycle::Empty) {
        return out;
    }
    SequenceState& state = active_sequence(lane);
    if (!clear_lane_strict(state, request)) { return out; }
    out.timings     = request.timings;
    out.speculative = std::move(request.speculative_stats);
    invalidate_lane(lane);
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

// 释放一张续接点句柄，也就是"这个目录项我不要了"。除句柄本身有效，还要确认没有物化事务正把它钉住
// （materialization_pins）——有事务拿它当落脚点就不能放。检查与执行照例分两步：前一个只回答能不能，
// 后一个假定已经能（失败即 std::terminate）。
ReleaseResult ProgramImpl::release_continuation(ContinuationHandle&& continuation) noexcept {
    ReleaseResult out;
    const std::uint32_t index      = ContractAccess::index(continuation);
    const std::uint64_t generation = ContractAccess::epoch(continuation);
    const bool valid               = !has_context_transaction() && !pending_transaction_ &&
                       valid_continuation(continuation) && !materialization_pins(index, generation);
    if (!valid) { return out; }
    try {
        if (!can_release_continuation_slot_strict(index)) { return out; }
    } catch (...) { return out; }
    release_continuation_slot_strict(index);
    ContractAccess::consume(continuation);
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

// 共享前缀能否释放的守卫：只回答"能不能"，什么都不改。要求槽位角色符合预期（调用者用它区分"目录中
// 的共享前缀"与其它状态）、没有任何活跃引用、状态镜像与 KV 都有效、KV 地址空间允许释放，并且这个槽位
// 确实持着自己那一份检查点引用——若它是最后一份，还得确认底层允许在退掉引用之后释放。
bool ProgramImpl::can_release_shared_prefix_state(std::uint32_t index,
                                                  SharedPrefixSlotRole expected_role) const {
    if (index >= shared_prefix_capacity || !state_store || !text_kv_addresses ||
        shared_prefix_slots[index].role != expected_role) {
        return false;
    }
    // 有活跃引用意味着还有请求正靠它复用，不能放。
    const SharedPrefixState& shared = shared_prefix_states[index];
    if (shared.active_references != 0 || !shared.kv || !shared.identity ||
        !state_store->valid(shared.state) || !text_kv_addresses->can_release(shared.kv->text) ||
        (shared.kv->backend &&
         (!backend_kv_addresses || !backend_kv_addresses->can_release(*shared.kv->backend)))) {
        return false;
    }
    // 自己至少持有一份检查点引用；若只剩这一份，就让底层再确认一次"退掉之后确实能释放"。
    const std::uint32_t state_references = state_store->checkpoint_references(shared.state);
    return state_references != 0 &&
           (state_references != 1 ||
            state_store->can_release_after_checkpoint_references(shared.state, 1));
}

// 真正放掉一份共享前缀：严格版本——前置检查不过就 std::terminate。走到这一步已经没有能接住失败的上层，
// 宁可就地终止，也不能留着一半释放的状态继续跑。
//
// 释放顺序有讲究：先放 KV（后端 KV 再到文本 KV），再退掉自己那份检查点引用，只有那确实是最后一份时才
// 真正释放状态镜像。全部放完槽位代次前推（旧句柄随之失效），并顺手清掉已无人引用的 host KV extent。
detail::PhysicalResources
ProgramImpl::release_shared_prefix_state_strict(std::uint32_t index,
                                                SharedPrefixSlotRole expected_role) noexcept {
    try {
        if (!can_release_shared_prefix_state(index, expected_role)) { std::terminate(); }
        SharedPrefixState& shared               = shared_prefix_states[index];
        SharedPrefixSlot& slot                  = shared_prefix_slots[index];
        const detail::PhysicalResources removed = owner_exclusive_resources(shared);
        const bool last_state_reference = state_store->checkpoint_references(shared.state) == 1;
        if (shared.kv->backend && !backend_kv_addresses->release(*shared.kv->backend)) {
            std::terminate();
        }
        if (!text_kv_addresses->release(shared.kv->text)) { std::terminate(); }
        state_store->release_checkpoint_reference(shared.state);
        if (last_state_reference && !state_store->release(shared.state)) { std::terminate(); }

        shared    = SharedPrefixState{};
        slot.role = SharedPrefixSlotRole::Free;
        if (++slot.generation == 0) { ++slot.generation; }
        if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
        return removed;
    } catch (...) { std::terminate(); }
}

// 对外入口：先做与 strict 版本相同的前置检查（角色按"目录中的共享前缀"要求），过了才真释放，最后消费
// 句柄并推进物理世界版本号。检查不过就根本走不到释放，句柄仍留在原主人手上。
ReleaseResult ProgramImpl::release_shared_prefix(SharedPrefixHandle&& handle) noexcept {
    ReleaseResult out;
    const std::uint32_t index      = ContractAccess::index(handle);
    const std::uint64_t generation = ContractAccess::epoch(handle);
    const bool valid =
        !has_context_transaction() && !pending_transaction_ && valid_shared_prefix(handle);
    if (!valid || index >= shared_prefix_capacity ||
        shared_prefix_slots[index].generation != generation) {
        return out;
    }
    try {
        if (!can_release_shared_prefix_state(index, SharedPrefixSlotRole::Catalogued)) {
            return out;
        }
    } catch (...) { return out; }
    (void)release_shared_prefix_state_strict(index, SharedPrefixSlotRole::Catalogued);
    ContractAccess::consume(handle);
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

// 进程级收场：把所有还在飞行中的事务、lane、槽位一次清干净。这里不追求语义完整，只保证两件事——不崩，
// 以及不错放（该等设备停下来的时候先等）。所有失败都被吞掉，因为调用它的场合本身已经在收摊。
void ProgramImpl::fail_all_cleanup() noexcept {
    pending_transaction_.reset();
    // 飞行中的事务先按类型撤：传输流里可能还有在途的拷贝，所以撤回之前必须先让流停下来，否则可能撤掉
    // 还在被读写的缓冲。
    if (auto* transaction = std::get_if<ActiveCaptureTransaction>(&context_transaction_)) {
        if (transaction->transfer_submitted && device.transfer_stream != nullptr) {
            (void)cudaStreamSynchronize(device.transfer_stream);
        }
        abort_active_capture(*transaction);
    }
    if (auto* transaction = std::get_if<MaterializationTransaction>(&context_transaction_)) {
        if (transaction->transfer_submitted && device.transfer_stream != nullptr) {
            (void)cudaStreamSynchronize(device.transfer_stream);
        }
        release_materialization_staging(*transaction);
    }
    context_transaction_.emplace<std::monostate>();
    // lane：还挂着序列的就 best-effort 拆掉（不求账本自洽，只求别漏别崩），然后一律作废代次。
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (active_continuations[lane] < continuation_capacity) {
            clear_lane_best_effort(active_sequence(lane), requests[lane]);
        }
        invalidate_lane(lane);
    }
    // 续接点槽位：只要不是 Free 就交还，不管它当时处在哪种中间态。
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role != ContinuationSlotRole::Free) {
            release_continuation_slot_best_effort(index);
        }
    }
    // 共享前缀：先主动放弃自己那一份活跃引用，再走正常释放通道（它此时必然已经没有别的活跃引用）。
    for (std::uint32_t index = 0; index < shared_prefix_capacity; ++index) {
        if (shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued) { continue; }
        shared_prefix_states[index].active_references = 0;
        auto handle =
            ContractAccess::make_shared_prefix(this, index, shared_prefix_slots[index].generation);
        (void)release_shared_prefix(std::move(handle));
    }
}


} // namespace ninfer::models::qwen3_5::detail
