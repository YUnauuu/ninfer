#pragma once

#include "runtime/contract/request.h"
#include "runtime/contract/timing.h"
#include <compare>
#include <span>

// ============================================================================
// runtime/contract/execution.h —— 逐轮（round）执行的契约词汇
// ============================================================================
//
// request.h 管"一个请求"，这里管"一批请求跑一次模型"：Program 上报提交状态与轮次结果，
// Engine 下行 CommitDecision 与 RoundBudget。主线是 **Program 报告"声称"，Engine 独立复核**。

namespace ninfer::runtime {

struct LaneId {
    std::uint32_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(LaneId, LaneId) noexcept  = default;
    [[nodiscard]] friend constexpr auto operator<=>(LaneId, LaneId) noexcept = default;
};

// FinishResult / AbortResult / ReleaseResult / DiscardResult 共用的报告（见 program.h）。
// **默认值是 InvariantMismatch**：成功路径必须显式赋成 Consumed，漏赋值就留在失败态
//（Engine 侧到处检查 `status != ConsumeStatus::Consumed` 并据此拒绝继续）。
enum class ConsumeStatus : std::uint8_t {
    Consumed,
    InvariantMismatch,
};

// 这一行提交之后的去向，由 Program 按行给出。Finishable 是"已终止但**资源还没结算**"，该行
// 仍需持有 SequenceHandle 与完整预留。Engine 独立复核 cancelled ? CancelledReleased : (terminal ? Finishable : Active)。
enum class CommitDisposition : std::uint8_t {
    Active,
    Finishable,
    CancelledReleased,
};

// 提交时"要不要顺带回传每行的统计快照"，是成本选项而不是语义选项。产品路径传 ReleasedRowsOnly
// （默认值是 AllRows —— 默认值不等于推荐值）；它只包住 timings/speculative 的赋值，是纯观测。
// The product Engine only needs statistics for rows whose sequence is released by commit.
// Direct diagnostic callers may temporarily request cumulative snapshots for every row.
enum class CommitObservation : std::uint8_t {
    ReleasedRowsOnly,
    AllRows,
};

// 下行：Engine 对"这一行该接受哪些 token"的裁决，合法性由 Engine 把关（engine_core.h）；
// **要么全收继续跑，要么只收一个前缀但必须就此终止**。
struct CommitDecision {
    std::uint32_t accepted_tokens = 0;
    bool terminal                 = false;
    bool cancelled                = false;
    // Copied unchanged from the corresponding OutputDecision; still relative to this row's
    // accepted span.
    std::optional<std::uint32_t> prefix_execution_split_after;
};

// 预填开场的摘要，身兼两职，而**第二职才是它必须有 operator== 的原因**：Engine 准入时的
// **承诺**（request_record 的 admitted_begin，流式 GenerationStart 从它来），以及预填完成时
// Program 回报的**实测值**；engine_core.h 用 `progress.summary != *request->admitted_begin`
// 逐字段比对，不等即抛 std::logic_error("runtime Begin summary differs from committed admission")。
// 给结构加字段却忘了让它语义可比，这里就会静默失去校验能力。
struct BeginSummary {
    std::uint32_t prompt_tokens        = 0;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;

    [[nodiscard]] friend constexpr bool operator==(BeginSummary, BeginSummary) noexcept = default;
};

struct GeneratedRound {
    std::span<const TokenId> tokens;
};

// 批量的一轮结果：第 r 行起于 tokens.data() + r * row_stride，长度看 row_counts[r]。
// **row_stride 是行距不是行长**。（缓冲按最大的行开槽，不能拿它当长度用。）
struct BatchedGeneratedRound {
    std::span<const TokenId> tokens;
    std::span<const std::int32_t> row_counts;
    std::uint32_t row_stride = 1;
    ExecutionTiming timing;
};

// 一次预填步的结果。processed_prompt_tokens 是**本次**推进的增量，不是累计量。
struct PrefillStepResult {
    BeginSummary summary;
    GeneratedRound round;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    ExecutionTiming timing;
};

// 逐行预算：这一行还允许再生成多少个 token。同一批里各请求的剩余额度不同，所以按行传。
struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

} // namespace ninfer::runtime
