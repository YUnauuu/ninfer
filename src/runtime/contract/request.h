#pragma once

#include "ninfer/types.h"
#include <atomic>
#include <cstdint>
#include <optional>

// ============================================================================
// runtime/contract/request.h —— 请求侧的"解析后"契约
// ============================================================================
//
// 跨边界（Engine ↔ Program）传递的纯值类型：向下是"要跑什么"（可选性与默认值都已被
// resolve_sampling 消掉，见 contract/sampling.h），向上是 OutputDecision（"实际发生了什么"，
// 有 Engine 侧校验）。全是**被信任的输入**，下游见到非法值按不变量违反处理（抛逻辑错误）。

namespace ninfer::runtime {

// Engine has already selected the model/mode preset, applied every explicit override,
// and validated these values before constructing the runtime request.
// requested_output_tokens 是原始请求数而**不是生效数**（生效数见 RequestPlanSummary）。
struct ResolvedExecutionOptions {
    ResolvedSamplingParameters sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
};

struct ResolvedRequestOptions {
    ResolvedExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

// 这次提交完成后请求该往哪走 —— 粒度是"下一跳的性质"，不是成功/失败。ApplyTargetControl 表示
// 模型先停一下、Engine 要注入目标控制内容再继续，**它不是终止**，且只允许出现在 finish_reason == None 上。
enum class ContinuationAction : std::uint8_t {
    Decode,
    ApplyTargetControl,
};

// 前端对"这一轮模型输出该如何被接受"的裁决：决定权在前端是因为"哪些 token 算数"是模型语义
// 问题（停止串可能跨 token 边界、思考段与正文分道……），而 Engine 是模型无关的。
struct OutputDecision {
    // 这是**接受**数不是生成数：投机解码下模型一次给多个候选，前端可以只认一部分，其余丢弃。
    std::uint32_t accepted_tokens   = 0;
    // None 表示"没结束，继续跑"；其它值表示终止以及原因。
    FinishReason finish_reason      = FinishReason::None;
    ContinuationAction continuation = ContinuationAction::Decode;
    // Empty or one token-aligned frontier within the accepted span where model-history
    // reconstruction gains an execution split. Frontend owns detection; Engine only transports
    // this relative position.
    // 相对本行**已接受区间起点**的偏移（空表示没有切点），不能是 0 也不能大于 accepted_tokens。
    std::optional<std::uint32_t> prefix_execution_split_after;

    [[nodiscard]] bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

// Non-owning cancellation observation used while the worker advances a context transaction. The
// request record owns the flag for longer than Program can retain this view.
// 宿主随上下文事务而定（见 engine_core.h 构造这个视图的 switch）；语义是"表达了取消意愿"而不是
// "立刻停止"，所以下一个安全边界之前仍可能产出有效 token，带 Cancelled 的终止原因返回。
struct CancellationFlagView {
    const std::atomic<bool>* flag = nullptr;

    [[nodiscard]] bool requested() const noexcept {
        // acquire 序，与 engine_core.h 里各写入点的 release store 配对；空指针表示无取消通道。
        return flag != nullptr && flag->load(std::memory_order_acquire);
    }
};

// 一个请求在**准入阶段**得出的计划摘要："打算怎么跑、要占多少公共资源"。
struct RequestPlanSummary {
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t reusable_prompt_tokens  = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    FinishReason effective_limit_reason   = FinishReason::None;
    PrefixReusePath prefix_reuse_path     = PrefixReusePath::Root;
    // 预期占用的调度**服务轮次**数，与"要花多少 FLOPs"无关：计法见 request_plan.cpp 的
    // projected_service_work() —— 在选中的复用基点上做一次 shared promotion 不执行任何模型
    // 计算，但仍然算一个服务单元。真实算力在 resources.h 的 PrefillWork 里，两者刻意分开。
    std::uint64_t service_work_quanta     = 0;
    // 终止时是否值得把状态发布成可复用的延续：false 时提交阶段直接释放该 lane 状态（见 commit.cpp）。
    bool publish_continuation             = true;
};

} // namespace ninfer::runtime
