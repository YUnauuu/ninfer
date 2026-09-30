#pragma once

#include "core/nvtx.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/admission_policy.h"
#include "runtime/engine/generation_budget.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// ============================================================================
// runtime/engine/request_record.h —— 一条在册请求的唯一可变记录
// ============================================================================
//
// EngineCore 用 std::shared_ptr<RequestRecord> 代表"一条被受理的请求"，这个结构就是那份记录的实体。
// 它的方向感来自三个同时碰它的线程：
//
//     调用方线程 ──submit 填好成品与选项──▶ [ 记录 ] ◀──worker 推进执行、写下结果
//                                            ▲
//     消费者线程 ──wait() 搬事件、取结果──────┘
//
//   * **worker 是唯一的变更 owner**：model_state、budget、generated、各类计时都只由它写。调用方线程
//     进来时只做两件事——在受理前填好字段，之后只读快照。
//   * 消费者线程与 worker 之间只通过 mutex + cv 交接（stream_start / stream_progress / events /
//     result / error / response_done 那一组），worker 写、wait() 搬。
//   * 唯一的反向通道是 cancelled：调用方置一个 atomic 标记，worker 在下一次稳定边界上收拢。这正是
//     "销毁句柄 = 取消"能在不阻塞调用方的前提下成立的原因。
//
// 于是字段天然分四组，读这个文件时按组读：
//   1. 受理时冻结的身份与选项——请求是谁、按什么规则算，此后不再改。
//   2. 执行态——worker 的状态机，以及它为此持有的资源（lane、sequence、plan、budget）。
//   3. 产出与观测——终态时会被汇总进 GenerationResult 的东西。
//   4. 发布面——消费者线程会碰的那一半，以及两个归还 outstanding 名额的标记。
//
// 记录的生命周期由 shared_ptr 决定，而不是由 Engine 决定：GenerationHandle 持有它，所以句柄能活得比
// Engine 长，Engine 析构不会打断还在跑的请求。

namespace ninfer::runtime {

// Engine 自己拥有的三段 Host 工作，用来把"延迟暴露"记到正确的去处。口径与 <ninfer/types.h> 里
// GenerationEngineTiming 的说明一致：exposed 值衡量的是**这条请求被同一批执行单位拖住了多久**，
// 所以并发请求之间不能相加。
enum class RequestEngineHostPhase : std::uint8_t {
    Boundary,     // 一次执行单位的边界处理（准入判定、状态推进、提交前准备）
    CommitOutput, // 提交并采用一轮结果（token 落账、输出发布）
    Maintenance,  // 维护性工作（资源回收、快照发布等）
};

// 一条请求的 Host 侧计时累加器：worker 在每个边界把它被拖住的耗时"暴露"进来，终态时经
// public_snapshot() 换算成公开的 GenerationEngineTiming。
//
// 两种累加方式的分工是刻意的：expose_engine 记 Engine 自己的三段相位，decode_member 表示这次工作
// 同时属于 decode 口径（同一段时间被两个视角看到，是子集关系而不是相加关系，见 types.h 同名说明）；
// expose_program 记 Program 回报的提交 / 收尾 / 等 Device 时间。
//
// prefill_units / decode_rounds / control_units 三个计数与时间不是一回事：它们回答的是"这条请求被算进
// 了哪些执行单位、各多少次"，也就是 exposed 时间的来源。
struct RequestHostTiming {
    std::uint64_t queue_wait_ns                   = 0;
    std::uint64_t engine_boundary_exposed_ns      = 0;
    std::uint64_t program_submit_exposed_ns       = 0;
    std::uint64_t program_post_exposed_ns         = 0;
    std::uint64_t engine_commit_output_exposed_ns = 0;
    std::uint64_t engine_maintenance_exposed_ns   = 0;
    std::uint64_t device_wait_exposed_ns          = 0;
    std::uint64_t decode_host_exposed_ns          = 0;
    std::uint64_t decode_device_wait_exposed_ns   = 0;
    std::uint64_t prefill_units                   = 0;
    std::uint64_t decode_rounds                   = 0;
    std::uint64_t control_units                   = 0;

    void expose_engine(RequestEngineHostPhase phase, std::uint64_t elapsed_ns,
                       bool decode_member) noexcept {
        switch (phase) {
        case RequestEngineHostPhase::Boundary:
            engine_boundary_exposed_ns += elapsed_ns;
            break;
        case RequestEngineHostPhase::CommitOutput:
            engine_commit_output_exposed_ns += elapsed_ns;
            break;
        case RequestEngineHostPhase::Maintenance:
            engine_maintenance_exposed_ns += elapsed_ns;
            break;
        }
        if (decode_member) { decode_host_exposed_ns += elapsed_ns; }
    }

    void expose_program(ExecutionTiming timing, bool decode_member) noexcept {
        program_submit_exposed_ns += timing.submit_host_ns;
        program_post_exposed_ns += timing.post_host_ns;
        device_wait_exposed_ns += timing.device_wait_ns;
        if (decode_member) {
            decode_host_exposed_ns += timing.host_ns();
            decode_device_wait_exposed_ns += timing.device_wait_ns;
        }
    }

    [[nodiscard]] GenerationEngineTiming public_snapshot() const noexcept {
        constexpr double kNanosecondsToSeconds = 1.0e-9;
        return GenerationEngineTiming{
            .queue_wait_seconds = static_cast<double>(queue_wait_ns) * kNanosecondsToSeconds,
            .engine_boundary_exposed_seconds =
                static_cast<double>(engine_boundary_exposed_ns) * kNanosecondsToSeconds,
            .program_submit_exposed_seconds =
                static_cast<double>(program_submit_exposed_ns) * kNanosecondsToSeconds,
            .program_post_exposed_seconds =
                static_cast<double>(program_post_exposed_ns) * kNanosecondsToSeconds,
            .engine_commit_output_exposed_seconds =
                static_cast<double>(engine_commit_output_exposed_ns) * kNanosecondsToSeconds,
            .engine_maintenance_exposed_seconds =
                static_cast<double>(engine_maintenance_exposed_ns) * kNanosecondsToSeconds,
            .device_wait_exposed_seconds =
                static_cast<double>(device_wait_exposed_ns) * kNanosecondsToSeconds,
            .decode_host_exposed_seconds =
                static_cast<double>(decode_host_exposed_ns) * kNanosecondsToSeconds,
            .decode_device_wait_exposed_seconds =
                static_cast<double>(decode_device_wait_exposed_ns) * kNanosecondsToSeconds,
            .prefill_units = prefill_units,
            .decode_rounds = decode_rounds,
            .control_units = control_units,
        };
    }
};

// worker 侧的执行状态机：一条请求从进队列到"模型侧结束"的必经状态。它有两重用途——Scheduler 按它决定
// 这条请求现在能参加哪种执行单位，EngineCore 按它给出 RuntimeStats 里那几个 gauge。
//
//   Waiting       —— 已受理、还没轮到准入（名额已占，但还没有 lane 与 sequence）
//   Materializing —— 正在做资源 transition：把复用的上下文变成可执行状态。全引擎至多一条处于此态，
//                    且此时**不能**持有 lane
//   Prefill       —— 已 Active，正在推进提示前沿
//   DecodeReady   —— 可以加入下一轮紧凑 decode
//   ControlReady  —— 有 Engine 注入的目标控制 token 要算（例如思考预算耗尽后的收尾指引）。它与 decode
//                    分成不同的执行单位，因为那些 token 是"喂进去的"而不是模型生成的
//   ModelFinished —— 模型执行已经终止
//
// 注意它止于 ModelFinished：文档里的 TerminalPending / Finished 不在这里表达——模型侧结束之后还有
// capture（若有）与终态发布，那两件由 EngineCore 的资源与发布流程承担，见下面的两个 release 标记。
enum class EngineRequestState : std::uint8_t {
    Waiting,
    Materializing,
    Prefill,
    DecodeReady,
    ControlReady,
    ModelFinished,
};

// 一条请求的全部可变状态。worker 是唯一变更 owner，消费者线程只碰最后一组（发布面）。
template <class ModelContract>
struct RequestRecord {
    // 记录本身不知道自己在服务哪个模型：所有模型侧类型都从 ModelContract 取，调度、发布、资源这套逻辑
    // 因此对所有模型只有一份。注意这里的 PreparedPrompt 是**模型侧产物**——公共 <ninfer/engine.h> 里的
    // PreparedPrompt 只是它的外壳。
    using Clock          = std::chrono::steady_clock;
    using PreparedPrompt = typename ModelContract::PreparedPrompt;
    using OutputSession  = typename ModelContract::OutputSession;
    using BasePlan       = typename ModelContract::RequestBasePlan;
    using SequenceHandle = typename ModelContract::SequenceHandle;
    // 流式发布的两种事件。Aggregate 消费者不会产生 events——发布面只在 Streaming 上启用。
    using StreamEvent    = std::variant<GenerationTimingObservation, OutputDelta>;

    // 一条记录只构造一次，参数就是受理那一刻的全部事实。构造即开始 NVTX 的异步 range：它跟着请求的
    // 所有权在 submit 线程、worker、消费者线程之间走，所以进程视角下这条请求的区间是完整的一段。
    RequestRecord(std::uint64_t request_identity, std::uint64_t publication_sequence,
                  PreparedPrompt input, OutputSession output_session, PromptSummary summary,
                  double frontend_seconds, ResolvedRequestOptions request_options,
                  OutputConsumerMode output_consumer, GenerationObservationOptions observation,
                  Clock::time_point limit, Clock::time_point submit_time)
        : generation_range(nvtx::Name::Generate, nvtx::Category::Runtime, request_identity),
          id(request_identity), publication_order(publication_sequence), prompt(std::move(input)),
          output(std::move(output_session)), prompt_summary(std::move(summary)),
          prepare_seconds(frontend_seconds), options(std::move(request_options)),
          consumer_mode(output_consumer), observation(observation), deadline(limit),
          submitted(submit_time) {}

    RequestRecord(const RequestRecord&)            = delete;
    RequestRecord& operator=(const RequestRecord&) = delete;

    // 状态查询：给 Scheduler / EngineCore 用，避免各处重复比对 model_state。
    [[nodiscard]] bool is_waiting() const noexcept {
        return model_state == EngineRequestState::Waiting;
    }

    [[nodiscard]] bool is_prefilling() const noexcept {
        return model_state == EngineRequestState::Prefill;
    }

    [[nodiscard]] bool is_materializing() const noexcept {
        return model_state == EngineRequestState::Materializing;
    }

    [[nodiscard]] bool is_decode_ready() const noexcept {
        return model_state == EngineRequestState::DecodeReady;
    }

    [[nodiscard]] bool is_control_ready() const noexcept {
        return model_state == EngineRequestState::ControlReady;
    }

    [[nodiscard]] bool is_model_finished() const noexcept {
        return model_state == EngineRequestState::ModelFinished;
    }

    // ---- 组 1：受理时冻结的事实 ----
    //
    // 这一组之后都不再改，因此谁在读它都不需要同步：请求是哪一个（id），以及完成次序的权威
    // （publication_order —— 提交更晚的序号更大，既用于同批候选之间定序，也是上下文缓存里共享前缀的
    // 发布权威）、按哪套规则执行（options、consumer_mode、observation）、以及排队期限（deadline —— 注意
    // 它只管**排队**这一段，不是整条请求的期限；提交时折算成绝对时刻，过期直接以 QueueTimeout 结束）。
    //
    // prompt 与 output 是"要算什么"与"怎么判停"两块模型侧状态：前者是渲染好的提示，后者（OutputSession）
    // 承载停止策略、思考控制、工具调用解析——输出语义的落点在这里，Engine 负责发布结果而不是解释它。
    nvtx::ScopedAsyncRange generation_range;
    const std::uint64_t id;
    const std::uint64_t publication_order;
    PreparedPrompt prompt;
    OutputSession output;
    PromptSummary prompt_summary;
    double prepare_seconds = 0.0;
    ResolvedRequestOptions options;
    const OutputConsumerMode consumer_mode;
    const GenerationObservationOptions observation;
    Clock::time_point deadline;
    Clock::time_point submitted;

    // ---- 组 2：执行态（worker 独占）----
    //
    // 只被观测需求驱动的时间戳是懒记的：admitted_at 只在开启任一观测时写入，first_token/last_token 则
    // 跟着 token 边界走。它们一起构成 prompt 墙钟与 generation 墙钟（N 个 token 是 N-1 个区间，见
    // types.h 里 GenerationTimings 的说明）。
    std::optional<Clock::time_point> admitted_at;
    std::optional<Clock::time_point> first_token;
    std::optional<Clock::time_point> last_token;
    // 排队等待只记一次：准入发生的那个边界把 queue_wait_ns 结算掉，之后的等待属于别的相位。
    bool queue_wait_recorded = false;

    // 生成预算：受理时按"请求的输出上限"与"剩余上下文"取小定下，并同时定死超限时的 finish reason
    // （OutputLimit 还是 ContextCapacity）。它只减不增，是这条请求能生成多少 token 的唯一权威。
    std::optional<GenerationBudget> budget;

    // 准入时**已提交**的复用选择：复用了多少前缀、走的是哪条复用路径。消费者看到的 GenerationStart 与
    // 之后的 PromptProgress 都锚在它上面，所以进度在整条请求里保持自洽。
    std::optional<BeginSummary> admitted_begin;
    // 执行实际回报的提示前沿。两者刻意分开：一个负责对外发布（准入时就能给的事实），一个负责终态结算
    // （真的跑了多少复用）。
    std::optional<BeginSummary> begin;

    // 产出积累：generated 是所有被采纳的 token，**包含** Engine 注入的目标控制 token，所以它与"模型
    // 生成了多少 token"不是一回事；content / reasoning 按通道分开存可见文本。
    std::vector<TokenId> generated;
    std::string content;
    std::string reasoning;

    // 准入换来的两个身份，终态时归还：lane 是 Engine 的长期 active 位置（同一时刻至多 max_concurrency 条
    // 请求持有它），sequence 是 Program 侧的序列句柄。要参加任何执行单位，两者都必须存在。
    // 注意 lane 与"outstanding 名额"不是一回事，也不能互相推导：名额从 submit 起就占着（见组 4 的两个
    // release 标记），lane 要等准入拿到；StateImage/KV 资源与 compact row 又是另外两种身份。
    std::optional<LaneId> lane;
    std::optional<SequenceHandle> sequence;

    // 唯一的反向通道（调用方 → worker）：取消只置标记，真正的收敛发生在 worker 的下一个稳定边界，
    // 因此放弃句柄绝不阻塞调用方，也绝不从调用方线程去碰 Program。
    std::atomic<bool> cancelled{false};
    EngineRequestState model_state        = EngineRequestState::Waiting;
    // 终态资源事务：模型侧已经结束、但 checkpoint 的 publish/release 还没提交。capture_pending 期间这条
    // 请求被排除在 decode/control 成员之外（Scheduler 会显式跳过），post_capture_state 则是 capture 完成
    // 后该回到哪个状态。
    bool capture_pending                  = false;
    EngineRequestState post_capture_state = EngineRequestState::Prefill;
    std::optional<FinishReason> terminal_reason;

    // 模型侧的 base plan，惰性建立：第一次真需要时才生成，生成后随请求走到终态。
    std::optional<BasePlan> base_plan;

    // 准入时的服务记账：一次准入授予固定份额的执行工作，decode 每采纳一个 token 消耗一份，归零意味着这次
    // 授权用完。backfill_epoch / backfill_class 是同一刻冻结的身份，供"被阻塞的队头"那套保护逻辑判断谁
    // 能当 donor——它们描述的是授权当时的关系，不能拿现在的状态反推。
    std::uint64_t remaining_service_work = 0;
    std::uint64_t backfill_epoch         = 0;
    BackfillClass backfill_class         = BackfillClass::None;

    // 真正被 prefill 评估过的提示 token 数（复用掉的不算）。PromptProgress 的前沿与 RuntimeStats::
    // computed_prefill_tokens 都从这里来。
    std::uint32_t computed_prompt_tokens = 0;

    // ---- 组 3：观测累加器 ----
    //
    // 前两个是请求自己的时间账（一个是最终交给调用方的 GenerationTimings，一个是 Host 暴露时间），后两个
    // 是执行侧留档，随结果一起交回，用来解释"这次为什么是这样跑的"。
    GenerationTimings generation_timings;
    RequestHostTiming host_timing;
    SpeculativeStats speculative_stats;
    MaterializationDiagnostics materialization_diagnostics;

    // ---- 组 4：发布面 ----
    //
    // 消费者线程只碰这一组，worker 写、wait() 搬，两边只通过这把锁与条件变量交接。要传的东西分三类：
    // 流式增量（恰好一条 stream_start 在最前，可选的 stream_progress，若干 events），终态产物（result
    // 或者 error 二选一），以及"终态已形成"这个事实本身（response_done）。
    //
    // 最后两个标记是 outstanding 名额归还协议的两半：容量要等 response_done（worker 已形成终态）与
    // consumer_released（wait 结束，或句柄被放弃）**都**成立才归还，谁后到谁负责还。这条规则是有意的——
    // 一条已经跑完却没人来取的请求，依然占着它的名额，这是调用方应该看到的容量语义。
    std::mutex mutex;
    std::condition_variable cv;
    std::optional<GenerationStart> stream_start;
    std::optional<PromptProgress> stream_progress;
    std::vector<StreamEvent> events;
    GenerationResult result;
    std::exception_ptr error;
    bool response_done     = false;
    bool consumer_released = false;
    bool capacity_released = false;
};

} // namespace ninfer::runtime
