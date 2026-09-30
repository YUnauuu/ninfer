#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <memory>
#include <string_view>
#include <vector>

// ============================================================================
// ninfer/engine.h —— NInfer 的公共推理接口
// ============================================================================
//
// 这是使用者唯一需要包含的头文件，也是"产品语义"与"执行实现"的分界线：这里只声明一次请求长
// 什么样、怎么交出去、结果怎么拿回来；怎么调度、怎么算，全在 src/ 里。它在分层里的位置是
//
//     Gateway（协议适配） → Frontend（提示渲染） → **Engine（本文件）** → Program（层与 KV）
//
// Engine 属于"请求控制面"：它理解请求、容量、预算、finish reason 和输出发布，但不解释任何一层
// transformer、KV plane 或 allocator。src/runtime/engine/engine_core.h 里的注释只讲执行核内部
// 怎么组织，公开语义以本文件为准。
//
// 一次请求的方向（三段式，每段的边界都很清楚）：
//
//   PromptInput ─prepare()─▶ PreparedPrompt ─submit()─▶ GenerationHandle ─wait()─▶ GenerationResult
//   对话消息 +           Host 侧成品：模板已渲染、   已在队列里、名额已占、   结果、finish reason、
//   模板/缓存选项         媒体已预处理、token 已定     发布次序已分配          异常、各类统计
//
//   * prepare —— 纯 Host 侧准备，不碰 Device，可能很慢（媒体解码），失败发生在受理之前。
//   * submit  —— **同步**完成"能不能受理"的判定；返回即代表请求已经排队，此后由 worker 独立推进。
//   * wait    —— 消费者侧动作，是结果与异常唯一出现的地方；它与 GPU 执行并行，何时调用不影响执行。
//   * generate() = submit + wait，是上面这条链的语法糖。
//
// 同一时刻只有一条链是活的：EnginePurpose 决定哪半边 API 可用，混用会在调用点上直接抛
// logic_error，而不是悄悄降级——
//   * Generation    —— prepare()/prepare_tokens() 加 submit()/generate()，可并发服务多个请求。
//   * CausalScoring —— 只有 score_tokens()，单任务同步，用于困惑度一类评测。
//
// 异常按"谁该负责"分三类，这是读本文件时的通用读法，也是调用方该按 kind 分支的地方：
//   * std::invalid_argument —— 参数写错了（budget 为 0、sink 与 consumer mode 不匹配、打分区间越界）。
//   * RequestError          —— 一次正常的调度/容量结果（队列满、超长、排队超时、取消），带 kind；
//                              它不是 bug，也不表示引擎坏了。
//   * std::logic_error      —— 内部不变量被破坏，是代码 bug（例如 Frontend 放行了超容量提示）。
//
// 两条使用上的约定：
//   * Engine 是长生命周期对象：构造一次、服务很多请求。构造开销极大（加载权重、准备 CUDA graph），
//     不要在请求路径上反复新建；能力也在构造期定死，运行期无法增补。
//   * 公开方法可被多个调用线程使用（Engine 内部只有一个 worker 是状态变更的 owner）；请求的生存期
//     与取消语义由下面两个句柄类表达。

namespace ninfer {

// 一次提示的 Host 侧成品：模板已渲染、媒体已预处理、token 已定、采样模式已判定，剩下的只是排上队
// 去算。prepare 阶段的全部失败（超长、媒体非法、模板或参数错误）都落在这里，也就是都发生在 Device
// 之前。
//
// 它是可移动、不可复制的独占对象：内部按值持有渲染后的具体产物，复制没有意义。空对象（默认构造，
// 或被 submit 消费之后）为 false，此时读摘要返回全默认值而不是抛异常——观测接口不该逼调用方写
// try/catch。
//
// 关键点：它描述"这次要算什么"，不再绑在某个 Engine 上。A Engine prepare 出来的成品可以交给 B
// Engine 去 submit（只要 B 的 max_context 容得下），也可能在 prepare 之后很久才被提交。
class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

// 一次已受理请求的句柄。submit() 返回它的那一刻，请求已经在队列里、并发名额已经占上了。
//
// 它只有两种归宿，这是本文件最承重的一条约定：
//   * wait() —— 唯一的出口，阻塞到终态，把结果或异常交出来；消费过一次就永久失效。
//   * 析构   —— 等价于**取消**这次请求。"不想要了就丢掉"因此是一个合法且干净的中断方式，代价是
//             失去结果；取消只是置标记，真正的收敛发生在 worker 的稳定边界上，不会撕裂已下发的
//             GPU 工作。
//
// wait() 是消费者侧的动作，与 GPU 执行并行：submit() 之后请求自己往下跑，句柄在谁手里、什么时候
// wait，都不影响执行——所以想要流式输出就得选 Streaming 模式并交出 sink。
//
// 句柄可以活得比 Engine 长：它内部持有 Engine 的共享所有权，Engine 析构不会打断还在跑的请求。
class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;

    // 受理那一刻冻结的采样参数，句柄自己带一份副本，所以 wait() 之前就读得到——UI 要显示"本次
    // 实际生效了什么参数"，即使随后 wait() 抛异常也拿得回来。
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    // 阻塞到终态。sink 必须与 submit 时声明的 consumer mode 一致（Streaming 必须给 sink，Aggregate
    // 必须不给），不一致直接抛 invalid_argument，这条规则不走捷径。下面几种情况都会在这里现形：
    // worker 侧存下的错误、cancellation 求值抛的异常、以及 sink 回调自己抛的异常——后者会顺带把本次
    // 请求取消掉。已受理之后的取消不是异常，只是把 finish_reason 置成 Cancelled。
    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

// 引擎：一个进程里只建一次的重对象，拥有设备上下文、模型实例与请求执行核。
//
// 构造是同步的重活（加载权重、准备 CUDA graph、拉起 worker），过程中通过 EngineOptions::
// startup_observer 汇报阶段事件；构造失败时观察者会收到一个 Failed 事件。本文件里的所有能力都在
// 构造期定死——max_context、KV 容量、并发度、投机后端、是否开 vision——后续请求无法启用启动时
// 省略的能力。
//
// 除三段式的主 API 外，剩下的是一组只读快照（load / memory / media / runtime 统计）。它们读的都是
// "已发布"的值而不是现算：好处是读统计不落在执行热路径上，代价是数值会略微滞后。
class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    // 主路径的第一步：走 chat template 渲染、多模态预处理、tokenize、前缀规划，产出可提交的成品。
    // 不碰 Device，但可能很慢（媒体解码），所以接受 PreparationControl 的 deadline 与 cancellation。
    // 采样模式（thinking / non-thinking）在这里定，而且只定一次：它决定之后用哪套模型默认采样值。
    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // 绕过模板与媒体，直接给 token。与 prepare 的差别集中在**校验种类**上：token 是调用方递进来的，
    // Engine 当场就能数清，所以超容量是 RequestError（合法的用户错误），而不是内部契约被破坏。
    // 保留这条路径是为了可复现的正确性与性能测量——同一串 token 可以反复跑出同样的序列。
    // allow_prefix_identity 置 false 会关闭"这段提示可被识别为复用身份"，打分用的正是这一点。
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    // 直接调用产物自带 tokenizer 编码原始文本：不加 chat template，也不隐式补任何特殊 token。
    // 它不产生可提交的提示，主要用于数 token 和排查分词。
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;

    // 因果打分：返回 log p(tokens[i] | tokens[0..i))，i 取 [first_target, tokens.size())。
    // 只属于 CausalScoring 用途：它不产生请求、不进队列、不计入 RuntimeStats，而是直接打到那个单任务
    // 同步 worker 上（该用途下并发被归一化为 1，不会有人插队）。内部强制不复用缓存前缀，因此冷跑与
    // 热跑结果一致，评测可复现。
    [[nodiscard]] std::vector<float> score_tokens(std::vector<TokenId> tokens,
                                                  std::uint32_t first_target);

    // 只回答"这段提示有多少 token"。同样要走渲染与媒体统计，但既不建请求也不查容量——Engine 没有义务
    // 也没法阻止调用方拿超长输入来数。
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;

    // 模型自带的采样默认值（thinking / non-thinking 两套），构造期读一次后冻结成快照。调用方未指定某
    // 个采样字段时用的就是它；seed 刻意不在其中——种子的执行选择权归调用方。
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // 提交并**同步**确立队列成员身份：返回时请求已经进 FIFO、并发名额与发布次序都已分配，所以"队列满 /
    // 引擎不可用 / 排队期限已过"都在这里抛，不会拖到 wait()。准入（分 lane、取资源）与执行留给 worker。
    //
    // 三个参数的分工：
    //   * consumer_mode 在提交时冻结，wait() 里的 sink 必须与它一致。Streaming 要求非空 sink，用来接收
    //     一个精确的 GenerationStart、增量 delta 与进度；Aggregate 要求空 sink，最终结果随 wait() 返回。
    //   * observation 只影响"怎么发布"这一件事，绝不改变模型执行、输出语义、调度或缓存选择；实时类
    //     观测（live_timings / prompt_progress）需要 Streaming 消费者，Aggregate 下会被拒绝而不是静默丢弃。
    //   * pending_deadline 是**排队期限的绝对时刻**（不是整个请求的期限），约束的只是这一段等待。传空
    //     表示未特别指定，由 EngineOptions::pending_timeout_ms 折算；已过期就直接以 QueueTimeout 结束。
    //
    // 要求输出 0 个 token 时不会建立请求：直接给一份诚实的空结果（finish_reason = OutputLimit），省掉
    // 一个并发槽和若干调度轮次。销毁未 wait 过的句柄即取消，见 GenerationHandle。
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           OutputConsumerMode consumer_mode                       = OutputConsumerMode::Aggregate,
           GenerationObservationOptions observation               = {},
           std::chrono::steady_clock::time_point pending_deadline = {});

    // submit + wait 的便捷形态，consumer mode 由 sink 是否为空推导（省得调用方再声明一次、也就不会出现
    // "声明 Aggregate 却传了 sink"这种自相矛盾）。一次调用拿完整结果时用它；要并发、或想提前读
    // resolved_sampling()、或想自己掌控取消，就用 submit + wait。
    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    // 归一化之后（optional 已填成具体值）的 options，返回的是引用而不是快照——也就是它随 Engine 析构
    // 一起失效。
    [[nodiscard]] const EngineOptions& options() const;

    // 加载期快照：产权重格式、耗时、上传字节数、上下文代价预设来源等，构造之后不再变。
    [[nodiscard]] LoadSummary load_summary() const;

    // 当前物理内存布局与占用：各 arena、KV 容量与已用、Host pinned 容量。它问的是 Program 持有的
    // arena，所以只能经由执行核拿到。
    [[nodiscard]] MemorySummary memory_summary() const;

    // 单调执行计数器 + 边界一致的当前 gauge + 若干显式的"上次决定"观测。调用方对两次快照相减即得区间值。
    // 打分用途下恒为空，那是约定而不是错误。
    [[nodiscard]] RuntimeStats runtime_stats() const;

    // 媒体预处理缓存，挂在 Frontend 上（媒体预处理是纯 Host 侧的事），不进执行核。
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;

    // 健康检查用的判据，唯一不抛的查询接口（空引擎与"核不存在"都返回 false，因为它会被放进循环里）。
    // false 的含义是"这个 Engine 已经报废"而不是"现在忙"：worker 因不可恢复的内部错误整体退出后，
    // 这里会永久为 false。
    [[nodiscard]] bool is_available() const;

    // 清零内存峰值计数。典型用法是启动完成后调一次，让加载期峰值不污染稳态观测。唯一的 noexcept 写操作。
    void reset_memory_peaks() noexcept;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer
