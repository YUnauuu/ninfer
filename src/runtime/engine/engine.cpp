#include "ninfer/engine.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/startup.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/request.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/engine_core.h"
#include "runtime/engine/model_instance.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

// ============================================================================
// runtime/engine/engine.cpp —— <ninfer/engine.h> 的实现
// ============================================================================
//
// 这个文件本身很薄：真正的引擎在 engine_core.h（模型无关的调度/提交/资源逻辑）和模型侧。
// 这里只做四件事——把 EngineOptions 落成"设备 + 模型 + 执行核"、定义两个 PIMPL 的身体、
// 把用户填的参数解析成具体值、在生成核与打分核之间分发。
//
// 全文件守住一条边界：Engine 是模型无关的。唯一一次提到具体模型类型是
// PreparedPrompt::Impl 的 value 字段——PIMPL 的身体必须按值持有那个产物，而产物类型
// 只有模型侧知道。
//
// 抛异常时按"谁该负责"分三类：
//   std::invalid_argument —— 参数不对（budget 为 0、sink 与 consumer mode 不匹配、打分区间越界）
//   RequestError          —— 一次正常的调度/容量结果（队列满、引擎不可用、提示超长）
//   std::logic_error      —— 内部不变量被破坏，是代码 bug（例如 Frontend 放行了超容量提示）
//
// core 用 variant 而不是基类：两个核没有公共基类，也不是同一种机器（生成核并发服务
// 最多 8 个请求，打分核连请求队列都没有）。monostate 表示"核已被拆毁 / 从未建立"。

namespace ninfer {
namespace {

// 包一层是为了让它成为一个具名的启动阶段：StartupPhaseScope 若没等到 complete() 就析构，
// 会发一个 Failed 事件出来。
DeviceContext initialize_device(const EngineOptions& options) {
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
    DeviceContext device(options.device);
    phase.complete();
    return device;
}

// 用户意图 → 运行时指令的唯一收敛点，也是唯一还能看见用户原始填写值的地方。
runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    // budget 是 optional：未填表示不做预算控制，显式填 0 才是非法配置。
    // 上界不在这里查——它取决于 prompt 长度，要到 EngineCore::submit 才算得出来。
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    runtime::ResolvedRequestOptions resolved;
    // 全函数唯一真正做计算的一行，其余字段都是搬运。
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.thinking                = options.execution.thinking;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    return resolved;
}

// 两个调用点共用的错误文案。
std::string context_capacity_error(std::size_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

// sampling_mode 的默认值 Thinking 是承重的：prepare_tokens() 构造它时没有传这个参数。
class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         models::qwen3_5::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    models::qwen3_5::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

// 空对象返回全默认值而不是抛异常：这两个是观测接口，调用方不该为了读一个摘要写 try/catch。
const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

// GenerationHandle 的身体：一次类型擦除。
//
// 需要擦除，是因为句柄要装两种毫无共同点的东西：EngineCore::Submission（模板的嵌套类型，
// 公共头文件不能提到它），以及 submit() 里那个局部结构体 ImmediateSubmission（背后没有引擎）。
// 两者唯一的契约就是都有一个 wait()，Concept 把这个契约固化成一个虚函数。
class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        // 声明顺序是承重的：成员逆序析构，所以 submission_ 先销毁。它的析构会走到
        // EngineCore::abandon_request()，而那里碰的是 Engine::Impl 内部的对象——
        // 也就是说 keep_alive_ 必须活到它之后。调换这两行就是一个悬垂指针。
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    // 采样参数按值存在这里：请求受理那一刻就冻住了，而且句柄可能活得比 Engine 长。
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

// 先把状态整体搬走再等：这样重复调用会干净地撞上 "is empty"，而且即使 wait 抛异常，
// 句柄也已经永久失效。局部 impl 在返回时析构，那时 Submission 内部的两个指针已被置空，
// 所以是空操作——"销毁句柄 = 取消请求"只对从未 wait 过的句柄成立。
GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

class Engine::Impl {
public:
    using GenerationCore = runtime::EngineCore<runtime::ModelInstance>;
    using ScoringCore    = runtime::CausalScoreCore<runtime::ModelInstance>;
    using Core =
        std::variant<std::monostate, std::unique_ptr<GenerationCore>, std::unique_ptr<ScoringCore>>;

    // 初始化顺序 = 声明顺序：options 必须最先，因为 device 的初始化要读它。
    // 存下来的是归一化之后的值，Engine::options() 返回的就是它。
    explicit Impl(EngineOptions engine_options)
        : options(runtime::normalize_engine_options(std::move(engine_options))),
          device(initialize_device(options)) {
        // 这个 range 一直覆盖到构造函数末尾，含建立执行核、拉起 worker 线程。
        nvtx::ScopedRange load_range(nvtx::Name::EngineLoad, nvtx::Category::Runtime);
        auto constructed  = runtime::construct_model(options, device);
        // construct_model 内部自己会发 ArtifactInspect / TargetPlan / FrontendInitialize /
        // TargetFinalize / ProgramInitialize 等更细的阶段事件，所以观察者看到的是嵌套两层。
        active            = std::move(constructed.instance);
        load              = std::move(constructed.load);
        // 读一次就存下来：之后 frontend 会被多线程访问，而 Engine::sampling_defaults()
        // 本来就要返回一份快照。
        sampling_defaults = active->frontend.sampling_defaults();
        StartupPhaseScope finalize_phase(options.startup_observer, StartupPhase::EngineFinalize);
        if (options.purpose == EnginePurpose::CausalScoring) {
            core = std::make_unique<ScoringCore>(*active, device);
        } else {
            core = std::make_unique<GenerationCore>(*active, device, options,
                                                    std::move(constructed.context_cost));
        }
        // 两个核的构造函数都会拉起 worker 并同步等它绑好设备，所以走到这里引擎已可用。
        finalize_phase.complete();
    }

    // 三步，顺序不能变：
    //   1) 设备上下文是线程私有的，而析构可能发生在从未碰过 CUDA 的线程上，先绑一次。
    //   2) 主动拆核换成 monostate。核的析构会 join worker；若留给成员析构去做（那发生在
    //      函数体之后），顺序就变成"先同步设备、再等 worker 停"，worker 可能在同步之后
    //      还在下发 kernel。这一行就是用来把"等 worker 退出"钉在"同步设备"前面的。
    //   3) 冲刷残留工作。析构不能抛，所以咽掉异常。
    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        core.emplace<std::monostate>();
        try {
            device.synchronize();
        } catch (...) {}
    }

    // core 引用 active，所以 active 必须比 core 活得久 —— 即声明在 core 之前。
    EngineOptions options;
    DeviceContext device;
    std::unique_ptr<runtime::ModelInstance> active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    Core core;
};

// observer 必须在 options 被移动之前拷出来，否则回调就空了。
// Impl 构造抛异常时 startup_phase 会补发一个 Failed 事件，调用方因此能确定"引擎没起来"。
Engine::Engine(EngineOptions options) {
    StartupObserver startup_observer = options.startup_observer;
    StartupPhaseScope startup_phase(startup_observer, StartupPhase::EngineStartup);
    impl_ = std::make_shared<Impl>(std::move(options));
    startup_phase.complete();
}

// Impl 的析构承担全部收尾，而 shared_ptr 只在最后一个引用消失时才触发它——
// 所以还有 GenerationHandle 活着时，Engine 析构不会打断正在跑的请求。
Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime);
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    auto prepared      = impl_->active->frontend.prepare(std::move(input), control);
    PromptSummary info = prepared.summary();
    // 采样模式在这里定，而且只定一次：starts_in_reasoning 是渲染之后才有的属性，
    // 而定下来就不能再变——resolve 出来的参数会随句柄交给调用方。
    const SamplingMode sampling_mode =
        info.starts_in_reasoning ? SamplingMode::Thinking : SamplingMode::NonThinking;
    // 这里抛 logic_error 而不是 RequestError：Frontend 知道 max_context，本该自己拦住。
    // 走到这里还超长说明它没守住契约，那是 bug，不是可重试的容量结果。
    if (info.prompt_tokens > impl_->active->capacity) {
        throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
    }
    const PromptPreparationStats preparation = prepared.preparation_stats();
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(info, preparation, sampling_mode,
                                                                 std::move(prepared)));
}

// 与 prepare 的差别集中在**校验的种类**上：token 是调用方直接递进来的，Engine 当场就能
// 数清，所以这里超容量是 RequestError（合法的用户错误，调用方按 kind 处理）；
// 下面那次用 Frontend 报出的长度复查，仍然是 logic_error。
PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime,
                                    static_cast<std::uint64_t>(token_ids.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (token_ids.size() > impl_->active->capacity) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           context_capacity_error(token_ids.size(), impl_->active->capacity));
    }
    auto prepared =
        impl_->active->frontend.prepare_tokens(std::move(token_ids), allow_prefix_identity);
    PromptSummary info = prepared.summary();
    if (info.prompt_tokens > impl_->active->capacity) {
        throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
    }
    const PromptPreparationStats preparation = prepared.preparation_stats();
    // 采样模式硬编码 Thinking：这条路径不渲染模板，没有可靠的 reasoning 信号。
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
        info, preparation, SamplingMode::Thinking, std::move(prepared)));
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.tokenize_text(text);
}

// 与 submit 不同，它不产生请求、不进队列、不进 RuntimeStats，而是直接打到那个单任务
// 同步 worker 上（打分用途的 max_concurrency 被归一化成 1，所以不会有人插队）。
std::vector<float> Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target) {
    nvtx::ScopedRange score_range(nvtx::Name::Score, nvtx::Category::Scoring,
                                  static_cast<std::uint64_t>(tokens.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::CausalScoring) {
        throw std::logic_error("score_tokens requires a CausalScoring Engine");
    }
    if (tokens.size() < 2 || tokens.size() > impl_->options.max_context) {
        throw std::invalid_argument("score_tokens token count must be in [2,max_context]");
    }
    if (first_target == 0 || first_target >= tokens.size()) {
        throw std::invalid_argument("score_tokens first_target must be in [1,token_count-1]");
    }
    // 第二个参数传 false 不能改：allow_prefix_identity = false 保证打分**与缓存状态无关**。
    // 否则冷跑和热跑会因复用长度不同导致浮点累加顺序漂移，评测就不可复现了。
    PreparedPrompt prompt      = prepare_tokens(std::move(tokens), false);
    // 期望长度取自 Frontend 报出的长度而不是入参 size——prepare_tokens 可能规范化 token 序列。
    const std::size_t expected = prompt.summary().prompt_tokens - first_target;
    std::vector<float> result  = std::visit(
        [&](auto& core) -> std::vector<float> {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoringCore>>) {
                return core->score(std::move(prompt.impl_->value), first_target);
            } else {
                throw std::logic_error("Engine scoring core is unavailable");
            }
        },
        impl_->core);
    // "Program 报告的是声称，Engine 独立复核"：少了这一句，返回错位数组的内核会把错误的
    // 概率一路喂进评测指标里，且无声无息。
    if (result.size() != expected) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    return result;
}

// 不检查容量：它只回答"有多少个 token"，没义务也没法阻止调用方拿着超长输入来数。
std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.count_tokens(std::move(input), control);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

// 校验全部同步完成：公开契约是 submit 返回时请求已经是队列成员了，队列满 / 引擎不可用 /
// 排队期限已过都必须在返回句柄之前抛出来，不能拖到 wait()。
GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                GenerationObservationOptions observation,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("submit requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    // 开 live_timings 就自动带上 phase_timings（前者蕴含后者）。这里改的是按值传进来的副本。
    if (observation.live_timings) { observation.phase_timings = true; }
    // 但 live_timings / prompt_progress 是增量推流，需要一个 sink 才有地方投递；Aggregate
    // 模式下会被静默丢弃，所以宁可在这里拒绝。phase_timings 不在此列——它能随最终结果交付。
    if (consumer_mode != OutputConsumerMode::Streaming &&
        (observation.live_timings || observation.prompt_progress)) {
        throw std::invalid_argument("live generation observations require a Streaming consumer");
    }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    // 先拷一份再往下：resolved_options 随后会被 move 进执行核，而句柄要自己留一份——
    // resolved_sampling() 读的必须是受理那一刻冻结的值。prompt_summary 同理，
    // 因为 prompt 之后会被消费掉。
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    // 这里用 RequestError 而不是 logic_error：提示已是调用方手里的成品，可能来自另一个
    // 能力不同的 Engine，也可能 prepare 之后过了很久才提交。Engine 当场能算出它超长，
    // 所以这是可预期的用户错误。
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    // 要求输出 0 个 token：不建立请求，直接给一个现成结果。走正常路径只会白白占掉一个
    // 并发槽和若干调度轮次，最后还是产出空结果。
    if (resolved_options.execution.requested_output_tokens == 0) {
        // 局部结构体，靠鸭子类型混过 GenerationHandle 的类型擦除——这正是擦除存在的理由。
        struct ImmediateSubmission {
            GenerationResult result;
            OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
                // 与 EngineCore::Submission::wait 同形：sink 与 consumer mode 的匹配是公开
                // 契约的一部分，捷径不能豁免，否则调用方会以为规则只在"真的跑了"时生效。
                const bool streaming = consumer_mode == OutputConsumerMode::Streaming;
                if (streaming != (sink != nullptr)) {
                    throw std::invalid_argument(
                        "GenerationHandle wait sink does not match its submitted consumer mode");
                }
                // 取消优先于 OutputLimit，而且同样不是异常，只换一个 finish_reason。
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
        } immediate{.consumer_mode = consumer_mode};

        // 一份"诚实的空结果"：摘要照实回填（Streaming 的 GenerationStart 从它来，必须与
        // 真正跑起来时形状一致）；OutputLimit 表示确实结束了；total == prepare，
        // 因为这条路径上只做了准备这一件事。
        immediate.result.prompt                     = prompt_summary;
        immediate.result.finish_reason              = FinishReason::OutputLimit;
        immediate.result.thinking.configured_budget = resolved_options.execution.thinking.budget;
        immediate.result.timings.prepare_seconds    = prepare_seconds;
        immediate.result.timings.total_seconds      = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    // 必须用 if constexpr 链：ScoringCore 根本没有 submit()，普通 if 会直接编译失败。
    // monostate 与 ScoringCore 分开报错，因为两者的排查方向完全不同。
    // pending_deadline 原样转发——传 {} 表示"没有特别指定"，"用 pending_timeout_ms 折算成
    // 绝对时刻"这条策略由核来执行，默认值的解释权在那边。
    return std::visit(
        [&](auto& core) -> GenerationHandle {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoringCore>>) {
                throw std::logic_error("Engine generation core is unavailable");
            } else {
                auto submission = core->submit(std::move(prompt.impl_->value), prompt_summary,
                                               prepare_seconds, std::move(resolved_options),
                                               consumer_mode, observation, pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling));
            }
        },
        impl_->core);
}

// submit + wait。consumer mode 由 sink 是否为空推导，省得调用方再传一次、
// 也就不可能出现"声明 Aggregate 却传了 sink"这种自相矛盾的调用。
// 临时句柄在 wait 返回之后才析构，那时 Submission 已经空了，不会触发取消。
GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode, {})
        .wait(sink, cancellation);
}

// 唯一返回引用的接口：直接把 Impl 里那份归一化后的 options 交出去，不是快照。
const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

// 加载期快照，构造之后不再变。
LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

// 必须问核：Program 持有的 arena 只有经由核才拿得到。
MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> MemorySummary {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->memory_summary();
            }
        },
        impl_->core);
}

// 走的是前端而不是执行核——媒体预处理是纯主机侧的事，缓存挂在前端上。
MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.media_cache_summary();
}

// 读的是核里**已发布**的快照（在若干边界点更新），所以读统计的开销不落在执行热路径上。
// ScoringCore::runtime_stats() 恒返回空，那是约定而不是错误。
RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> RuntimeStats {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->runtime_stats();
            }
        },
        impl_->core);
}

// 与其它观测接口不同，这个不抛：空引擎和 monostate 都返回 false，因为它会被放进健康检查
// 循环里。判据由核给出——生成核是 !stopping_ && !failed_（failed_ 是 worker 因不可恢复
// 错误整体退出后的终态），打分核只有 !stopping_。
bool Engine::is_available() const {
    if (impl_ == nullptr) { return false; }
    return std::visit(
        [](const auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return false;
            } else {
                return core != nullptr && core->is_available();
            }
        },
        impl_->core);
}

// 唯一的 noexcept 写操作，下面几处反常都是它带来的：空引擎直接返回而不是抛；
// monostate 分支是个**空操作**（判断写成否定式），因为"核不存在"对一次清峰值不算错误——
// 这与其它访问接口"monostate ⇒ 抛"正好相反，是有意为之。
void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                core->reset_memory_peaks();
            }
        },
        impl_->core);
}

} // namespace ninfer
