#include "options.h"
#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"
#include "product/logging/startup_log.h"
#include "product/prompt_input/prompt_input.h"
#include "product/speculative_options.h"

#include "ninfer/engine.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include <spdlog/logger.h>

// ============================================================================
// ninfer CLI —— 一次调用 = 一个请求的命令行前端
// ============================================================================
//
// 本文件把 apps/cli/options.{h,cpp}（argv → Options）和 src/product/*（产品层：日志、
// 美化格式、prompt 组装、投机参数校验）粘到 ninfer::Engine 上，并且**只调用
// include/ninfer/ 里的公共 API**——CLI 是外部使用者，不能依赖 src/ 下的运行时实现。
//
// 输出纪律（决定重定向行为）：stdout 只放 content 通道的模型产出，stderr 放 reasoning 和
// 全部统计/日志；退出码 0 = 成功（含 --help），1 = 参数错误或推理失败。

namespace {

// 这些 format_* 只是转发到 pretty_format.cpp，好让 CLI 与 serve 的显示风格一致。
std::string format_seconds(double seconds) {
    return ninfer::product::format_pretty_duration(seconds);
}

std::string format_rate(double tokens, double seconds) {
    if (tokens <= 0.0 || seconds <= 0.0) { return "n/a"; }
    return ninfer::product::format_pretty_rate(tokens / seconds, "tok");
}

// 分母为 0 是正常状态（没开投机、或一轮都没起草），要显示 n/a 而不是 100%。
std::string format_percent(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) { return "n/a"; }
    return ninfer::product::format_pretty_percent(static_cast<double>(numerator) /
                                                  static_cast<double>(denominator));
}

std::string format_bytes(std::uint64_t bytes) {
    return ninfer::product::format_pretty_bytes(bytes);
}

// weights/sequence 的占用不随请求变化，所以看 used 而不是 peak。
std::string format_arena_used(const ninfer::ArenaMemorySummary& arena) {
    return format_bytes(arena.used_bytes) + " / " + format_bytes(arena.capacity_bytes);
}

std::string format_arena_peak(const ninfer::ArenaMemorySummary& arena) {
    return format_bytes(arena.peak_used_bytes) + " / " + format_bytes(arena.capacity_bytes);
}

// 打的是 resolved_sampling() 解析后的结果（模型默认值 + CLI 覆盖），不是 CLI 里的原始值。
std::string format_sampling(const ninfer::ResolvedSamplingParameters& sampling) {
    // --greedy 把 temperature 置 0；退化成 argmax 后其它采样参数都不再生效。
    if (sampling.temperature <= 0.0F) { return "greedy (temperature 0)"; }
    std::ostringstream output;
    output << std::fixed << std::setprecision(2) << "temp=" << sampling.temperature
           << " top_p=" << sampling.top_p << " top_k=" << sampling.top_k
           << " min_p=" << sampling.min_p << " presence=" << sampling.presence_penalty
           << " freq=" << sampling.frequency_penalty << " seed=" << sampling.seed;
    return output.str();
}

std::string format_finish(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::None:
        return "none";
    case ninfer::FinishReason::OutputLimit:
        return "output-limit";
    case ninfer::FinishReason::ContextCapacity:
        return "context-capacity";
    case ninfer::FinishReason::StopToken:
        return "stop-token";
    case ninfer::FinishReason::StopString:
        return "stop-string";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

std::string format_kv_cache(ninfer::KvCacheStorage storage) {
    switch (storage) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8-group64";
    case ninfer::KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-row256";
    case ninfer::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case ninfer::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

// explicit = 按 --kv-capacity N 写死 token 数；auto = 按显存剩余量减 headroom 推算。
std::string format_kv_capacity_mode(ninfer::KvCapacityMode mode) {
    return mode == ninfer::KvCapacityMode::Automatic ? "auto" : "explicit";
}

void print_stage(std::string_view group, std::string_view detail, double seconds) {
    std::cerr << std::left << std::setw(12) << group << std::setw(26) << detail << std::right
              << std::setw(12) << format_seconds(seconds) << '\n';
}

void print_metric(std::string_view label, std::string_view value) {
    std::cerr << std::left << std::setw(12) << "summary" << std::setw(26) << label << value << '\n';
}

// StreamingSink —— 把引擎的流式回调接到终端。本程序没设 GenerationObservationOptions，
// 所以只有 start()（这里用不到）和 publish() 会真的触发，progress()/timing() 不会来。
class StreamingSink final : public ninfer::OutputSink {
public:
    void start(ninfer::GenerationStart) override {}

    void progress(ninfer::PromptProgress) override {}

    void timing(ninfer::GenerationTimingObservation) override {}

    // reasoning 走 stderr、content 走 stdout；每段都 flush，否则重定向后看不到实时输出。
    // 两个 *_seen_ / *_ends_in_newline_ 供 finish_streams() 判断要不要补换行。
    void publish(ninfer::OutputDelta delta) override {
        std::ostream& output =
            delta.channel == ninfer::OutputChannel::Reasoning ? std::cerr : std::cout;
        output << delta.text;
        output.flush();
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            reasoning_seen_ = reasoning_seen_ || !delta.text.empty();
            if (!delta.text.empty()) { reasoning_ends_in_newline_ = delta.text.back() == '\n'; }
        } else {
            content_seen_ = content_seen_ || !delta.text.empty();
            if (!delta.text.empty()) { content_ends_in_newline_ = delta.text.back() == '\n'; }
        }
    }

    // 收尾：让统计表和错误日志从干净的新行开始，又不因"什么都没输出"多打空行；reasoning
    // 在 stderr 上，不补换行会和日志粘在一起。幂等（finished_ 守卫），catch 路径也会调用。
    void finish_streams(bool successful = true) {
        if (finished_) { return; }
        finished_ = true;
        if ((successful && !content_seen_) || (content_seen_ && !content_ends_in_newline_)) {
            std::cout << '\n';
        }
        std::cout.flush();
        if (reasoning_seen_ && !reasoning_ends_in_newline_) { std::cerr << '\n'; }
    }

private:
    bool content_seen_              = false;
    bool content_ends_in_newline_   = false;
    bool reasoning_seen_            = false;
    bool reasoning_ends_in_newline_ = false;
    bool finished_                  = false;
};

void print_generation_summary(const ninfer::GenerationResult& result,
                              const ninfer::ResolvedSamplingParameters& sampling,
                              const ninfer::MemorySummary& memory) {
    print_stage("prepare", "render/preprocess", result.timings.prepare_seconds);
    print_stage("generate", "vision", result.timings.vision_seconds);
    print_stage("generate", "text prefill", result.timings.prefill_seconds);
    print_stage("generate", "decode", result.timings.decode_seconds);
    print_stage("generate", "total", result.timings.total_seconds);

    const std::size_t generated = result.generated_token_ids.size();
    // decode 速度的分母是 N-1 而不是 N：第一个 token 由 prefill 最后一步顺带产出（见
    // types.h 的 RuntimeStats::committed_decode_tokens），只有后 N-1 个 token 对应解码轮。
    const std::size_t decoded   = generated == 0 ? 0 : generated - 1;
    const double model_seconds  = result.timings.vision_seconds + result.timings.prefill_seconds +
                                 result.timings.decode_seconds;
    print_metric("sampling", format_sampling(sampling));
    print_metric("finish reason", format_finish(result.finish_reason));
    print_metric("prompt tokens", std::to_string(result.prompt.prompt_tokens));
    print_metric("reused prompt tokens", std::to_string(result.reused_prompt_tokens));
    print_metric("generated tokens", std::to_string(generated));
    if (result.thinking.configured_budget) {
        print_metric("thinking budget", std::to_string(*result.thinking.configured_budget));
        print_metric("model thinking tokens",
                     std::to_string(result.thinking.model_thinking_tokens));
        print_metric("thinking control tokens", std::to_string(result.thinking.injected_tokens));
        print_metric("thinking control", result.thinking.applied ? "applied" : "not applied");
    }
    print_metric("model elapsed", format_seconds(model_seconds));
    print_metric("prefill speed", format_rate(static_cast<double>(result.prompt.prompt_tokens),
                                              result.timings.prefill_seconds));
    print_metric("decode speed",
                 format_rate(static_cast<double>(decoded), result.timings.decode_seconds));
    print_metric("throughput (overall)",
                 format_rate(static_cast<double>(generated), model_seconds));

    // "计划占用" = weights 容量 + runtime reservation（引擎规划显存的口径）；workspace 峰值不算在内。
    const std::uint64_t reserved = static_cast<std::uint64_t>(memory.weights.capacity_bytes) +
                                   memory.runtime_reservation_bytes;
    print_metric("device", std::to_string(memory.device));
    print_metric("max context", std::to_string(memory.max_context));
    print_metric("KV capacity policy", format_kv_capacity_mode(memory.kv_capacity_mode));
    print_metric("KV capacity", std::to_string(memory.kv_capacity));
    print_metric("KV page groups", std::to_string(memory.kv_capacity_page_groups) + " / " +
                                       std::to_string(memory.kv_capacity_max_page_groups));
    print_metric("gpu weights used", format_arena_used(memory.weights));
    print_metric("gpu sequence used", format_arena_used(memory.sequence));
    print_metric("kv cache dtype", format_kv_cache(memory.kv_cache));
    print_metric("kv cache payload", format_bytes(memory.kv_payload_bytes));
    print_metric("gpu workspace peak", format_arena_peak(memory.workspace));
    print_metric("runtime reservation", format_bytes(memory.runtime_reservation_bytes));
    print_metric("free after weights", format_bytes(memory.available_after_weights_bytes));
    print_metric("free after startup", format_bytes(memory.available_after_startup_bytes));
    print_metric("KV capacity headroom", format_bytes(memory.kv_capacity_headroom_bytes));
    print_metric("planned slack", format_bytes(memory.planned_slack_bytes));
    print_metric("CUDA Graph allowance", format_bytes(memory.cuda_graph_allowance_bytes));
    print_metric("planned device total", format_bytes(reserved));

    // 指标名带 backend 前缀；acceptance length ≤ 1 说明投机白干（每轮只推进一个 token）。
    const ninfer::SpeculativeStats& speculative = result.speculative;
    if (speculative.enabled) {
        const std::string backend = ninfer::product::speculative_backend_name(speculative.backend);
        print_metric(backend + " draft window", std::to_string(speculative.draft_window));
        print_metric(backend + " rounds", std::to_string(speculative.rounds));
        print_metric(backend + " fallback steps", std::to_string(speculative.fallback_steps));
        print_metric(backend + " drafted tokens", std::to_string(speculative.drafted_tokens));
        print_metric(backend + " accepted tokens", std::to_string(speculative.accepted_tokens));
        print_metric(backend + " acceptance rate",
                     format_percent(speculative.accepted_tokens, speculative.drafted_tokens));
        if (speculative.rounds != 0) {
            std::ostringstream length;
            length << std::fixed << std::setprecision(2)
                   << 1.0 + static_cast<double>(speculative.accepted_tokens) /
                                static_cast<double>(speculative.rounds)
                   << " tok/round";
            print_metric(backend + " acceptance length", length.str());
        }
        if (!speculative.accepted_per_position.empty()) {
            std::ostringstream positions;
            for (std::size_t i = 0; i < speculative.accepted_per_position.size(); ++i) {
                if (i != 0) { positions << ','; }
                positions << speculative.accepted_per_position[i];
            }
            print_metric(backend + " accepted by pos", positions.str());
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    ninfer::cli::Options cli;
    try {
        cli = ninfer::cli::parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        std::cerr << ninfer::cli::usage_text(argv[0]);
        return 1;
    }
    if (cli.help_requested) {
        std::cout << ninfer::cli::usage_text(argv[0]);
        return 0;
    }

    // logging 必须先于 logger / startup_log 构造并活到最后：后两者持有的是弱引用语义的句柄。
    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer",
         .level        = cli.log_level,
         .presentation = ninfer::product::LogPresentation::Tool});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);

    try {

        // --prompt 给文本（包成一条 user 消息），--messages 给 JSON 文件（可含 messages/tools/
        // 图片视频 part）；两者都不给会由 parse_options 提前拦下。
        ninfer::PromptInput input =
            cli.messages_path.empty()
                ? ninfer::product::prompt_from_text(cli.prompt, cli.enable_thinking)
                : ninfer::product::prompt_from_messages(cli.messages_path, cli.enable_thinking,
                                                        cli.enable_vision);
        input.options.reasoning_effort = cli.reasoning_effort;

        // sampling 里没填的字段是 nullopt，含义是"用模型/mode 的默认值"；真正的解析发生在
        // submit()，结果可以从 GenerationHandle::resolved_sampling() 读回。
        ninfer::RequestOptions request;
        request.execution.sampling                = cli.sampling;
        request.execution.requested_output_tokens = cli.max_new;
        request.execution.thinking.budget         = cli.thinking_budget;
        request.stop.token_ids                    = cli.stop_token_ids;
        request.stop.strings                      = cli.stop_strings;
        request.output.raw                        = cli.raw_output;

        ninfer::EngineOptions engine_options;
        engine_options.artifact_path      = cli.artifact_path;
        engine_options.chat_template_path = cli.chat_template_path;
        engine_options.device             = cli.device;
        engine_options.max_context        = cli.max_context;
        engine_options.kv_capacity        = cli.kv_capacity;
        engine_options.prefill_chunk      = cli.prefill_chunk;
        engine_options.kv_cache           = cli.kv_cache;
        engine_options.speculative        = cli.speculative;
        engine_options.enable_vision      = cli.enable_vision;
        engine_options.use_cuda_graph     = cli.use_cuda_graph;
        // One CLI invocation owns exactly one request, so retained cross-request context has no
        // consumer and must not reserve an extra Device StateImage or run terminal capture.
        engine_options.context_cache.enabled                = false;
        engine_options.context_cache.host_state_slots       = 0;
        engine_options.context_cache.host_kv_capacity_bytes = 0;
        engine_options.startup_observer                     = startup_log.observer();

        ninfer::Engine engine(std::move(engine_options));
        startup_log.engine_ready(engine.load_summary());
        engine.reset_memory_peaks();

        ninfer::PreparedPrompt prompt = engine.prepare(std::move(input));

        // 用 submit + wait 而不是 generate()：wait 之前就要拿到 resolved_sampling()，这样
        // 即使 wait 抛异常，统计里也能显示本次实际生效的采样参数。
        StreamingSink sink;
        ninfer::GenerationHandle generation = engine.submit(std::move(prompt), std::move(request),
                                                            ninfer::OutputConsumerMode::Streaming);
        const ninfer::ResolvedSamplingParameters sampling = generation.resolved_sampling();
        ninfer::GenerationResult result;
        try {
            result = generation.wait(&sink);
            sink.finish_streams();
        } catch (...) {
            sink.finish_streams(false);
            throw;
        }

        std::cerr << "phase       detail                      elapsed/progress\n";
        if (cli.print_token_ids) {
            // --print-token-ids：便于逐 token 排查分叉；注意这一行含引擎注入的控制 token
            // （例如 --thinking-budget 预算耗尽时追加的提前关闭指引）。
            std::cerr << std::left << std::setw(12) << "tokens" << std::setw(26) << "generated ids";
            for (std::size_t i = 0; i < result.generated_token_ids.size(); ++i) {
                if (i != 0) { std::cerr << ' '; }
                std::cerr << result.generated_token_ids[i];
            }
            std::cerr << '\n';
        }
        print_generation_summary(result, sampling, engine.memory_summary());
        return 0;
    } catch (const std::exception& error) {
        // format_pretty_text 把异常信息压成一行——多行异常会破坏日志的逐行结构。
        logger->error("{}", ninfer::product::format_pretty_text(error.what()));
        return 1;
    }
}
