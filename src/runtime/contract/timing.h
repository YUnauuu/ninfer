#pragma once

#include "core/nvtx.h"
#include <chrono>
#include <cstdint>
#include <optional>

// ============================================================================
// runtime/contract/timing.h —— 一次 Program 执行的耗时分解
// ============================================================================
//
// 三段互斥分解：submit_host_ns / device_wait_ns / post_host_ns（主机侧提交、等设备、主机侧
// 后处理）。主机侧的时间可以被并发掩盖，设备等待不行，混进一个总时长就再也拆不开了。
//
// 这些值由执行层实测上报，**不能**用"请求总时长"反推（总时长里还混着排队）；紧凑批处理下
// 它们适合解释单个请求的延迟构成，但**不能跨并发请求相加**。

namespace ninfer::runtime {

// One Program execution may alternate between Host submission, a blocking Device completion
// wait, and Host post-processing. The three monotonic components are returned to Engine as part of
// the execution capability; serve never infers them from total request time.
struct ExecutionTiming {
    std::uint64_t submit_host_ns = 0;
    std::uint64_t device_wait_ns = 0;
    std::uint64_t post_host_ns   = 0;

    ExecutionTiming& operator+=(ExecutionTiming other) noexcept {
        submit_host_ns += other.submit_host_ns;
        device_wait_ns += other.device_wait_ns;
        post_host_ns += other.post_host_ns;
        return *this;
    }

    // **刻意不含 device_wait_ns**：等设备的时候主机是空闲的，算进"主机活跃"会严重高估主机负载。
    [[nodiscard]] std::uint64_t host_ns() const noexcept { return submit_host_ns + post_host_ns; }

    // 执行真正占用的时间（主机活跃 + 设备等待），仍不等于请求延迟——排队不算在内。
    [[nodiscard]] std::uint64_t elapsed_ns() const noexcept { return host_ns() + device_wait_ns; }
};

// Paused 是"不属于任何一段"的空档（交还调度器、被同批其它请求阻塞），Paused 下不开 NVTX range。
enum class ExecutionTimingPhase : std::uint8_t {
    Submit,
    Wait,
    Post,
    Paused,
};

// Fixed-cost coarse recorder used only at Program phase boundaries, never in token/page/layer
// loops. It starts in Submit, permits explicit Submit -> Wait -> Post transitions, and can resume
// Submit for a later segment in the same execution unit.
// 记账方式是**区间累加**而不是"起止相减"：每次切换结算上一段再重新计时，所以同一个记录器
// 天然支持一次执行里的多段往返。
class ExecutionTimingRecorder {
public:
    using Clock = std::chrono::steady_clock;

    explicit ExecutionTimingRecorder(
        ExecutionTimingPhase initial_phase = ExecutionTimingPhase::Submit,
        ExecutionTiming* abandoned_timing  = nullptr) noexcept
        : started_(Clock::now()), phase_(initial_phase), abandoned_timing_(abandoned_timing) {
        open_range();
    }

    // 没有显式 finish() 就析构（典型是异常展开）时，已累计的时间仍会并进构造时传入的汇入口：
    // 失败路径的耗时同样有价值，tests/test_host_timing.cpp 专门钉住了这个行为。
    ~ExecutionTimingRecorder() noexcept {
        if (finished_) { return; }
        const ExecutionTiming timing = finish();
        if (abandoned_timing_ != nullptr) { *abandoned_timing_ += timing; }
    }

    ExecutionTimingRecorder(const ExecutionTimingRecorder&)            = delete;
    ExecutionTimingRecorder& operator=(const ExecutionTimingRecorder&) = delete;

    void begin_wait() noexcept { transition(ExecutionTimingPhase::Wait); }

    // 目标是 Post 而不是 Submit：拿结果之后的整理算后处理，重新下发才叫提交。
    void end_wait() noexcept { transition(ExecutionTimingPhase::Post); }

    void resume_submit() noexcept { transition(ExecutionTimingPhase::Submit); }

    void resume_post() noexcept { transition(ExecutionTimingPhase::Post); }

    void pause() noexcept { transition(ExecutionTimingPhase::Paused); }

    void include(ExecutionTiming timing) noexcept { timing_ += timing; }

    // 幂等；收尾后相位固定为 Paused、finished_ 置位，析构不再做事。
    [[nodiscard]] ExecutionTiming finish() noexcept {
        if (finished_) { return timing_; }
        accumulate(Clock::now());
        range_.reset();
        phase_    = ExecutionTimingPhase::Paused;
        finished_ = true;
        return timing_;
    }

private:
    [[nodiscard]] static nvtx::Name range_name(ExecutionTimingPhase phase) noexcept {
        switch (phase) {
        case ExecutionTimingPhase::Submit:
            return nvtx::Name::ProgramSubmit;
        case ExecutionTimingPhase::Wait:
            return nvtx::Name::DeviceWait;
        case ExecutionTimingPhase::Post:
            return nvtx::Name::ProgramPost;
        case ExecutionTimingPhase::Paused:
            break;
        }
        return nvtx::Name::ProgramSubmit;
    }

    void open_range() noexcept {
        if (phase_ != ExecutionTimingPhase::Paused) {
            range_.emplace(range_name(phase_), nvtx::Category::Runtime);
        }
    }

    // 已 finish、或切到当前相位，都是静默空操作；**必须先 accumulate 再重置 started_**，否则这一段就丢。
    void transition(ExecutionTimingPhase next) noexcept {
        if (finished_ || next == phase_) { return; }
        const Clock::time_point now = Clock::now();
        accumulate(now);
        range_.reset();
        phase_   = next;
        started_ = Clock::now();
        open_range();
    }

    void accumulate(Clock::time_point now) noexcept {
        const auto count =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - started_).count();
        const std::uint64_t elapsed = count > 0 ? static_cast<std::uint64_t>(count) : 0;
        switch (phase_) {
        case ExecutionTimingPhase::Submit:
            timing_.submit_host_ns += elapsed;
            break;
        case ExecutionTimingPhase::Wait:
            timing_.device_wait_ns += elapsed;
            break;
        case ExecutionTimingPhase::Post:
            timing_.post_host_ns += elapsed;
            break;
        case ExecutionTimingPhase::Paused:
            break;
        }
    }

    Clock::time_point started_;
    ExecutionTimingPhase phase_ = ExecutionTimingPhase::Submit;
    ExecutionTiming timing_;
    std::optional<nvtx::ScopedRange> range_;
    ExecutionTiming* abandoned_timing_ = nullptr;
    bool finished_                     = false;
};

} // namespace ninfer::runtime
