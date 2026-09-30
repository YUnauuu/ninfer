#pragma once

// ============================================================================
// runtime/engine/engine_core.h —— 模型无关的请求执行核（生成侧）
// ============================================================================
//
// EngineCore 是 <ninfer/engine.h> 里 Engine 的生成侧身体，也是"请求控制面"的实现：它拥有
// outstanding 容量、FIFO 等待队列与排队期限、请求记录与 active lane、取消与 response 事件、
// Scheduler 与 ResourceManager，并编排 admission / prefill / decode / control / capture /
// terminal 的先后顺序。它理解请求、预算、finish reason 和可发布输出，但不解释 transformer
// 层、KV plane 或 allocator——那些属于 Program。
//
// 它按模板参数 Instance 泛化，靠 Instance::ModelContract 把具体模型的类型绑进来（见类内开头的
// using 列表），因此调度、提交、资源与发布这套逻辑对所有模型只有一份。
//
// 三条贯穿全文件的结构性约定：
//
//   * **单一线程拥有全部变更**：worker 线程是唯一的 mutation owner，握着 execution_mutex_ 逐个
//     执行单位推进；调用方线程只做三件事——submit 入队、在 wait() 里搬运已发布的事件、读快照。
//   * **一次执行单位 = 一次事务边界**：一个 prefill 步、一轮 decode、一批 control 就是一次
//     Program 事务，要么 commit 并采用其结果，要么整体 abort。取消与失败都只在这个边界上生效，
//     不会撕裂已经下发的 GPU 工作。
//   * **不可恢复的错误使整个引擎失败**：从 worker 里逃出的异常意味着共享物理状态已经无法安全
//     解释，于是所有请求一起失败、worker 永久退出、is_available() 从此为 false。内部不变量错误
//     不允许降级成 cache miss、等待或重试。
//
// 对外契约（异常、finish reason、取消语义、观测含义）与 <ninfer/engine.h> 保持一致；本文件里的
// 注释只讲这个核内部怎么组织，不再重复公开语义。

#include "core/device.h"
#include "core/nvtx.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/request_record.h"
#include "runtime/engine/context_cache/resource_manager.h"
#include "runtime/engine/scheduler.h"
#include "runtime/engine/generation_budget.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::runtime {

// EngineCore —— 一次引擎生命周期内唯一的请求执行核：一个后台 worker + 一组固定容量的请求位。
//
// 容量在构造期定死、之后不再改变：最多 max_concurrency（1..8）个 active lane，在册请求上限
// max_concurrency + max_pending_requests，等待期限 pending_timeout_ms。满了不排队干等，直接以
// Overloaded 或 QueueTimeout 结束——有界 FIFO ingress、不做抢占，是有意选定的产品行为。
//
// 与 Engine 的分工：Engine 负责把它立起来、把用户参数解析成具体值、在生成核与打分核之间分发；
// 核负责请求从入队到终态的全部过程，包括把结果发布出去和回收并发名额。
template <class Instance>
class EngineCore {

public:
    // 把模型实例的契约绑成核里使用的名字，是"模型无关"的落点：核只认这份契约，不认具体模型，
    // 缺哪个别名就在编译期报错而不是运行期失败。
    using ModelContract      = typename Instance::ModelContract;
    using Program            = typename ModelContract::Program;
    using BasePlan           = typename ModelContract::RequestBasePlan;
    using Plan               = typename ModelContract::AdmissionCandidate;
    using SequenceHandle     = typename ModelContract::SequenceHandle;
    using CaptureOffer       = typename ModelContract::CaptureOffer;
    using PendingBatch       = typename ModelContract::PendingBatch;
    using PreparedPrompt     = typename ModelContract::PreparedPrompt;
    using OutputSession      = typename ModelContract::OutputSession;
    using PublishedOutput    = typename ModelContract::PublishedOutput;
    using Request            = RequestRecord<ModelContract>;
    using Scheduling         = Scheduler<Request>;
    using FifoSnapshot       = typename Scheduling::FifoSnapshot;
    using RoundMembership    = typename Scheduling::RoundMembership;
    using ControlMembership  = typename Scheduling::ControlMembership;
    using ActiveAdmissionSet = typename Scheduling::ActiveAdmissionSet;
    using ExecutionAction    = typename Scheduling::ExecutionAction;
    using AdmissionGrant     = typename Scheduling::AdmissionGrant;
    using ResourceManagement = ResourceManager<ModelContract>;
    using ResourceInspection = typename ResourceManagement::Inspection;
    using Clock              = std::chrono::steady_clock;

    // 构造即进入可用状态：校验容量边界，拉起 worker，并等它把设备上下文绑到自己的线程上——绑不上
    // 就整个构造失败（worker 先 join 再抛）。走到返回处，引擎已经可以接 submit()。instance 与
    // device 都是引用：模型实例与设备上下文的生命期由 Engine 拥有，核只借来用。
    EngineCore(Instance& instance, DeviceContext& device, const EngineOptions& options,
               ContextMachineCostModel context_cost)
        : instance_(instance), device_(device), max_context_(options.max_context),
          max_concurrency_(options.max_concurrency),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          resources_(max_concurrency_, options.context_cache.max_private_continuations.value(),
                     options.context_cache.max_shared_prefixes.value(),
                     options.context_cache.enabled,
                     options.context_cache.max_long_anchors_per_continuation.value_or(0),
                     std::move(context_cost)) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("Engine core bounds are invalid");
        }
        if (!options.context_cache.max_private_continuations ||
            !options.context_cache.max_shared_prefixes) {
            throw std::logic_error("target admission capacity does not match the Engine");
        }
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                device_.bind_to_current_thread();
                startup.set_value();
            } catch (...) {
                startup.set_exception(std::current_exception());
                return;
            }
            worker_loop();
        });
        try {
            started.get();
        } catch (...) {
            if (worker_.joinable()) { worker_.join(); }
            throw;
        }
    }

    // 析构 = 停机：置 stopping_ 唤醒 worker，由 worker 先把在途请求全部以 Unavailable 结束，
    // 再 join 它。所以这里等的是"worker 退出"，不是"请求跑完"。
    // 调用顺序有讲究：Engine::Impl 特意把它排在设备 synchronize 之前，否则 worker 可能在同步之后
    // 还在下发 kernel（见 engine.cpp）。
    ~EngineCore() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    EngineCore(const EngineCore&)            = delete;
    EngineCore& operator=(const EngineCore&) = delete;

    // Submission —— 一次已入队请求的等待句柄，也是 GenerationHandle 的身体。
    //
    // 它只有两种归宿：被 wait() 消费掉结果，或者被析构——析构即取消（abandon_request），这正是
    // 公开契约"销毁没 wait 过的 GenerationHandle = 取消请求"的落点。取消只是置一个标记，真正的
    // 收敛发生在 worker 的稳定边界上；句柄被放弃也绝不会从消费者线程去碰 Program。
    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        // 阻塞到终态。sink 必须与提交时声明的 consumer mode 一致，否则在这里直接拒绝；随后交给
        // 消费者侧的泵函数去搬事件、轮询取消、取回结果或重抛 worker 存下的错误。
        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            const bool streaming = request_->consumer_mode == OutputConsumerMode::Streaming;
            if (streaming != (sink != nullptr)) {
                throw std::invalid_argument(
                    "GenerationHandle wait sink does not match its submitted consumer mode");
            }
            EngineCore* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(EngineCore& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        // 放弃这次请求：只有"从未 wait 过"的句柄会走到这里，wait() 已经先把两个指针换走了，
        // 所以"销毁句柄 = 取消"不会误伤已经消费过的请求。
        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        EngineCore* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class EngineCore;
    };

    // 提交：把"能不能受理"这件事**同步**做完。返回时请求一定已经躺在 pending_ 里、并发名额一定
    // 已经占上、请求身份与 publication order 一定已经分配；准入（分 lane、取资源）和执行都留给
    // worker。这里抛出的都是"根本没排上队"这一类：排队期限已过（QueueTimeout）、引擎不可用
    // （Unavailable）、队列满（Overloaded）、思考预算装不下（ThinkingBudgetCapacityInsufficient）。
    //
    // 两个容易忽略的细节：capacity_output 表达的是"这条提示最多还能生成多少 token"，用户的
    // requested_output_tokens 会被夹进剩余上下文再校验；以及所有抛出路径都先归还已经预占的
    // outstanding 名额，失败不会泄漏容量。
    Submission submit(PreparedPrompt prompt, PromptSummary prompt_summary, double prepare_seconds,
                      ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
                      GenerationObservationOptions observation,
                      Clock::time_point pending_deadline = {}) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        std::uint64_t request_id        = 0;
        std::uint64_t publication_order = 0;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding_ >= max_outstanding_) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            if (next_request_id_ == 0 || next_publication_order_ == 0) {
                throw std::overflow_error("request identity space exhausted");
            }
            ++outstanding_;
            request_id        = next_request_id_++;
            publication_order = next_publication_order_++;
        }

        std::shared_ptr<Request> request;
        try {
            auto output = instance_.frontend.make_output_session(
                prompt, options.stop, options.output, options.execution.thinking);
            const std::uint32_t capacity_output =
                max_context_ - prompt_summary.prompt_tokens + static_cast<std::uint32_t>(1);
            try {
                output.validate_generation_capacity(
                    std::min(options.execution.requested_output_tokens, capacity_output));
            } catch (const std::invalid_argument& error) {
                throw RequestError(RequestErrorKind::ThinkingBudgetCapacityInsufficient,
                                   error.what());
            }
            request = std::make_shared<Request>(request_id, publication_order, std::move(prompt),
                                                std::move(output), prompt_summary, prepare_seconds,
                                                std::move(options), consumer_mode, observation,
                                                pending_deadline, submitted);
        } catch (...) {
            release_reserved_capacity();
            throw;
        }

        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                --outstanding_;
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
        }
        request_admission_check();
        queue_cv_.notify_one();
        return Submission(*this, std::move(request));
    }

    // ---- 公开观测：从 Engine 透传给调用方，语义见 <ninfer/engine.h> ----

    // 活的物理视图：Program 的 arena / workspace / KV 占用，叠上构造期解出的 KV 容量事实。
    // 它要和 worker 的执行单位互斥，所以可能被短暂挡住。
    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        return out;
    }

    // 读的是 worker 在稳定边界发布的快照，不是活的状态：代价只有一把小锁，数值允许滞后，
    // 从未跑过请求的引擎返回全零。计数器单调递增，gauge 类字段（waiting/running/…）随边界变化，
    // 调用方靠两次快照相减得到区间值。故意不提供"清零统计"的接口。
    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex_);
        return published_stats_;
    }

    // 停机中（stopping_）或 worker 已经因不可恢复错误永久退出（failed_）都算不可用，此后 submit
    // 一律抛 Unavailable。failed_ 是单向的：一旦置上，引擎不会自己恢复。
    [[nodiscard]] bool is_available() const {
        std::lock_guard lock(queue_mutex_);
        return !stopping_ && !failed_;
    }

    // 唯一的 noexcept 写操作：清峰值本身失败没有副作用，所以异常就地吞掉，不往上抛。
    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
        } catch (...) {}
    }

private:
    // ========================================================================
    // 主机侧计时与统计
    // ========================================================================
    //
    // worker 的墙上时间被记成两类事实：全局的 cumulative_stats_（引擎累计）和每请求的
    // RequestHostTiming（暴露给单个请求）。记账遵循 <ninfer/types.h> 里 RuntimeStats 与
    // GenerationEngineTiming 的约定：
    //
    //   * 顶层 host 阶段互斥，detail 值是某个顶层的子集，不能重复相加；
    //   * device_wait 是阻塞等待，单独成栏，不算 host 活跃时间；
    //   * "exposed" 是**延迟暴露**而不是耗时归属——被一个执行单位延迟到的每个 active 请求都
    //     观察到该单位的完整耗时，所以并发请求之间的这些数字**不能相加**。

    // 当前执行单位的类别：决定新记的时间落进 prefill / decode / control 哪一栏。
    enum class HostWorkClass : std::uint8_t {
        Decode,
        Prefill,
        Control,
    };

    // Engine 自己的三个顶层阶段（不含 Program 内部）：边界处理、输出提交、维护。
    using EngineHostPhase = RequestEngineHostPhase;

    // 阶段名到 NVTX 名字的映射，保证 profiler 里的区间与 RuntimeStats 的栏目一一对应。
    [[nodiscard]] static nvtx::Name phase_range_name(EngineHostPhase phase) noexcept {
        switch (phase) {
        case EngineHostPhase::Boundary:
            return nvtx::Name::EngineBoundary;
        case EngineHostPhase::CommitOutput:
            return nvtx::Name::EngineCommitOutput;
        case EngineHostPhase::Maintenance:
            return nvtx::Name::EngineMaintenance;
        }
        return nvtx::Name::EngineBoundary;
    }

    // 一个会被某个执行单位"延迟到"的 active 请求（连同它所在的 lane）。
    struct ActiveExposure {
        std::shared_ptr<Request> request;
        std::uint32_t lane = 0;
    };

    struct ActiveExposureSet {
        std::array<ActiveExposure, kMaximumConcurrency> entries{};
        std::size_t size = 0;
    };

    // 一段 host 阶段开始时的取景：起点、已经记过账的嵌套时间、以及当时在场的 active 请求。
    // 收尾时用 accounted_before 把嵌套时间从墙上时间里扣掉，避免父阶段重复吞掉子阶段。
    struct HostPhaseMeasurement {
        Clock::time_point started;
        std::uint64_t accounted_before = 0;
        ActiveExposureSet exposed;
    };

    // 单调时钟差值取非负值：计时口径统一走这里，避免各处重复写 duration_cast。
    [[nodiscard]] static std::uint64_t elapsed_ns(Clock::time_point started,
                                                  Clock::time_point finished) noexcept {
        const auto count =
            std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count();
        return count > 0 ? static_cast<std::uint64_t>(count) : 0;
    }

    // 快照当前所有 active lane，作为"这次执行会耽误到谁"的名单。
    [[nodiscard]] ActiveExposureSet active_exposure_set() const {
        ActiveExposureSet result;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            result.entries[result.size++] = ActiveExposure{.request = slots_[lane], .lane = lane};
        }
        return result;
    }

    // 阶段开始：同时记下时间基准、已记账的嵌套量、以及当时在场的请求名单。
    [[nodiscard]] HostPhaseMeasurement begin_host_phase() const {
        return HostPhaseMeasurement{
            .started          = Clock::now(),
            .accounted_before = worker_accounted_elapsed_ns_,
            .exposed          = active_exposure_set(),
        };
    }

    // 声明"接下来这一段属于哪类工作"，decode 时还要给出参与本轮的行，用于判断某个请求究竟是被
    // decode 耽误的，还是被别的工作耽误的。
    void set_host_work_class(HostWorkClass work_class,
                             std::span<const std::uint32_t> decode_lanes = {}) noexcept {
        current_host_work_class_   = work_class;
        current_decode_lane_count_ = decode_lanes.size();
        for (std::size_t i = 0; i < decode_lanes.size(); ++i) {
            current_decode_lanes_[i] = decode_lanes[i];
        }
    }

    [[nodiscard]] bool current_decode_contains(std::uint32_t lane) const noexcept {
        return std::find(current_decode_lanes_.begin(),
                         current_decode_lanes_.begin() +
                             static_cast<std::ptrdiff_t>(current_decode_lane_count_),
                         lane) != current_decode_lanes_.begin() +
                                      static_cast<std::ptrdiff_t>(current_decode_lane_count_);
    }

    // 按当前工作类别把时间记进 prefill / decode / control 三栏之一。host 时间与 device 等待
    // 分开记：后者是阻塞，不算 host 活跃。
    void add_class_host_time(std::uint64_t host_ns, std::uint64_t device_wait_ns) noexcept {
        RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
        switch (current_host_work_class_) {
        case HostWorkClass::Decode:
            stats.decode_host_ns += host_ns;
            stats.decode_device_wait_ns += device_wait_ns;
            break;
        case HostWorkClass::Prefill:
            stats.prefill_host_ns += host_ns;
            stats.prefill_device_wait_ns += device_wait_ns;
            break;
        case HostWorkClass::Control:
            stats.control_host_ns += host_ns;
            stats.control_device_wait_ns += device_wait_ns;
            break;
        }
    }

    // 把这段耗时记到名单里每个请求头上——这就是"暴露"语义的实现：一个单位的耗时，会被所有被它
    // 耽误的请求同时观察到。
    void expose_engine_phase(const ActiveExposureSet& exposed, EngineHostPhase phase,
                             std::uint64_t elapsed) noexcept {
        for (std::size_t i = 0; i < exposed.size; ++i) {
            const ActiveExposure& exposure = exposed.entries[i];
            RequestHostTiming& timing      = exposure.request->host_timing;
            timing.expose_engine(phase, elapsed,
                                 current_host_work_class_ == HostWorkClass::Decode &&
                                     current_decode_contains(exposure.lane));
        }
    }

    // Engine 阶段的收尾：扣掉嵌套时间得到本阶段自己的耗时，记进全局与每请求两处账本。
    void finish_engine_phase(const HostPhaseMeasurement& measurement,
                             EngineHostPhase phase) noexcept {
        const std::uint64_t wall    = elapsed_ns(measurement.started, Clock::now());
        const std::uint64_t nested  = worker_accounted_elapsed_ns_ - measurement.accounted_before;
        const std::uint64_t own     = wall > nested ? wall - nested : 0;
        RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
        switch (phase) {
        case EngineHostPhase::Boundary:
            stats.engine_boundary_ns += own;
            break;
        case EngineHostPhase::CommitOutput:
            stats.engine_commit_output_ns += own;
            break;
        case EngineHostPhase::Maintenance:
            stats.engine_maintenance_ns += own;
            break;
        }
        add_class_host_time(own, 0);
        expose_engine_phase(measurement.exposed, phase, own);
        worker_accounted_elapsed_ns_ += own;
    }

    // 接收 Program 报回的耗时（提交 host 时间、post host 时间、device 等待），按类别记账，并把它
    // 暴露给名单里的请求。时间是 Program 自己报的，Engine 直接采信；报少了的那部分由
    // finish_program_call 用墙上时间补齐。
    void record_program_timing(runtime::ExecutionTiming timing,
                               const ActiveExposureSet& exposed) noexcept {
        RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
        stats.program_submit_ns += timing.submit_host_ns;
        stats.program_post_ns += timing.post_host_ns;
        stats.device_wait_ns += timing.device_wait_ns;
        add_class_host_time(timing.host_ns(), timing.device_wait_ns);
        for (std::size_t i = 0; i < exposed.size; ++i) {
            const ActiveExposure& exposure = exposed.entries[i];
            RequestHostTiming& request     = exposure.request->host_timing;
            request.expose_program(timing, current_host_work_class_ == HostWorkClass::Decode &&
                                               current_decode_contains(exposure.lane));
        }
        worker_accounted_elapsed_ns_ += timing.elapsed_ns();
    }

    // Program 调用的收尾：墙上时间若比"Program 自报 + 已记账的嵌套"更长，差额补记到提交侧——
    // 那部分是真花在 host 上、但没被 Program 记账的时间。
    void finish_program_call(const HostPhaseMeasurement& measurement,
                             runtime::ExecutionTiming timing) noexcept {
        const std::uint64_t wall     = elapsed_ns(measurement.started, Clock::now());
        const std::uint64_t nested   = worker_accounted_elapsed_ns_ - measurement.accounted_before;
        const std::uint64_t observed = timing.elapsed_ns() + nested;
        if (wall > observed) { timing.submit_host_ns += wall - observed; }
        record_program_timing(timing, measurement.exposed);
    }

    // 明细计时：给某个顶层阶段里的子步骤记账（耗时 + 调用次数）。明细是顶层的子集，
    // 所以只加进 cumulative_stats_，不再参与顶层互斥时间的分配。
    void record_detail(std::uint64_t RuntimeHostWorkStats::*elapsed_member,
                       std::uint64_t RuntimeHostWorkStats::*invocation_member,
                       Clock::time_point started) noexcept {
        RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
        stats.*elapsed_member += elapsed_ns(started, Clock::now());
        ++(stats.*invocation_member);
    }

    // 三个 RAII 计时器，存在的理由都是"异常路径上也必须把账记上"。DetailScope 记明细；
    // EnginePhaseScope 记 Engine 自己的阶段，并允许在调用 Program 前后暂停/恢复 NVTX 区间；
    // ProgramCallScope 记一次 Program 调用，失败时也能拿到兜底的计时对象。
    class DetailScope {
    public:
        DetailScope(EngineCore& owner, std::uint64_t RuntimeHostWorkStats::*elapsed_member,
                    std::uint64_t RuntimeHostWorkStats::*invocation_member,
                    nvtx::Name range_name) noexcept
            : owner_(owner), elapsed_member_(elapsed_member), invocation_member_(invocation_member),
              started_(Clock::now()) {
            range_.emplace(range_name, nvtx::Category::Control);
        }

        ~DetailScope() {
            range_.reset();
            owner_.record_detail(elapsed_member_, invocation_member_, started_);
        }

        DetailScope(const DetailScope&)            = delete;
        DetailScope& operator=(const DetailScope&) = delete;

    private:
        EngineCore& owner_;
        std::uint64_t RuntimeHostWorkStats::*elapsed_member_;
        std::uint64_t RuntimeHostWorkStats::*invocation_member_;
        Clock::time_point started_;
        std::optional<nvtx::ScopedRange> range_;
    };

    class EnginePhaseScope {
    public:
        EnginePhaseScope(EngineCore& owner, EngineHostPhase phase)
            : owner_(owner), phase_(phase), measurement_(owner.begin_host_phase()) {
            range_.emplace(phase_range_name(phase), nvtx::Category::Runtime);
        }

        ~EnginePhaseScope() { finish(); }

        EnginePhaseScope(const EnginePhaseScope&)            = delete;
        EnginePhaseScope& operator=(const EnginePhaseScope&) = delete;

        // 暂停 / 恢复只影响 NVTX 区间：调用 Program 期间时间记在 Program 那一侧，避免同一个区间
        // 在 profiler 里被套两层。计时本身不停。
        void pause_range() noexcept { range_.reset(); }

        void resume_range() noexcept {
            if (active_ && !range_) {
                range_.emplace(phase_range_name(phase_), nvtx::Category::Runtime);
            }
        }

        void finish() noexcept {
            if (!active_) { return; }
            range_.reset();
            owner_.finish_engine_phase(measurement_, phase_);
            active_ = false;
        }

    private:
        EngineCore& owner_;
        EngineHostPhase phase_;
        HostPhaseMeasurement measurement_;
        std::optional<nvtx::ScopedRange> range_;
        bool active_ = true;
    };

    class ProgramCallScope {
    public:
        explicit ProgramCallScope(EngineCore& owner)
            : owner_(owner), measurement_(owner.begin_host_phase()) {}

        ~ProgramCallScope() noexcept { finish(failed_timing_); }

        ProgramCallScope(const ProgramCallScope&)            = delete;
        ProgramCallScope& operator=(const ProgramCallScope&) = delete;

        // 交给 Program 填写的兜底计时：调用抛异常时 Program 也要把已经花掉的时间写进来，这样
        // 失败路径的耗时不会凭空消失。
        [[nodiscard]] runtime::ExecutionTiming& failed_timing() noexcept { return failed_timing_; }

        // 成功路径用 Program 返回的计时收尾；重复调用无副作用。
        void finish(runtime::ExecutionTiming timing) noexcept {
            if (!active_) { return; }
            owner_.finish_program_call(measurement_, timing);
            active_ = false;
        }

    private:
        EngineCore& owner_;
        HostPhaseMeasurement measurement_;
        runtime::ExecutionTiming failed_timing_;
        bool active_ = true;
    };

    // 组装并发布一次统计快照：累计计数器 + ResourceManager/Program 的贡献 + 当场数的 gauge
    // （队列深度、各状态请求数）。只在 worker 的稳定边界被调用，所以读统计的开销不落在执行
    // 热路径上，代价是快照会略微滞后于真实状态。
    void publish_runtime_stats() {
        HostPhaseMeasurement measurement = begin_host_phase();
        std::optional<nvtx::ScopedRange> phase_range;
        phase_range.emplace(nvtx::Name::EngineMaintenance, nvtx::Category::Runtime);
        const Clock::time_point detail_started = Clock::now();
        std::optional<nvtx::ScopedRange> detail_range;
        detail_range.emplace(nvtx::Name::StatsPublication, nvtx::Category::Control);
        RuntimeStats snapshot = cumulative_stats_;
        resources_.populate_runtime_stats(*instance_.program, snapshot);
        {
            std::lock_guard lock(queue_mutex_);
            snapshot.waiting_requests = static_cast<std::uint32_t>(pending_.size());
        }
        snapshot.prefilling_requests = 0;
        if (const auto lane = scheduler_.prefill_lane();
            lane && slots_[*lane] != nullptr && !slots_[*lane]->capture_pending) {
            snapshot.prefilling_requests = 1;
        }
        snapshot.materializing_requests = materializing_.has_value() ? 1U : 0U;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            ++snapshot.running_requests;
            if (slots_[lane]->is_decode_ready()) { ++snapshot.decode_ready_requests; }
            if (slots_[lane]->capture_pending) { ++snapshot.capture_pending_requests; }
            if (slots_[lane]->terminal_reason) { ++snapshot.terminal_pending_requests; }
        }
        detail_range.reset();
        record_detail(&RuntimeHostWorkStats::stats_publication_ns,
                      &RuntimeHostWorkStats::stats_publication_invocations, detail_started);
        phase_range.reset();
        finish_engine_phase(measurement, EngineHostPhase::Maintenance);
        snapshot.host_work = cumulative_stats_.host_work;
        std::lock_guard lock(stats_mutex_);
        published_stats_ = snapshot;
    }

    // 记录本次准入实际选中的前缀复用路径与复用长度，用于观察缓存收益来自哪一类来源。
    void record_prefix_selection(const RequestPlanSummary& summary) noexcept {
        switch (summary.prefix_reuse_path) {
        case PrefixReusePath::Root:
            ++cumulative_stats_.root_selections;
            break;
        case PrefixReusePath::PrivateEndpoint:
            ++cumulative_stats_.private_endpoint_selections;
            break;
        case PrefixReusePath::PrivateTurnClosure:
            ++cumulative_stats_.private_turn_closure_selections;
            break;
        case PrefixReusePath::PrivateResponseReplay:
            ++cumulative_stats_.private_response_replay_selections;
            break;
        case PrefixReusePath::PrivateLongAnchor:
            ++cumulative_stats_.private_long_anchor_selections;
            break;
        case PrefixReusePath::SharedStablePrefix:
            ++cumulative_stats_.shared_stable_prefix_selections;
            break;
        }
        cumulative_stats_.reused_prompt_tokens += summary.reusable_prompt_tokens;
        cumulative_stats_.last_selected_frontier_tokens = summary.reusable_prompt_tokens;
    }

    // 消费者侧的泵：wait() 的全部工作都在这里，而且是整份代码里唯一由调用方线程驱动、又需要
    // 与 worker 协作的部分。它反复做三件事——把 worker 已经发布的事件搬给 sink、采样调用方的
    // cancellation、检查请求是否到达终态——直到能返回结果或重抛错误为止。
    //
    // 两条边界必须守住：
    //   * sink 抛出的异常归调用方自己：先取消这次请求（不能让一个已经没人接收的请求继续跑完），
    //     再把异常原样抛出；
    //   * 无论怎么退出（正常、抛异常、取消），都要释放消费者这一侧，容量才可能回收。
    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            EngineCore* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        std::optional<GenerationStart> start;
        std::optional<PromptProgress> progress;
        std::vector<typename Request::StreamEvent> events;
        for (;;) {
            start.reset();
            progress.reset();
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                    return request->response_done || request->stream_start.has_value() ||
                           request->stream_progress.has_value() || !request->events.empty();
                });
                start = std::move(request->stream_start);
                request->stream_start.reset();
                progress = std::move(request->stream_progress);
                request->stream_progress.reset();
                events.swap(request->events);
                done = request->response_done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    if (start) { sink->start(std::move(*start)); }
                    if (progress) { sink->progress(std::move(*progress)); }
                    for (auto& event : events) {
                        if (auto* timing = std::get_if<GenerationTimingObservation>(&event)) {
                            sink->timing(std::move(*timing));
                        } else {
                            sink->publish(std::move(std::get<OutputDelta>(event)));
                        }
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    request_admission_check();
                    queue_cv_.notify_one();
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        request_admission_check();
                        queue_cv_.notify_one();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    request_admission_check();
                    queue_cv_.notify_one();
                }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    // 一次控制动作是否真的推进了什么：ControlProgress 表示队列或资源状态变了、值得让 worker
    // 再试一轮准入。
    enum class AdmissionProgress : std::uint8_t {
        None,
        ControlProgress,
    };

    // 合并 admission 可见的队列/资源变化。它是个标志而不是信号量：一次被暂时判定为阻塞的检查
    // 不会被普通 prefill/decode 进展重新武装，否则每轮执行都会白白重跑一次准入。
    void request_admission_check() noexcept {
        admission_check_pending_.store(true, std::memory_order_release);
    }

    // 取走标志：同一批变化只触发一次准入尝试。
    [[nodiscard]] bool consume_admission_check() noexcept {
        return admission_check_pending_.exchange(false, std::memory_order_acq_rel);
    }

    // 一次正在进行的资源 transition 在 Engine 侧需要的账目：目标 lane、已备好的生成预算、计划
    // 摘要、以及准入时拿到的回填类别与保护期。请求此刻还没有 lane/sequence，只有这份记录。
    struct MaterializingRequest {
        std::shared_ptr<Request> request;
        LaneId destination;
        GenerationBudget budget;
        RequestPlanSummary summary;
        BackfillClass backfill_class   = BackfillClass::None;
        std::uint64_t protection_epoch = 0;
        Clock::time_point started;
    };

    // ========================================================================
    // 输出与进度发布
    // ========================================================================
    //
    // 发布分两路：Aggregate 请求只在终态把累积文本交出去；Streaming 请求则要按顺序把
    // GenerationStart → 增量 OutputDelta / 进度 / 计时事件流出去。两路共用同一份"累积文本 +
    // 事件队列"，所以无论走哪条路，最终 content/reasoning 都是一致的。

    // 记下本单元真正被接受了多少 token，并据此产出计时观测（只在观察开关打开时才有值）。
    // 第一次接受 token 的时刻就是 first token，prompt 与 generation 的墙钟口径都从它分界。
    [[nodiscard]] std::optional<GenerationTimingObservation>
    record_committed_output(const std::shared_ptr<Request>& request,
                            std::uint32_t accepted_tokens) {
        if (accepted_tokens == 0) { return std::nullopt; }
        const bool observe_wall =
            request->observation.phase_timings || request->observation.live_timings;
        const bool need_now         = !request->first_token || observe_wall;
        const Clock::time_point now = need_now ? Clock::now() : Clock::time_point{};
        if (!request->first_token) { request->first_token = now; }
        if (!observe_wall) { return std::nullopt; }
        if (!request->admitted_at || !request->first_token) {
            throw std::logic_error("committed output has no observed admission boundary");
        }
        request->last_token = now;
        if (!request->observation.live_timings) { return std::nullopt; }
        if (request->generated.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("generated token count exceeds observation domain");
        }
        return GenerationTimingObservation{
            .generated_tokens      = static_cast<std::uint32_t>(request->generated.size()),
            .prompt_elapsed_ns     = elapsed_ns(*request->admitted_at, *request->first_token),
            .generation_elapsed_ns = elapsed_ns(*request->first_token, *request->last_token),
        };
    }

    // 一次发布的落点：累积文本永远更新，事件只对 Streaming 入队并唤醒消费者。所以 Aggregate
    // 请求不需要任何消费者在线，结果照样是完整的。
    void append_output(const std::shared_ptr<Request>& request, PublishedOutput output,
                       std::optional<GenerationTimingObservation> timing = std::nullopt) {
        if (output.empty() && !timing) { return; }
        const bool streaming = request->consumer_mode == OutputConsumerMode::Streaming;
        {
            std::lock_guard lock(request->mutex);
            if (streaming && timing) { request->events.emplace_back(std::move(*timing)); }
            for (OutputDelta& delta : output) {
                std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                              : request->content;
                full += delta.text;
                if (streaming) { request->events.emplace_back(std::move(delta)); }
            }
        }
        if (streaming) { request->cv.notify_one(); }
    }

    // prompt 前沿：只按**已经完成**的 Program 工作推进，不预支。它挂在 begin 这份准入时已提交的
    // 复用选择上，所以进度里的总数与复用数在整条请求里保持自洽。
    void publish_prompt_progress(const std::shared_ptr<Request>& request) {
        if (!request->observation.prompt_progress) { return; }
        if (!request->admitted_begin || !request->admitted_at) {
            throw std::logic_error("prompt progress has no admitted request boundary");
        }
        const BeginSummary& begin = *request->admitted_begin;
        if (begin.reused_prompt_tokens > begin.prompt_tokens ||
            request->computed_prompt_tokens > begin.prompt_tokens - begin.reused_prompt_tokens) {
            throw std::logic_error("prompt progress exceeds the admitted prompt frontier");
        }
        const PromptProgress progress{
            .total_prompt_tokens     = begin.prompt_tokens,
            .reused_prompt_tokens    = begin.reused_prompt_tokens,
            .processed_prompt_tokens = begin.reused_prompt_tokens + request->computed_prompt_tokens,
            .elapsed_ns              = elapsed_ns(*request->admitted_at, Clock::now()),
        };
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->stream_progress = progress;
        }
        request->cv.notify_one();
    }

    // Streaming 的第一条事件，恰好一条，且在一切输出增量之前。它公布的是准入时已经定下的资源
    // 选择事实（提示规模与复用长度），不必等 prefill 跑完；顺带在这里记下"请求已受理"的时间
    // 基准，之后的 prompt/generation 墙钟都从它算起。
    void publish_generation_start(const std::shared_ptr<Request>& request, BeginSummary begin) {
        if (request->admitted_begin) {
            throw std::logic_error("request admission published generation start twice");
        }
        request->admitted_begin = begin;
        if (request->observation.phase_timings || request->observation.live_timings ||
            request->observation.prompt_progress) {
            request->admitted_at = Clock::now();
        }
        if (request->consumer_mode != OutputConsumerMode::Streaming) { return; }
        {
            std::lock_guard lock(request->mutex);
            if (request->stream_start || request->response_done) {
                throw std::logic_error("streaming request has an invalid generation-start state");
            }
            request->stream_start = GenerationStart{
                .prompt               = request->prompt_summary,
                .reused_prompt_tokens = begin.reused_prompt_tokens,
            };
        }
        request->cv.notify_one();
    }

    // ========================================================================
    // 容量与请求生命周期
    // ========================================================================
    //
    // outstanding 名额由两个独立事实共同决定何时归还：
    //
    //     response_done       worker 已经形成了最终结果或错误
    //     consumer_released   wait() 已结束，或者句柄被放弃
    //
    // 两者可以任意先后到达，但容量只释放一次。这条规则是有意的：一个已经跑完却没人来取的请求，
    // 依然占着它的名额——这是调用方能看到、也应该看到的容量语义。

    void release_reserved_capacity() noexcept {
        std::lock_guard lock(queue_mutex_);
        if (outstanding_ != 0) { --outstanding_; }
    }

    // 消费者侧释放：到达终态就顺手把容量还掉。
    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->response_done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    // 句柄被放弃：只做两件事——置取消标记（worker 会在下一个稳定边界收拢），以及释放消费者。
    // 它绝不从调用方线程去碰 Program，所以放弃句柄永远不会阻塞或拖慢执行。
    void abandon_request(std::shared_ptr<Request> request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        request_admission_check();
        queue_cv_.notify_one();
        release_consumer(request);
    }

    // 终态侧释放：与 release_consumer 互为镜像，谁后到谁负责归还容量。
    bool mark_completed(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        return release;
    }

    // 计划只服务于从入队到准入这一段，进入执行后就没有意义了，及时放掉它占的内存。
    void release_planning_state(const std::shared_ptr<Request>& request) noexcept {
        request->base_plan.reset();
    }

    // ========================================================================
    // 终态：三种收尾方式
    // ========================================================================
    //
    // complete_error 交付一个异常，complete_success 交付一份结果，complete_cancelled 是"取消"
    // 这一种正常终态（交付结果，finish_reason == Cancelled，并带上尽力而为的部分输出）。
    // 三者都在最后一步置 response_done、尝试归还容量、唤醒消费者，且都幂等——已经终态的请求
    // 不会被覆盖。

    // 失败终态：清掉执行期状态，把错误交给等待方去重抛。
    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        release_planning_state(request);
        request->prompt      = {};
        request->model_state = EngineRequestState::ModelFinished;
        request->sequence.reset();
        request->lane.reset();
        request->budget.reset();
        request->terminal_reason.reset();
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->error         = std::move(error);
            request->response_done = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    // 成功终态：把散在请求各处的东西装配成一份 GenerationResult——输出文本、工具调用、token
    // 序列、各类统计与耗时——然后释放执行期状态。结果只交付给 wait()，Aggregate 与 Streaming
    // 拿到的结构完全相同。
    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason) {
        HostPhaseMeasurement completion = begin_host_phase();
        double prompt_wall_seconds      = 0.0;
        double generation_wall_seconds  = 0.0;
        if (request->observation.phase_timings && request->first_token) {
            if (!request->admitted_at || !request->last_token) {
                throw std::logic_error("observed request completed without stable timing bounds");
            }
            prompt_wall_seconds =
                std::chrono::duration<double>(*request->first_token - *request->admitted_at)
                    .count();
            generation_wall_seconds =
                std::chrono::duration<double>(*request->last_token - *request->first_token).count();
        }
        release_planning_state(request);
        request->prompt      = {};
        request->model_state = EngineRequestState::ModelFinished;
        if (!request->queue_wait_recorded) {
            request->host_timing.queue_wait_ns = elapsed_ns(request->submitted, Clock::now());
            request->queue_wait_recorded       = true;
        }
        GenerationResult result;
        result.prompt                  = request->prompt_summary;
        result.generated_token_ids     = std::move(request->generated);
        result.content                 = std::move(request->content);
        result.reasoning               = std::move(request->reasoning);
        result.tool_calls              = request->output.take_tool_calls();
        result.tool_call_parse         = request->output.tool_call_parse_diagnostics();
        result.reasoning_tokens        = request->output.reasoning_tokens();
        result.finish_reason           = reason;
        result.matched_stop_string     = request->output.matched_stop_string();
        result.timings.prepare_seconds = request->prepare_seconds;
        if (request->begin) {
            result.reused_prompt_tokens = request->begin->reused_prompt_tokens;
            result.prefix_reuse_path    = request->begin->prefix_reuse_path;
        }
        result.timings                 = request->generation_timings;
        result.timings.prepare_seconds = request->prepare_seconds;
        result.speculative             = std::move(request->speculative_stats);
        result.thinking                = request->output.thinking_stats();
        result.materialization         = request->materialization_diagnostics;
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.prompt_wall_seconds     = prompt_wall_seconds;
        result.timings.generation_wall_seconds = generation_wall_seconds;
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        request->sequence.reset();
        request->lane.reset();
        request->budget.reset();
        request->terminal_reason.reset();
        finish_engine_phase(completion, EngineHostPhase::CommitOutput);
        result.engine_timing = request->host_timing.public_snapshot();
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->result        = std::move(result);
            request->response_done = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    // 取消终态：先让输出策略结账（停在哪里、要不要保留已经解出的内容），再按成功路径收尾。
    // 取消不是失败，所以走 complete_success 而不是 complete_error。
    void complete_cancelled(const std::shared_ptr<Request>& request) {
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    // "没人等"的取消：取消收尾本身也可能失败（策略结账是用户可见逻辑，会抛）。这里兜住它——
    // 先降级成一个错误终态保证请求有结论，再把异常抛给 worker 决定是否升级为引擎级失败。
    void complete_detached_cancelled(const std::shared_ptr<Request>& request) {
        try {
            complete_cancelled(request);
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            complete_error(request, error);
            throw;
        }
    }

    // lane 空出来就是一次 admission 可见的容量变化。
    void remove_completed_slot(std::uint32_t lane) {
        slots_[lane].reset();
        request_admission_check();
    }

    // ========================================================================
    // 取消与终态结算
    // ========================================================================
    //
    // 取消是协作式的，且只在这些边界上被采样：worker 每轮把取消标记快照一次，随后按快照处理，
    // 已下发的 GPU 工作不会被中途改写。这样"用户点了取消"与"引擎实际停下"之间允许有一个执行
    // 单位的延迟，换来的是物理状态永远处在可解释的位置。

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept {
        std::array<bool, kMaximumConcurrency> cancelled{};
        // 已有 active 单位在跑、且还有另一个 row 持有全局资源事务时，取消不能释放拓扑——必须等
        // 那个事务到达稳定终态，否则会拆掉别人正在使用的物理状态。
        if (instance_.program->has_context_transaction()) { return cancelled; }
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
            }
        }
        return cancelled;
    }

    // 终态结算：模型执行已经结束的请求，在这里完成"保留还是释放"的选择，并采其完整结果。
    // 按 publication order 从小到大依次处理，保证一个请求的终态效果（例如 SessionIndex 的绑定）
    // 总是被更晚提交的结果覆盖，而不会被更早的抢先。资源事务进行中时整轮跳过。
    bool settle_terminal_requests(HostPhaseMeasurement& boundary) {
        const bool manager_transaction = resources_.context_transaction_kind().has_value();
        const bool program_transaction = instance_.program->has_context_transaction();
        if (manager_transaction != program_transaction) {
            throw std::logic_error("Engine and Program disagree before terminal settlement");
        }
        if (program_transaction) { return false; }

        bool changed = false;
        for (;;) {
            std::optional<std::uint32_t> selected;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                const auto& request = slots_[lane];
                if (request == nullptr || !request->terminal_reason) { continue; }
                if (!selected ||
                    request->publication_order < slots_[*selected]->publication_order) {
                    selected = lane;
                }
            }
            if (!selected) { break; }

            const std::uint32_t lane = *selected;
            const auto request       = slots_[lane];
            if (!request->is_model_finished() || request->capture_pending || !request->sequence ||
                !request->lane || request->lane->value != lane ||
                resources_.lane_state(LaneId{lane}) != LogicalLaneState::TerminalPending) {
                throw std::logic_error("terminal-pending request has invalid ownership");
            }
            const FinishReason reason = *request->terminal_reason;
            auto finished =
                resources_.finish(*instance_.program, *request->lane, *request->sequence);
            request->generation_timings = finished.timings;
            request->speculative_stats  = std::move(finished.speculative);
            request->terminal_reason.reset();

            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            complete_success(request, reason);
            remove_completed_slot(lane);
            boundary = begin_host_phase();
            changed  = true;
        }
        if (changed) { publish_runtime_stats(); }
        return changed;
    }

    // 取消已经 Active 的请求：abort 现有 sequence、采用 abort 的完整结果、带着部分输出进入取消
    // 终态。正在做 capture 的请求要等到 capture 落定（它持有全局资源事务），所以不在这一轮处理。
    void cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary,
                                HostPhaseMeasurement& boundary) {
        if (instance_.program->has_context_transaction()) { return; }
        bool changed = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
            if (request->capture_pending) { continue; }
            if (!request->sequence || !request->lane || request->lane->value != lane) {
                throw std::logic_error("active cancellation has no sequence binding");
            }
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            auto aborted = resources_.abort(*instance_.program, *request->lane, *request->sequence);
            request->generation_timings = aborted.timings;
            request->speculative_stats  = std::move(aborted.speculative);
            if (scheduler_.prefill_lane() == lane) { scheduler_.clear_prefill_lane(lane); }
            append_output(request, request->output.commit_preview());
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            complete_success(request, FinishReason::Cancelled);
            remove_completed_slot(lane);
            boundary = begin_host_phase();
            changed  = true;
        }
        if (changed) { publish_runtime_stats(); }
    }

    // 还没有准入的等待者在这里出清：被取消的以 Cancelled 结束，过了排队期限的以 QueueTimeout
    // 结束。它们从未建立 Program 状态，所以可以直接了结，不需要任何资源事务。
    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
            for (auto it = pending_.begin(); it != pending_.end();) {
                if ((*it)->cancelled.load(std::memory_order_acquire)) {
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (now >= (*it)->deadline) {
                    expired.push_back(*it);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
            have_pending = !pending_.empty();
        }
        for (const auto& request : cancelled) { scheduler_.on_waiting_removed(request->id); }
        for (const auto& request : expired) { scheduler_.on_waiting_removed(request->id); }
        try {
            for (const auto& request : cancelled) { complete_detached_cancelled(request); }
            for (const auto& request : expired) {
                complete_error(request,
                               std::make_exception_ptr(RequestError(
                                   RequestErrorKind::QueueTimeout,
                                   "inference request expired while waiting for admission")));
            }
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            for (const auto& request : cancelled) { complete_error(request, error); }
            for (const auto& request : expired) { complete_error(request, error); }
            throw;
        }
        if (!cancelled.empty() || !expired.empty()) {
            request_admission_check();
            publish_runtime_stats();
        }
        return have_pending;
    }

    // ========================================================================
    // 模型单位事务：采用一次 GPU 执行的结果
    // ========================================================================
    //
    // pending batch 是 Program 交出的一批"待接受"token（每行一个请求，行宽相同）。这个函数把它们
    // 变成 Engine 认可的既成事实，顺序固定：
    //
    //   1. 校验 batch 布局与每一行的状态都还是准入时的那一个；
    //   2. 问输出策略：这行能接受几个 token、是不是终点、要不要注入控制 token（取消行直接判 0）；
    //   3. 把接受的 token 暂存进请求（此时还没有提交，失败要能整体退回）；
    //   4. 调 Program::commit 落定物理状态，并核对它报回的 disposition 与我们的预期逐行一致；
    //   5. 采用结果、发布输出、推进每个请求的模型状态。
    //
    // 事务性是重点：第 2~4 步任何一步失败，都要把暂存的 token 退回去并把 batch 整体 abort。这些
    // 失败都是"不变量被破坏"，所以异常会一路抛到 worker 升级成引擎级失败，而不是悄悄丢弃这一轮。

    void commit_pending(PendingBatch&& pending, std::span<const std::uint32_t> lane_indices,
                        bool decode_round,
                        const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        const std::size_t row_count = lane_indices.size();
        if (row_count == 0 || row_count != pending.row_count() || pending.row_stride() == 0 ||
            (!pending.row_counts().empty() && pending.row_counts().size() != row_count) ||
            pending.tokens().size() < static_cast<std::size_t>(pending.row_stride()) * row_count) {
            const auto discarded = instance_.program->abort_pending(std::move(pending));
            std::array<LaneId, kMaximumConcurrency> invalid_lanes{};
            for (std::size_t row = 0; row < row_count; ++row) {
                invalid_lanes[row] = LaneId{lane_indices[row]};
            }
            resources_.apply_discard(std::span<const LaneId>(invalid_lanes.data(), row_count),
                                     discarded);
            throw std::logic_error("pending batch returned an invalid ragged layout");
        }

        std::array<LaneId, kMaximumConcurrency> lanes{};
        std::array<CommitDecision, kMaximumConcurrency> decisions{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        std::array<ContinuationAction, kMaximumConcurrency> continuations{};
        std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
        std::array<bool, kMaximumConcurrency> cancelled{};
        bool generated_staged = false;
        std::array<std::shared_ptr<Request>, kMaximumConcurrency> terminal_requests{};
        std::array<std::uint32_t, kMaximumConcurrency> terminal_lanes{};
        std::array<FinishReason, kMaximumConcurrency> terminal_reasons{};
        std::size_t terminal_count = 0;
        for (std::size_t row = 0; row < row_count; ++row) {
            lanes[row] = LaneId{lane_indices[row]};
        }
        const auto rollback_generated = [&]() noexcept {
            if (!generated_staged) { return; }
            for (std::size_t row = 0; row < row_count; ++row) {
                const auto& request = slots_[lane_indices[row]];
                if (request != nullptr && request->generated.size() >= generated_sizes[row]) {
                    request->generated.resize(generated_sizes[row]);
                }
            }
            generated_staged = false;
        };
        try {
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t lane = lane_indices[row];
                const auto& request      = slots_[lane];
                if (request == nullptr || !request->sequence || !request->lane ||
                    request->lane->value != lane || !request->budget) {
                    throw std::logic_error("pending row has no active Engine request");
                }
                if (decode_round) { ++request->host_timing.decode_rounds; }
                cancelled[row] = cancelled_at_unit_start[lane];
                const std::int32_t raw_count =
                    pending.row_counts().empty() ? 1 : pending.row_counts()[row];
                if (raw_count <= 0 || raw_count > static_cast<std::int32_t>(pending.row_stride())) {
                    throw std::logic_error("pending row has an invalid licensed extent");
                }
                const std::uint32_t count = static_cast<std::uint32_t>(raw_count);
                const auto row_tokens     = pending.tokens().subspan(row * pending.row_stride(),
                                                                     static_cast<std::size_t>(count));
                generated_sizes[row]      = request->generated.size();
                if (cancelled[row]) {
                    (void)request->output.preview_terminal(FinishReason::Cancelled);
                    decisions[row] = CommitDecision{
                        .accepted_tokens = 0,
                        .terminal        = true,
                        .cancelled       = true,
                    };
                    finish_reasons[row] = FinishReason::Cancelled;
                    continue;
                }
                const OutputDecision decision = request->output.preview_model(
                    row_tokens, request->budget->remaining(), request->budget->limit_reason());
                if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                    (!decision.finished() && decision.accepted_tokens != count) ||
                    (decision.finished() && decision.continuation != ContinuationAction::Decode) ||
                    (decision.prefix_execution_split_after &&
                     (*decision.prefix_execution_split_after == 0 ||
                      *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                    throw std::logic_error("output policy returned an invalid licensed prefix");
                }
                decisions[row] = CommitDecision{
                    .accepted_tokens              = decision.accepted_tokens,
                    .terminal                     = decision.finished(),
                    .cancelled                    = false,
                    .prefix_execution_split_after = decision.prefix_execution_split_after,
                };
                finish_reasons[row] = decision.finish_reason;
                continuations[row]  = decision.continuation;
            }
            generated_staged = true;
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t accepted = decisions[row].accepted_tokens;
                if (accepted == 0) { continue; }
                const auto& request = slots_[lane_indices[row]];
                if (request->generated.size() > request->generated.capacity() ||
                    accepted > request->generated.capacity() - request->generated.size()) {
                    throw std::logic_error("admission did not reserve generated-token capacity");
                }
                const auto first = pending.tokens().begin() +
                                   static_cast<std::ptrdiff_t>(row * pending.row_stride());
                request->generated.insert(request->generated.end(), first,
                                          first + static_cast<std::ptrdiff_t>(accepted));
            }
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            rollback_generated();
            const auto discarded = instance_.program->abort_pending(std::move(pending));
            if (discarded.status == ConsumeStatus::Consumed) {
                resources_.apply_discard(std::span<const LaneId>(lanes.data(), row_count),
                                         discarded);
            } else if (!instance_.program->has_context_transaction()) {
                throw std::logic_error("Program could not abort a failed pending batch");
            }
            std::rethrow_exception(error);
        }

        std::optional<typename ModelContract::CommitResult> committed_storage;
        try {
            phase.pause_range();
            ProgramCallScope program_call(*this);
            auto committed = instance_.program->commit(
                std::move(pending), std::span<const CommitDecision>(decisions.data(), row_count),
                CommitObservation::ReleasedRowsOnly, &program_call.failed_timing());
            program_call.finish(committed.timing);
            committed_storage.emplace(std::move(committed));
            phase.resume_range();
        } catch (...) {
            rollback_generated();
            if (!instance_.program->has_context_transaction()) {
                resources_.release_failed_commit(std::span<const LaneId>(lanes.data(), row_count));
            }
            throw;
        }
        generated_staged = false;
        auto& committed  = *committed_storage;
        if (committed.row_count != row_count) {
            throw std::logic_error("Runtime commit result is not row aligned");
        }
        for (std::size_t row = 0; row < row_count; ++row) {
            const CommitDisposition expected = cancelled[row] ? CommitDisposition::CancelledReleased
                                               : decisions[row].terminal
                                                   ? CommitDisposition::Finishable
                                                   : CommitDisposition::Active;
            if (committed.rows[row].disposition != expected) {
                throw std::logic_error("Runtime commit row disposition is invalid");
            }
            if (committed.captures[row].has_value() &&
                (decode_round || expected != CommitDisposition::Active)) {
                throw std::logic_error("Runtime exposed a capture outside a committed Begin row");
            }
        }
        resources_.apply_commit(std::span<const LaneId>(lanes.data(), row_count), committed);
        const bool terminal_in_batch = std::any_of(
            decisions.begin(), decisions.begin() + static_cast<std::ptrdiff_t>(row_count),
            [](const CommitDecision& decision) { return decision.terminal; });

        for (std::size_t row = 0; row < row_count; ++row) {
            const auto& request = slots_[lane_indices[row]];
            if (cancelled[row]) {
                request->generation_timings = committed.rows[row].timings;
                request->speculative_stats  = std::move(committed.rows[row].speculative);
            }
        }

        if (decode_round) {
            ++cumulative_stats_.decode_rounds;
            cumulative_stats_.decode_row_rounds += row_count;
            for (std::size_t row = 0; row < row_count; ++row) {
                if (!cancelled[row]) {
                    cumulative_stats_.committed_decode_tokens += decisions[row].accepted_tokens;
                }
            }
        }

        try {
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t lane     = lane_indices[row];
                const auto& request          = slots_[lane];
                const std::uint32_t accepted = decisions[row].accepted_tokens;
                if (!cancelled[row]) {
                    request->budget->commit(accepted);
                    if (decode_round) { Scheduling::consume_service_work(*request, accepted); }
                }
                auto published = request->output.commit_preview();
                auto timing    = record_committed_output(request, accepted);
                append_output(request, std::move(published), std::move(timing));
                if (decisions[row].terminal) {
                    if (cancelled[row]) {
                        terminal_requests[terminal_count] = request;
                        terminal_lanes[terminal_count]    = lane;
                        terminal_reasons[terminal_count]  = finish_reasons[row];
                        ++terminal_count;
                    } else {
                        request->model_state     = EngineRequestState::ModelFinished;
                        request->terminal_reason = finish_reasons[row];
                    }
                } else if (committed.captures[row]) {
                    if (!request->is_prefilling()) {
                        throw std::logic_error("prompt-frontier capture lost its prefill owner");
                    }
                    const EngineRequestState post_capture_state =
                        continuations[row] == ContinuationAction::ApplyTargetControl
                            ? EngineRequestState::ControlReady
                            : EngineRequestState::DecodeReady;
                    if (terminal_in_batch) {
                        instance_.program->skip_capture(std::move(*committed.captures[row]));
                        request->model_state = post_capture_state;
                    } else {
                        reserve_active_capture(request, std::move(*committed.captures[row]),
                                               post_capture_state);
                    }
                    committed.captures[row].reset();
                    if (!request->capture_pending) {
                        request->model_state =
                            continuations[row] == ContinuationAction::ApplyTargetControl
                                ? EngineRequestState::ControlReady
                                : EngineRequestState::DecodeReady;
                    }
                } else {
                    request->model_state =
                        continuations[row] == ContinuationAction::ApplyTargetControl
                            ? EngineRequestState::ControlReady
                            : EngineRequestState::DecodeReady;
                }
            }
        } catch (...) {
            phase.finish();
            for (std::size_t index = 0; index < terminal_count; ++index) {
                complete_success(terminal_requests[index], terminal_reasons[index]);
                remove_completed_slot(terminal_lanes[index]);
            }
            throw;
        }
        phase.finish();
        for (std::size_t index = 0; index < terminal_count; ++index) {
            complete_success(terminal_requests[index], terminal_reasons[index]);
            remove_completed_slot(terminal_lanes[index]);
        }
    }

    // active capture（把 active 状态发布成一个可复用的 checkpoint）是全局唯一的资源事务，
    // 所以最多只允许一个请求持有它——多于一个就是所有权被破坏了。
    [[nodiscard]] std::shared_ptr<Request> active_capture_owner() const {
        std::shared_ptr<Request> request;
        for (std::uint32_t candidate = 0; candidate < max_concurrency_; ++candidate) {
            if (slots_[candidate] == nullptr || !slots_[candidate]->capture_pending) { continue; }
            if (request != nullptr) {
                throw std::logic_error("multiple requests own one active-capture transaction");
            }
            request = slots_[candidate];
            if (!request->lane || request->lane->value != candidate || !request->sequence) {
                throw std::logic_error("capture-pending request has no active sequence binding");
            }
        }
        return request;
    }

    // 启动一次 active capture。要不要做由 ResourceManager 按压力与收益决定（它可能直接跳过），
    // 而"有多少请求正等着进入"是它判断阻塞代价的输入之一。capture 期间请求照常占着 lane，
    // 只是不能参与执行单位。
    void reserve_active_capture(const std::shared_ptr<Request>& request, CaptureOffer&& offer,
                                EngineRequestState post_capture_state) {
        if (!request->lane || !request->sequence || request->capture_pending ||
            post_capture_state == EngineRequestState::Materializing ||
            post_capture_state == EngineRequestState::Waiting ||
            post_capture_state == EngineRequestState::ModelFinished) {
            throw std::logic_error("committed capture offer has invalid Engine ownership");
        }
        std::uint64_t blocked = 0;
        {
            std::lock_guard lock(queue_mutex_);
            blocked = pending_.size();
        }
        for (const auto& active : slots_) {
            if (active != nullptr && active != request && !active->terminal_reason) { ++blocked; }
        }
        const std::uint32_t blocked_runnable_requests =
            blocked > std::numeric_limits<std::uint32_t>::max()
                ? std::numeric_limits<std::uint32_t>::max()
                : static_cast<std::uint32_t>(blocked);
        const auto reserved = resources_.reserve_active_capture(
            *instance_.program, *request->lane, std::move(offer), blocked_runnable_requests,
            CancellationFlagView{&request->cancelled});
        if (reserved == ResourceManagement::ActiveCaptureReserveResult::Skipped) { return; }
        request->capture_pending    = true;
        request->post_capture_state = post_capture_state;
        (void)progress_context_transaction(false);
    }

    // ========================================================================
    // prefill
    // ========================================================================
    //
    // 同一时刻只有一个请求在 prefill（prefill_chunk 分批推进），走完前沿才算 Active。单元的输入是
    // Program 报回的 PrefillProgress，三种形态：还没跑完、要在中途做一次 capture、或者跑完了
    // 并带回 pending batch。Engine 在这里做的核心事情是**核对前沿**：已处理的 token 加上复用的
    // 前缀必须正好等于准入时承诺的提示长度，多了少了都是不变量被破坏。

    void
    resolve_prefill_progress(const std::shared_ptr<Request>& request,
                             typename ModelContract::PrefillProgress&& progress,
                             const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        ++cumulative_stats_.host_work.prefill_units;
        ++request->host_timing.prefill_units;
        cumulative_stats_.computed_prefill_tokens += progress.processed_prompt_tokens;
        Scheduling::consume_service_work(*request, 1);
        if (!request->admitted_begin) {
            throw std::logic_error("prefill progress has no committed admission summary");
        }
        const BeginSummary& begin = *request->admitted_begin;
        if (begin.reused_prompt_tokens > begin.prompt_tokens) {
            throw std::logic_error("admitted prefix exceeds its prompt");
        }
        const std::uint32_t suffix_tokens = begin.prompt_tokens - begin.reused_prompt_tokens;
        if (request->computed_prompt_tokens > suffix_tokens ||
            progress.processed_prompt_tokens > suffix_tokens - request->computed_prompt_tokens) {
            throw std::logic_error("prefill unit exceeded the admitted prompt suffix");
        }
        request->computed_prompt_tokens += progress.processed_prompt_tokens;
        if (progress.complete && request->computed_prompt_tokens != suffix_tokens) {
            throw std::logic_error("completed prefill did not reach the admitted prompt frontier");
        }
        if (progress.processed_prompt_tokens != 0) { publish_prompt_progress(request); }
        if (progress.capture) {
            if (progress.complete || progress.pending) {
                throw std::logic_error("prefill capture offer overlaps prompt completion");
            }
            reserve_active_capture(request, std::move(*progress.capture),
                                   EngineRequestState::Prefill);
            progress.capture.reset();
            return;
        }
        if (!progress.complete) { return; }
        if (!request->lane || !progress.pending) {
            throw std::logic_error("completed prefill has no lane or pending token");
        }
        if (!request->admitted_begin || progress.summary != *request->admitted_begin) {
            throw std::logic_error("runtime Begin summary differs from committed admission");
        }
        const std::uint32_t lane = request->lane->value;
        if (scheduler_.prefill_lane() == lane) {
            scheduler_.clear_prefill_lane(lane);
            request_admission_check();
        }
        request->begin = progress.summary;
        const std::array<std::uint32_t, 1> lanes{lane};
        phase.finish();
        commit_pending(std::move(*progress.pending), lanes, false, cancelled_at_unit_start);
        progress.pending.reset();
    }

    // 推进一次 prefill：只有 prefill 拥有者（scheduler_.prefill_lane()）能被调到这里，程序侧不
    // 接受"没有归属的 prefill"。
    void run_prefill_step(const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        nvtx::ScopedRange prefill_range(nvtx::Name::Prefill, nvtx::Category::Prefill);
        EnginePhaseScope setup(*this, EngineHostPhase::CommitOutput);
        const auto prefill_lane = scheduler_.prefill_lane();
        if (!prefill_lane) { throw std::logic_error("no request owns staged prefill"); }
        const std::uint32_t lane = *prefill_lane;
        const auto request       = slots_[lane];
        if (request == nullptr || !request->is_prefilling() || request->capture_pending) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        if (!request->sequence) {
            throw std::logic_error("prefill request has no sequence handle");
        }
        setup.finish();
        ProgramCallScope program_call(*this);
        auto progress =
            instance_.program->advance_prefill(*request->sequence, &program_call.failed_timing());
        program_call.finish(progress.timing);
        resolve_prefill_progress(request, std::move(progress), cancelled_at_unit_start);
        publish_runtime_stats();
    }

    // ========================================================================
    // 等待队列访问
    // ========================================================================
    //
    // pending_ 是**唯一**的入口队列：新请求只能排到队尾，没有插队、没有优先级反转，回填也只
    // 能在队首被阻塞时从队列内部挑（见 try_admit_one）。队列访问都被 queue_mutex_ 保护，因为
    // submit 来自调用方线程。

    [[nodiscard]] FifoSnapshot pending_snapshot() const {
        std::lock_guard lock(queue_mutex_);
        return Scheduling::fifo_snapshot(pending_);
    }

    [[nodiscard]] bool has_pending_requests() const {
        std::lock_guard lock(queue_mutex_);
        return !pending_.empty();
    }

    [[nodiscard]] bool erase_pending(const std::shared_ptr<Request>& request) {
        std::lock_guard lock(queue_mutex_);
        const auto it = std::find(pending_.begin(), pending_.end(), request);
        if (it == pending_.end()) { return false; }
        pending_.erase(it);
        return true;
    }

    // 让 Scheduler 知道这个等待者不会再参与准入，否则它会一直记着一份已经不存在的等待。
    void on_waiting_removed(const std::shared_ptr<Request>& request) noexcept {
        scheduler_.on_waiting_removed(request->id);
    }

    // 请求的"基础计划"只与提示本身有关（与当时有哪些邻居无关），所以算一次就缓存住，
    // 回填评估时可以反复复用。计划给出的 service work 是准入记账的基准，为 0 说明计划本身异常。
    void ensure_base_plan(const std::shared_ptr<Request>& request) {
        if (!request->base_plan) {
            request->base_plan.emplace(
                instance_.program->plan_request(request->prompt, request->options.execution));
        }
        const RequestPlanSummary& summary = request->base_plan->summary();
        if (summary.service_work_quanta == 0) {
            throw std::logic_error("target request plan has invalid admission accounting");
        }
    }

    // 问资源侧"这个请求现在能不能进来"：答案可能是 Ready / NeedsTransfer（要搬运复用上下文）
    // / 暂时不可行 / 永久不可行，并附上一个可用的选择方案。取消标记与排队期限一并传下去，让
    // 长时间的资源规划也能被打断。
    [[nodiscard]] ResourceInspection inspect_admission(const std::shared_ptr<Request>& request,
                                                       PlanningAllowance allowance) {
        allowance.cancellation = &request->cancelled;
        allowance.control_deadline_ns =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           request->deadline.time_since_epoch())
                                           .count());
        return resources_.inspect(*instance_.program, request->prompt, *request->base_plan,
                                  request->publication_order, allowance);
    }

    // 把一个还在等待的请求以错误了结：先从队列摘掉，摘得到才算数——摘不到说明它已经因为别的原因
    // 离开队列（例如刚被采纳），此时什么都不做，避免同一个请求被了结两次。
    [[nodiscard]] AdmissionProgress remove_pending_error(const std::shared_ptr<Request>& request,
                                                         std::exception_ptr error) {
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        on_waiting_removed(request);
        complete_error(request, std::move(error));
        publish_runtime_stats();
        return AdmissionProgress::ControlProgress;
    }

    // ========================================================================
    // 资源事务：materialization 与 active capture
    // ========================================================================
    //
    // 两类跨多轮的资源工作，都由 ResourceManager 驱动、由这里逐轮推进到终态。它们之所以要"跨轮"
    // 而不是一次做完，是因为中间要 GPU 参与（搬运/转换 KV），而 GPU 工作必须落在执行单位里。
    //
    //   * Materialization：为一个等待中的请求准备好它要复用的上下文，成功后请求才真正拿到 lane
    //     与 sequence，从 Waiting 变成正在 prefill 的 Active。中途被 abort 就是取消。
    //   * ActiveCapture：把一个正在跑的请求的当前状态发布成可复用的 checkpoint，成功后请求回到
    //     它的 post-capture 状态继续跑；失败只是没做成，请求不受影响。
    //
    // Engine 与 Program 必须对"当前是哪一类事务"有完全一致的看法，不一致就说明所有权已经乱了。

    [[nodiscard]] AdmissionProgress progress_context_transaction(bool yield_requested) {
        const std::optional<ContextTransactionKind> kind = resources_.context_transaction_kind();
        if (kind.has_value() != instance_.program->has_context_transaction()) {
            throw std::logic_error("Engine and Program disagree on context-transaction ownership");
        }
        if (!kind) { return AdmissionProgress::None; }
        DetailScope detail(*this, &RuntimeHostWorkStats::context_progress_ns,
                           &RuntimeHostWorkStats::context_progress_invocations,
                           nvtx::Name::ContextProgress);
        ++cumulative_stats_.host_work.control_units;

        const std::shared_ptr<Request> capture = active_capture_owner();
        std::atomic<bool> yield{yield_requested};
        CancellationFlagView cancellation{&yield};
        switch (*kind) {
        case ContextTransactionKind::Materialization: {
            if (!materializing_ || capture) {
                throw std::logic_error("materialization has conflicting Engine ownership");
            }
            const MaterializingRequest& control = *materializing_;
            const std::uint32_t lane            = control.destination.value;
            const auto& request                 = control.request;
            if (request == nullptr || !request->is_materializing() || request->lane ||
                request->sequence || request->budget || lane >= max_concurrency_ ||
                slots_[lane] != nullptr) {
                throw std::logic_error("materializing request has invalid Engine ownership");
            }
            cancellation = CancellationFlagView{&request->cancelled};
            break;
        }
        case ContextTransactionKind::ActiveCapture:
            if (materializing_ || !capture) {
                throw std::logic_error("active capture has conflicting Engine ownership");
            }
            cancellation = CancellationFlagView{&capture->cancelled};
            break;
        }

        auto outcome = resources_.progress_context_transaction(*instance_.program, cancellation);
        return std::visit(
            [&](auto&& terminal) -> AdmissionProgress {
                using Outcome = std::decay_t<decltype(terminal)>;
                if constexpr (std::is_same_v<Outcome, ContextTransactionInProgress>) {
                    return AdmissionProgress::ControlProgress;
                } else if constexpr (std::is_same_v<
                                         Outcome,
                                         typename ResourceManagement::MaterializationOutcome>) {
                    if (*kind != ContextTransactionKind::Materialization || !materializing_) {
                        throw std::logic_error(
                            "materialization outcome has no Engine control record");
                    }
                    MaterializingRequest& control = *materializing_;
                    const std::uint32_t lane      = control.destination.value;
                    const auto request            = control.request;
                    if (terminal.status == ContextTransactionStatus::Aborted) {
                        if (terminal.activation) {
                            throw std::logic_error(
                                "aborted materialization retained an activation");
                        }
                        materializing_.reset();
                        complete_detached_cancelled(request);
                        request_admission_check();
                        publish_runtime_stats();
                        return AdmissionProgress::ControlProgress;
                    }
                    if (terminal.status != ContextTransactionStatus::Published ||
                        !terminal.activation) {
                        throw std::logic_error("published materialization has no adoption token");
                    }
                    auto activation = std::move(*terminal.activation);
                    terminal.activation.reset();
                    const SequenceHandle sequence = activation.sequence();
                    resources_.adopt(*instance_.program, std::move(activation));
                    request->sequence.emplace(sequence);
                    request->budget.emplace(std::move(control.budget));
                    request->lane.emplace(control.destination);
                    request->remaining_service_work      = control.summary.service_work_quanta;
                    request->backfill_epoch              = control.protection_epoch;
                    request->backfill_class              = control.backfill_class;
                    request->materialization_diagnostics = terminal.diagnostics;
                    request->model_state                 = EngineRequestState::Prefill;
                    request->host_timing.queue_wait_ns =
                        elapsed_ns(request->submitted, Clock::now());
                    request->queue_wait_recorded = true;
                    slots_[lane]                 = request;
                    record_prefix_selection(control.summary);
                    materializing_.reset();
                    scheduler_.set_prefill_lane(lane);
                    request_admission_check();
                    publish_runtime_stats();
                    return AdmissionProgress::ControlProgress;
                } else if constexpr (std::is_same_v<
                                         Outcome,
                                         typename ResourceManagement::ActiveCaptureOutcome>) {
                    if (*kind != ContextTransactionKind::ActiveCapture || !capture) {
                        throw std::logic_error("active-capture outcome has no Engine owner");
                    }
                    if (terminal.status == ContextTransactionStatus::Published) {
                        ++cumulative_stats_.active_captures_completed;
                    } else if (terminal.status == ContextTransactionStatus::Aborted) {
                        ++cumulative_stats_.active_captures_aborted;
                    } else {
                        throw std::logic_error("active capture returned an invalid terminal state");
                    }
                    capture->capture_pending = false;
                    capture->model_state     = capture->post_capture_state;
                    request_admission_check();
                    publish_runtime_stats();
                    return AdmissionProgress::ControlProgress;
                }
                throw std::logic_error("unknown resource transaction outcome");
            },
            std::move(outcome));
    }

    // ========================================================================
    // admission：谁在何时进入执行
    // ========================================================================
    //
    // 队列是 FIFO 的，准入的基本形态就是"先看队首"：能进就进，进不去就等——但等的方式有讲究。
    //
    //   * 队首被现有 active 请求挡住时，Scheduler 会给它一个保护期，然后**从队列内部挑一个能进的
    //     先放进来**（回填）。回填不是插队：它必须由资源侧证明"放它进来不会让队首等更久"，否则
    //     队列顺序就失去了意义。
    //   * 怎么都进不来的请求（例如整张卡腾空了也装不下它的 KV）会被判定为永久不可行，直接以
    //     ContextLengthExceeded 出清，而不是让它永远排在队首把后面的人一起堵死。
    //
    // 一次准入尝试最多只放进一个请求：放进一个就可能改变资源状态，剩下的交给下一轮。

    // 采纳一个已经通过可行性检查的请求：校验 Scheduler 的 grant 仍然对应这个请求，备好预算与
    // token 容量，再让资源侧开始 materialization。走到这里请求就离开等待队列了。
    [[nodiscard]] AdmissionProgress
    admit_planned_request(const std::shared_ptr<Request>& request,
                          typename ResourceManagement::Choice&& choice, AdmissionGrant grant) {
        if (Clock::now() >= request->deadline) {
            const AdmissionProgress progress = remove_pending_error(
                request, std::make_exception_ptr(RequestError(
                             RequestErrorKind::QueueTimeout,
                             "inference request expired while waiting for admission")));
            if (progress == AdmissionProgress::ControlProgress) { request_admission_check(); }
            return progress;
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            if (!erase_pending(request)) { return AdmissionProgress::None; }
            on_waiting_removed(request);
            complete_detached_cancelled(request);
            request_admission_check();
            publish_runtime_stats();
            return AdmissionProgress::ControlProgress;
        }

        const LaneId destination         = choice.destination();
        const std::uint32_t lane         = destination.value;
        const RequestPlanSummary summary = choice.summary();
        if (grant.request_id() != request->id ||
            grant.service_work_quanta() != summary.service_work_quanta ||
            !scheduler_.validate_grant(grant)) {
            throw std::logic_error("admission choice lost its Scheduler grant");
        }
        GenerationBudget prepared_budget(summary.effective_output_tokens,
                                         summary.effective_limit_reason);
        try {
            request->generated.reserve(summary.effective_output_tokens);
        } catch (...) {
            const AdmissionProgress progress =
                remove_pending_error(request, std::current_exception());
            if (progress == AdmissionProgress::ControlProgress) { request_admission_check(); }
            return progress;
        }
        MaterializingRequest control{
            .request          = request,
            .destination      = destination,
            .budget           = std::move(prepared_budget),
            .summary          = summary,
            .backfill_class   = grant.backfill_class(),
            .protection_epoch = grant.protection_epoch(),
            .started          = Clock::now(),
        };

        const auto reserved = resources_.reserve_materialization(
            *instance_.program, std::move(choice), std::move(request->prompt),
            CancellationFlagView{&request->cancelled});
        if (reserved == ResourceManagement::MaterializationReserveResult::Stale) {
            request_admission_check();
            return AdmissionProgress::ControlProgress;
        }
        if (reserved == ResourceManagement::MaterializationReserveResult::Aborted) {
            if (!erase_pending(request)) {
                throw std::logic_error("aborted materialization lost its waiting request");
            }
            on_waiting_removed(request);
            complete_detached_cancelled(request);
            request_admission_check();
            publish_runtime_stats();
            return AdmissionProgress::ControlProgress;
        }
        if (!erase_pending(request)) {
            throw std::logic_error("admitted request disappeared from the FIFO queue");
        }
        release_planning_state(request);
        if (materializing_ || slots_[lane] != nullptr) {
            throw std::logic_error("reserved materialization destination is not empty");
        }
        request->model_state = EngineRequestState::Materializing;
        materializing_.emplace(std::move(control));
        scheduler_.commit_admission(std::move(grant));
        publish_generation_start(
            request, BeginSummary{.prompt_tokens        = summary.prompt_tokens,
                                  .reused_prompt_tokens = summary.reusable_prompt_tokens,
                                  .prefix_reuse_path    = summary.prefix_reuse_path});

        publish_runtime_stats();
        return progress_context_transaction(false);
    }

    // 准入主循环：从队首开始处理，顺带清掉等待期间被取消或过期的请求，直到放进一个请求、
    // 或者确认当前没有能进的为止。
    AdmissionProgress try_admit_one() {
        const auto other_runnable = static_cast<std::uint32_t>(
            std::count_if(slots_.begin(), slots_.end(), [](const auto& request) {
                return request && !request->capture_pending &&
                       (request->is_decode_ready() || request->is_prefilling());
            }));
        const PlanningAllowance allowance = PlanningAllowance::boundary(other_runnable);
        DetailScope detail(*this, &RuntimeHostWorkStats::admission_policy_ns,
                           &RuntimeHostWorkStats::admission_policy_invocations,
                           nvtx::Name::AdmissionPolicy);
        bool control_progress = false;
        for (;;) {
            const FifoSnapshot queued = pending_snapshot();
            if (queued.empty()) {
                scheduler_.observe_fifo_head(std::nullopt);
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }
            const std::shared_ptr<Request>& head = queued.head();
            scheduler_.observe_fifo_head(head->id);
            if (head->cancelled.load(std::memory_order_acquire)) {
                if (erase_pending(head)) {
                    on_waiting_removed(head);
                    complete_detached_cancelled(head);
                    publish_runtime_stats();
                    control_progress = true;
                }
                continue;
            }
            if (Clock::now() >= head->deadline) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::QueueTimeout,
                              "inference request expired while waiting for admission")));
                control_progress = true;
                continue;
            }

            try {
                ensure_base_plan(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            auto head_inspection = inspect_admission(head, allowance);
            if (head_inspection.readiness == Readiness::PermanentlyInfeasible) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::ContextLengthExceeded,
                              "request reservation exceeds Engine shared KV capacity")));
                control_progress = true;
                continue;
            }
            if (head_inspection.readiness == Readiness::Ready ||
                head_inspection.readiness == Readiness::NeedsTransfer) {
                if (!head_inspection.choice) {
                    throw std::logic_error("ready resource inspection has no admission choice");
                }
                AdmissionGrant grant = scheduler_.grant_head(
                    head->id, head_inspection.choice->summary().service_work_quanta);
                return admit_planned_request(head, std::move(*head_inspection.choice),
                                             std::move(grant));
            }

            //队首等待资源，暂时性阻塞。头部被挡住：保护期 + 冻结 donor
            const ActiveAdmissionSet active =
                scheduler_.active_admission_set(slots_, max_concurrency_);
            if (active.size == 0) {
                throw std::logic_error("isolated-feasible request is blocked in an idle Engine");
            }
            if (!scheduler_.protect_blocked_head(head->id, active.span(),
                                                 instance_.program->resource_revision())) {
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }
            const std::optional<std::uint64_t> protection_epoch = scheduler_.protection_epoch();
            if (!protection_epoch) {
                throw std::logic_error("blocked FIFO head has no protection epoch");
            }

            std::array<SequenceHandle, kMaximumConcurrency> persistent_borrowers{};
            std::size_t persistent_borrower_count = 0;
            for (const auto& active_request : slots_) {
                if (active_request == nullptr ||
                    active_request->backfill_class != BackfillClass::Persistent ||
                    active_request->backfill_epoch != *protection_epoch) {
                    continue;
                }
                if (!active_request->sequence) {
                    throw std::logic_error("persistent borrower has no sequence reservation");
                }
                persistent_borrowers[persistent_borrower_count++] = *active_request->sequence;
            }

            for (const std::shared_ptr<Request>& candidate : queued.backfill_candidates()) {
                if (candidate->cancelled.load(std::memory_order_acquire)) {
                    if (erase_pending(candidate)) {
                        on_waiting_removed(candidate);
                        complete_detached_cancelled(candidate);
                        publish_runtime_stats();
                        control_progress = true;
                    }
                    continue;
                }
                if (Clock::now() >= candidate->deadline) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::QueueTimeout,
                                       "inference request expired while waiting for admission")));
                    control_progress = true;
                    continue;
                }
                try {
                    ensure_base_plan(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                auto candidate_inspection = inspect_admission(candidate, allowance);
                if (candidate_inspection.readiness == Readiness::PermanentlyInfeasible) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::ContextLengthExceeded,
                                       "request reservation exceeds Engine shared KV capacity")));
                    control_progress = true;
                    continue;
                }
                if ((candidate_inspection.readiness != Readiness::Ready &&
                     candidate_inspection.readiness != Readiness::NeedsTransfer) ||
                    !candidate_inspection.choice) {
                    continue;
                }
                const auto proof = resources_.prove_persistent_backfill(
                    *instance_.program, *head->base_plan, *candidate_inspection.choice,
                    std::span<const SequenceHandle>(persistent_borrowers.data(),
                                                    persistent_borrower_count));
                if (!proof) { continue; }
                const RequestPlanSummary& candidate_plan = candidate_inspection.choice->summary();
                auto grant =
                    scheduler_.qualify_backfill(candidate->id, candidate_plan.service_work_quanta,
                                                active.span(), proof->resource_revision());
                if (grant) {
                    return admit_planned_request(candidate, std::move(*candidate_inspection.choice),
                                                 std::move(*grant));
                }
            }
            return control_progress ? AdmissionProgress::ControlProgress : AdmissionProgress::None;
        }
    }

    // ========================================================================
    // decode 与 control
    // ========================================================================
    //
    // decode 是稳态：把当前所有可解码的请求压成一个紧凑 batch 跑一轮，一轮只出一个 token/行。
    // control 是例外路径：某些时刻 Frontend 要求向模型追加一段规范 token（例如思考预算耗尽时注入
    // 提前收尾的指引），这批 token 由 Engine 定为既成事实，不经过采样。

    void run_decode_round(const RoundMembership& membership,
                          const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        nvtx::ScopedRange decode_range(nvtx::Name::Decode, nvtx::Category::Decode,
                                       static_cast<std::uint64_t>(membership.size));
        ProgramCallScope program_call(*this);
        auto pending = instance_.program->decode(
            membership.sequence_span(), membership.budget_span(), &program_call.failed_timing());
        program_call.finish(pending.execution_timing());
        commit_pending(std::move(pending), membership.lane_span(), true, cancelled_at_unit_start);
        publish_runtime_stats();
    }

    // 提交一批 control token：整行都是前端给定的规范内容，所以这里的校验是"逐行必须全额接受、
    // 且必须继续 decode"。这些 token 消耗用户可见的输出预算，但不算模型自发的思考量。
    void run_control_batch(const ControlMembership& membership) {
        nvtx::ScopedRange control_range(nvtx::Name::ControlBatch, nvtx::Category::Control,
                                        static_cast<std::uint64_t>(membership.size));
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        if (membership.empty() || membership.row_stride == 0 ||
            membership.tokens.size() !=
                static_cast<std::size_t>(membership.row_stride) * membership.size) {
            throw std::logic_error("thinking control membership is invalid");
        }

        std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
        std::array<std::optional<std::uint32_t>, kMaximumConcurrency> prefix_execution_splits{};
        bool generated_staged         = false;
        const auto rollback_generated = [&]() noexcept {
            if (!generated_staged) { return; }
            for (std::size_t row = 0; row < membership.size; ++row) {
                const auto& request = slots_[membership.lanes[row]];
                if (request != nullptr && request->generated.size() >= generated_sizes[row]) {
                    request->generated.resize(generated_sizes[row]);
                }
            }
            generated_staged = false;
        };

        for (std::size_t row = 0; row < membership.size; ++row) {
            const auto& request = slots_[membership.lanes[row]];
            if (request == nullptr) {
                throw std::logic_error("thinking control membership lost its request");
            }
            generated_sizes[row] = request->generated.size();
        }
        generated_staged = true;
        try {
            for (std::size_t row = 0; row < membership.size; ++row) {
                const std::uint32_t lane = membership.lanes[row];
                const auto& request      = slots_[lane];
                if (request == nullptr || !request->is_control_ready() ||
                    request->capture_pending || !request->budget || !request->sequence ||
                    !request->lane || request->lane->value != lane) {
                    throw std::logic_error("thinking control row lost its active request");
                }
                const std::span<const TokenId> tokens =
                    std::span<const TokenId>(membership.tokens)
                        .subspan(row * membership.row_stride, membership.row_stride);
                const OutputDecision decision =
                    request->output.preview_control(tokens, request->budget->remaining());
                if (decision.accepted_tokens != membership.row_stride || decision.finished() ||
                    decision.continuation != ContinuationAction::Decode ||
                    (decision.prefix_execution_split_after &&
                     (*decision.prefix_execution_split_after == 0 ||
                      *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                    throw std::logic_error("target control preview returned an invalid decision");
                }
                prefix_execution_splits[row] = decision.prefix_execution_split_after;
                if (request->generated.size() > request->generated.capacity() ||
                    tokens.size() > request->generated.capacity() - request->generated.size()) {
                    throw std::logic_error(
                        "admission did not reserve thinking-control token capacity");
                }
                request->generated.insert(request->generated.end(), tokens.begin(), tokens.end());
            }
            phase.pause_range();
            ProgramCallScope program_call(*this);
            const runtime::ExecutionTiming timing = instance_.program->append_forced_tokens(
                membership.sequence_span(), membership.tokens, membership.row_stride,
                std::span<const std::optional<std::uint32_t>>(prefix_execution_splits.data(),
                                                              membership.size),
                &program_call.failed_timing());
            program_call.finish(timing);
            phase.resume_range();
        } catch (...) {
            rollback_generated();
            throw;
        }
        generated_staged = false;

        ++cumulative_stats_.host_work.control_units;
        for (const std::uint32_t lane : membership.lane_span()) {
            ++slots_[lane]->host_timing.control_units;
        }

        for (std::size_t row = 0; row < membership.size; ++row) {
            const std::uint32_t lane = membership.lanes[row];
            const auto& request      = slots_[lane];
            request->budget->commit(membership.row_stride);
            Scheduling::consume_service_work(*request, membership.row_stride);
            cumulative_stats_.committed_decode_tokens += membership.row_stride;
            auto timing = record_committed_output(request, membership.row_stride);
            append_output(request, request->output.commit_preview(), std::move(timing));
            request->model_state = EngineRequestState::DecodeReady;
        }
        publish_runtime_stats();
    }

    // 引擎级失败：worker 从出错的那次操作到清理结束一直握着 execution_mutex_，因此没有任何
    // Program 侧的观测能撞见"清了一半"的物理状态。清理顺序是先终止 Program 里未决的事务、
    // 释放 active 状态，再清空 ResourceManager，最后让每个请求都收到同一个错误——请求都要有
    // 结论，不能有人被晾在那。只有"共享物理状态已经无法安全解释"才走到这里。
    void fail_all_locked(std::exception_ptr error) noexcept {
        std::deque<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.swap(pending_);
        }
        scheduler_.reset();
        const std::shared_ptr<Request> materializing_request =
            materializing_ ? materializing_->request : nullptr;
        materializing_.reset();
        instance_.program->fail_all_cleanup();
        resources_.clear_after_program_cleanup();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                complete_error(slots_[lane], error);
                slots_[lane].reset();
            }
        }
        if (materializing_request != nullptr) { complete_error(materializing_request, error); }
        for (const auto& request : pending) { complete_error(request, error); }
        publish_runtime_stats();
    }

    // ========================================================================
    // worker：唯一的 mutation owner
    // ========================================================================
    //
    // 整个核只有一个线程在改状态，循环体就是它的一轮，顺序固定：
    //
    //   边界处理 → 资源事务推进 → 终态结算 → 采样并处理取消 → 准入尝试 → 选一个执行单位执行
    //
    // "执行单位"三选一：control batch（如果 Frontend 要求）、prefill 步、或者一轮 decode。
    // 每轮只做一个，做完重新回到边界——这样取消、终态、准入都能在最细的粒度上被重新评估。
    //
    // 没有任何事可做时线程在条件变量上睡下；有活跃请求时改为 1ms 轮询，因为活跃请求的进展
    // 依赖设备而不是队列变化。
    //
    // 异常处理是终局性的：任何逃出循环体的异常都意味着不变量已经破了，走 fail_all_locked 让
    // 所有请求失败并永久退出——不做降级、不重试。

    void worker_loop() noexcept {
        bool previous_unit_was_decode = false;
        for (;;) {
            {
                // 睡眠段
                std::unique_lock lock(queue_mutex_);
                if (!stopping_ && pending_.empty()) {
                    bool active = materializing_.has_value();
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        active = active || slots_[lane] != nullptr;
                    }
                    if (!active) {
                        queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                    }
                }
                // 停机走的是与失败同一条清理路径：所有在途请求以 Unavailable 收场，然后 worker
                // 退出。调用方（Engine 析构）在外面 join。
                if (stopping_) {
                    lock.unlock();
                    const auto error = std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down"));
                    std::scoped_lock execution_lock(execution_mutex_);
                    fail_all_locked(error);
                    return;
                }
            }

            std::unique_lock execution_lock(execution_mutex_);
            try {
                set_host_work_class(HostWorkClass::Control);
                HostPhaseMeasurement boundary = begin_host_phase();
                const bool have_pending       = expire_pending_requests();
                (void)progress_context_transaction(have_pending);
                (void)settle_terminal_requests(boundary);
                const auto cancelled_at_boundary = snapshot_cancellations(); // 取消标记快照
                cancel_active_requests(cancelled_at_boundary, boundary);  // 取消已经 Active 的请求，并非立即取消
                RoundMembership membership =
                    scheduler_.build_round_membership(slots_, max_concurrency_);
                const bool admission_check_pending =
                    admission_check_pending_.load(std::memory_order_acquire);
                if (scheduler_.should_attempt_admission(
                        have_pending, admission_check_pending, !membership.empty(),
                        previous_unit_was_decode, instance_.program->has_context_transaction()) &&
                    consume_admission_check()) {
                    (void)try_admit_one();
                    membership = scheduler_.build_round_membership(slots_, max_concurrency_);
                }

                // 取消对这个执行单位只采样一次：GPU 单位在飞时到达的取消要等下一个边界才被看见，
                // 提交阶段也不会用后来读到的原子值去重新解释一个已经下发的单位。
                const auto cancelled_at_unit_start = snapshot_cancellations();
                cancel_active_requests(cancelled_at_unit_start, boundary);
                const ControlMembership control_membership =
                    scheduler_.build_control_membership(slots_, max_concurrency_);
                if (!control_membership.empty()) {
                    set_host_work_class(HostWorkClass::Control);
                    finish_engine_phase(boundary, EngineHostPhase::Boundary);
                    run_control_batch(control_membership);
                    previous_unit_was_decode = true;
                    continue;
                }
                membership = scheduler_.build_round_membership(slots_, max_concurrency_);

                bool prefill_runnable = false;
                if (const auto lane = scheduler_.prefill_lane(); lane) {
                    if (slots_[*lane] == nullptr || !slots_[*lane]->is_prefilling()) {
                        throw std::logic_error("prefill owner has no active Engine request");
                    }
                    prefill_runnable = !slots_[*lane]->capture_pending;
                }
                const ExecutionAction action = scheduler_.choose_execution(
                    !membership.empty(), prefill_runnable, previous_unit_was_decode);
                if (action == ExecutionAction::Prefill) {
                    set_host_work_class(HostWorkClass::Prefill);
                    finish_engine_phase(boundary, EngineHostPhase::Boundary);
                    run_prefill_step(cancelled_at_unit_start);
                    previous_unit_was_decode = false;
                    continue;
                }
                if (action == ExecutionAction::Decode) {
                    set_host_work_class(HostWorkClass::Decode, membership.lane_span());
                    finish_engine_phase(boundary, EngineHostPhase::Boundary);
                    run_decode_round(membership, cancelled_at_unit_start);
                    previous_unit_was_decode = true;
                    continue;
                }
                set_host_work_class(HostWorkClass::Control);
                finish_engine_phase(boundary, EngineHostPhase::Boundary);
            } catch (...) {
                const std::exception_ptr error = std::current_exception();
                HostPhaseMeasurement cleanup   = begin_host_phase();
                fail_all_locked(error);
                finish_engine_phase(cleanup, EngineHostPhase::Maintenance);
                try {
                    publish_runtime_stats();
                } catch (...) {}
                return;
            }
            execution_lock.unlock();
            std::unique_lock wait_lock(queue_mutex_);
            queue_cv_.wait_for(wait_lock, std::chrono::milliseconds(1));
        }
    }

    // ========================================================================
    // 状态
    // ========================================================================
    //
    // 按访问者分三类：只读的构造期常量；由 worker 独占、用 execution_mutex_ 划界的执行状态；
    // 以及调用方线程也会碰的队列状态（queue_mutex_）与发布出来的统计快照（stats_mutex_）。
    // 请求本身的数据则由每个 Request 自己的 mutex 保护——这是唯一允许两个线程同时接触的对象。

    Instance& instance_;  // 模型实例：Frontend / Program 的入口，生命期长于本核
    DeviceContext& device_;
    const std::uint32_t max_context_;                          // 单请求的逻辑上限
    const std::uint32_t max_concurrency_;                      // 同时在飞的请求数上限
    const std::size_t max_outstanding_;                        // 在册请求上限（active + 排队）
    const std::chrono::milliseconds pending_timeout_;          // 排队期限的默认值
    ResourceManagement resources_;                             // 上下文/资源选择的唯一 owner

    mutable std::mutex execution_mutex_;  // 串行化执行单位与需要看物理状态的观测
    mutable std::mutex queue_mutex_;      // pending_ / outstanding_ / stopping_ / failed_
    mutable std::mutex stats_mutex_;      // published_stats_
    std::condition_variable queue_cv_;    // 新请求、取消、停机都靠它叫醒 worker
    std::deque<std::shared_ptr<Request>> pending_;  // FIFO 等待队列，只从尾部进入
    std::size_t outstanding_              = 0;      // 在册请求数，两个事实都到齐才归还
    std::uint64_t next_request_id_        = 1;
    std::uint64_t next_publication_order_ = 1;  // 完成顺序的权威：谁后完成谁说了算
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};  // active lane
    std::optional<MaterializingRequest> materializing_;  // 正在做的资源事务（至多一个）
    Scheduling scheduler_;                               // 只决定谁在何时运行
    std::atomic<bool> admission_check_pending_{false};   // 合并准入可见变化，见上文
    std::uint64_t worker_accounted_elapsed_ns_ = 0;      // 已记账的嵌套时间，用于扣除重复
    HostWorkClass current_host_work_class_     = HostWorkClass::Control;
    std::array<std::uint32_t, kMaximumConcurrency> current_decode_lanes_{};
    std::size_t current_decode_lane_count_ = 0;
    RuntimeStats cumulative_stats_;  // worker 私有，只在发布时拷出去
    RuntimeStats published_stats_;   // 给 runtime_stats() 读的快照
    bool stopping_ = false;          // 停机中：worker 收尾后退出
    bool failed_   = false;          // 已经引擎级失败，单向不可恢复
    std::thread worker_;
};

} // namespace ninfer::runtime
