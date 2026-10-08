#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// ============================================================================
// 物化事务 —— 一次请求"落位"的完整流水线
//
// 输入是一份已经封印好的准入候选（AdmissionCandidate：它说清了要复用谁、要腾掉谁、要占多少资源），
// 输出是一个已经发布、可以开始预填充的请求。中间要跨过三道坎，也是本文件的三个大块：
//   1) 压力：把候选里承诺腾出的空间真正腾出来（牺牲者逐出、副本降级/丢弃、checkpoint 丢弃）；
//   2) 搬运：把要复用的状态与 KV 恢复到设备上（Host→Device 恢复、前缀 COW、状态 fork）；
//   3) 落位：建立序列状态、装采样配置，最后在唯一那个发布点交给 Runtime 的目录。
//
// 三条贯穿全文件的性质：
//   · **可中断但不可抢占**：取消只在阶段边界生效（每个阶段末尾查一次 cancel_pending）。已经物理提交的
//     部分不回滚——压力阶段丢掉的 Host 副本不会因为随后的取消而回来，剩下的只是不再往前走。
//   · **每步都可以重来**：progress 是幂等的状态机，同一个阶段被反复调用不会重复提交；每个"已提交"
//     标志（submitted / published / prepared）就是这条幂等性的支点。
//   · **事务不会半途消失**：要么走到 Published，要么走 Aborted，两者都会把牺牲者与来源的账补齐再交回。
// ============================================================================
runtime::ContextTransactionReserveStatus
ProgramImpl::reserve_materialization(AdmissionCandidate&& plan, PreparedPromptData&& prompt,
                                     runtime::CancellationFlagView cancellation) {
    // 建立事务这一段的整体职责：把候选"冻结"成事务现场——候选此后即使被外部改坏也不影响执行，
    // 事务只按封印时的样子走。这里做的校验属于"最后一刻复核"：封印之后到真正动手之间，来源序列的
    // 前缀、checkpoint、捕获身份、媒体载荷都可能已被别人动过。
    if (cancellation.requested()) { return runtime::ContextTransactionReserveStatus::Aborted; }
    const runtime::PreflightStatus preflight = revalidate_materialization(plan, prompt);
    if (preflight != runtime::PreflightStatus::Ready) {
        throw std::logic_error("materialization changed after successful preflight");
    }
    if (has_context_transaction() || pending_transaction_) {
        throw std::logic_error("Program already owns a physical transaction");
    }
    if (plan.impl_ == nullptr) {
        throw std::invalid_argument("materialization reservation is invalid");
    }

    const AdmissionCandidateImpl& details = *plan.impl_;
    const std::uint32_t lane              = details.destination.value;
    if (lane >= max_concurrency || details.destination_epoch != lane_epochs[lane] ||
        requests[lane].lifecycle != Lifecycle::Empty ||
        active_continuations[lane] < continuation_capacity) {
        throw std::logic_error("materialization activation is stale");
    }

    const SequenceState* source_state =
        details.has_source ? &continuation_states[details.source_index] : nullptr;
    const SharedPrefixState* shared_state =
        details.has_shared_source ? &shared_prefix_states[details.shared_source_index] : nullptr;
    MaterializationTransaction transaction;
    transaction.id                  = next_materialization_id_++;
    transaction.destination         = details.destination;
    transaction.has_source          = details.has_source;
    transaction.has_shared_source   = details.has_shared_source;
    transaction.source_mode         = details.source_mode;
    transaction.source_index        = details.has_source ? details.source_index : 0;
    transaction.source_generation   = details.has_source ? details.source_generation : 0;
    transaction.shared_source_index = details.has_shared_source ? details.shared_source_index : 0;
    transaction.shared_source_generation =
        details.has_shared_source ? details.shared_source_generation : 0;
    if (source_state != nullptr) {
        transaction.source_result.emplace();
        transaction.source_result->final_summary.emplace();
        transaction.source_result->final_summary->long_anchors.reserve(
            source_state->long_anchors.size());
    }
    if (shared_state != nullptr) { transaction.shared_source_result.emplace(); }
    // 把候选里的压力动作逐条物化成事务自己的工作记录（每项一个 PressureWork）。这里就把"谁是被牺牲
    // 者"以及"它当时是哪一个（下标 + 代次）"钉死：后面每一步动作前都要复核这一对，代次不符说明牺牲者
    // 已经换人，动作必须作废而不是作用到别人身上。来源自己被选成牺牲者、或同一个牺牲者被选两次，都是
    // 候选本身的错误，在这里直接报出来。
    const std::size_t victim_count        = details.pressure_options.size();
    const std::size_t shared_victim_count = details.shared_pressure_options.size();
    transaction.victim_count              = victim_count;
    transaction.victim_indices.resize(victim_count);
    transaction.victim_generations.resize(victim_count);
    transaction.victim_released.resize(victim_count, false);
    transaction.pressure.reserve(victim_count);
    transaction.pressure_results.resize(victim_count);
    transaction.shared_victim_count = shared_victim_count;
    transaction.shared_victim_indices.resize(shared_victim_count);
    transaction.shared_victim_generations.resize(shared_victim_count);
    transaction.shared_victim_released.resize(shared_victim_count, false);
    transaction.shared_pressure_results.resize(shared_victim_count);
    transaction.shared_pressure.reserve(shared_victim_count);
    if (victim_count + shared_victim_count > (std::numeric_limits<std::size_t>::max() - 3U) / 3U) {
        throw std::overflow_error("materialization transfer observation capacity overflow");
    }
    transaction.transfer_observations.reserve(3U * (victim_count + shared_victim_count) + 3U);
    const SequenceKVBundle* source_kv =
        source_state != nullptr
            ? (source_state->kv ? &*source_state->kv : nullptr)
            : (shared_state != nullptr && shared_state->kv ? &*shared_state->kv : nullptr);
    if ((source_state != nullptr || shared_state != nullptr) && source_kv == nullptr) {
        throw std::logic_error("materialization source has no KV address space");
    }
    if (source_kv != nullptr) {
        const std::uint32_t text_pages = text_kv_addresses->mapped_pages(source_kv->text);
        transaction.text_restores.reserve(text_pages);
        transaction.text_restore_destinations.reserve(text_pages);
        if (source_kv->backend) {
            const std::uint32_t backend_pages =
                backend_kv_addresses->mapped_pages(*source_kv->backend);
            transaction.backend_restores.reserve(backend_pages);
            transaction.backend_restore_destinations.reserve(backend_pages);
        }
    }
    for (std::size_t victim = 0; victim < victim_count; ++victim) {
        const std::uint32_t index      = details.pressure_indices[victim];
        const std::uint64_t generation = details.pressure_generations[victim];
        if (details.has_source && index == transaction.source_index &&
            generation == transaction.source_generation) {
            throw std::logic_error("materialization source was also selected as a victim");
        }
        for (std::size_t prior = 0; prior < victim; ++prior) {
            if (transaction.victim_indices[prior] == index &&
                transaction.victim_generations[prior] == generation) {
                throw std::logic_error("materialization victim capability is duplicated");
            }
        }
        transaction.victim_indices[victim]         = index;
        transaction.victim_generations[victim]     = generation;
        transaction.pressure_results[victim].owner = details.pressure_owner_ids[victim];
        transaction.pressure_results[victim].final_summary.emplace();
        transaction.pressure_results[victim].final_summary->long_anchors.reserve(
            continuation_states[index].long_anchors.size());
        transaction.pressure.push_back(MaterializationTransaction::PressureWork{
            .option                  = details.pressure_options[victim],
            .continuation_index      = index,
            .continuation_generation = generation,
        });
        prepare_pressure_bookkeeping(transaction.pressure.back());
    }
    for (std::size_t victim = 0; victim < shared_victim_count; ++victim) {
        const std::uint32_t index      = details.shared_pressure_indices[victim];
        const std::uint64_t generation = details.shared_pressure_generations[victim];
        if ((details.has_shared_source && index == transaction.shared_source_index &&
             generation == transaction.shared_source_generation) ||
            shared_prefix_states[index].active_references != 0) {
            throw std::logic_error("materialization shared source was also selected as a victim");
        }
        for (std::size_t prior = 0; prior < victim; ++prior) {
            if (transaction.shared_victim_indices[prior] == index &&
                transaction.shared_victim_generations[prior] == generation) {
                throw std::logic_error("materialization shared victim capability is duplicated");
            }
        }
        transaction.shared_victim_indices[victim]     = index;
        transaction.shared_victim_generations[victim] = generation;
        transaction.shared_pressure_results[victim].owner =
            details.shared_pressure_owner_ids[victim];
        transaction.shared_pressure.push_back(MaterializationTransaction::PressureWork{
            .option                  = details.shared_pressure_options[victim],
            .continuation_index      = index,
            .continuation_generation = generation,
            .shared_owner            = true,
        });
        prepare_pressure_bookkeeping(transaction.shared_pressure.back());
    }
    if (transaction.id == 0) { transaction.id = next_materialization_id_++; }

    // 根请求（没有来源，或者来源是"保留"而不是"吃掉"）需要一个新落脚点：找一个空闲的续跑槽先占上。
    // 一个都找不到时，只能指望压力阶段真的逐出某个续跑——此时把落脚点押在它身上，但要等逐出完成才
    // 能真正占位（root_waiting_for_victim），因为"逐出"本身也可能在这之前失败。
    if (!details.has_source || details.source_mode == runtime::PrivateSourceMode::Retain) {
        for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
            if (continuation_slots[index].role != ContinuationSlotRole::Free) { continue; }
            transaction.root_continuation_index = index;
            break;
        }
        if (!transaction.root_continuation_index) {
            const auto eviction =
                std::find_if(details.pressure_options.begin(), details.pressure_options.end(),
                             [](const qwen3_5::detail::PressureDecision& option) {
                                 return option.evicts_continuation;
                             });
            if (eviction == details.pressure_options.end()) {
                throw std::logic_error(
                    "preserving materialization has no continuation destination");
            }
            const std::size_t position =
                static_cast<std::size_t>(eviction - details.pressure_options.begin());
            transaction.root_continuation_index = transaction.victim_indices[position];
            transaction.root_waiting_for_victim = true;
        }
    }

    const auto host_started = Clock::now();
    transaction.plan.emplace(std::move(plan));
    AdmissionCandidateImpl& request_plan = *transaction.plan->impl_;
    RequestControl& request              = requests[lane];
    // 最后一刻复核（从候选封印到此刻之间，世界可能变了）：提示词与计划必须互相描述；复用路径要选的
    // 来源仍在且前缀仍匹配；选中的 checkpoint / 捕获身份仍然可用；要保留的改写点仍然保留得住。任何
    // 一条不成立都说明这份候选已经过期，宁可作废也不按错的前提动手。
    try {
        const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
        if (prompt_tokens != request_plan.summary.prompt_tokens ||
            (request_plan.vision.has_value() && !prompt.has_media())) {
            throw std::invalid_argument("request plan does not describe the prepared prompt");
        }
        if (prompt.identity.rewrite_checkpoint &&
            (prompt.identity.rewrite_checkpoint->frontier == 0 ||
             prompt.identity.rewrite_checkpoint->frontier > prompt_tokens)) {
            throw std::invalid_argument("prepared prompt has an invalid rewrite checkpoint");
        }
        const bool suffix_has_visual = std::any_of(
            prompt.token_types.begin() + static_cast<std::ptrdiff_t>(request_plan.reuse_base),
            prompt.token_types.end(), [](std::uint8_t type) { return type != 0; });
        if (suffix_has_visual != request_plan.vision.has_value()) {
            throw std::invalid_argument(
                "request plan does not describe the prompt suffix modality");
        }
        if (((source_state == nullptr && shared_state == nullptr) !=
             (request_plan.reuse == ReusePath::Root))) {
            throw std::logic_error("materialization source does not match the selected reuse path");
        }
        if (source_state != nullptr &&
            !qwen3_5::detail::prefix_matches(prompt, source_state->ledger,
                                             source_state->prefix_identity,
                                             request_plan.reuse_base)) {
            throw std::logic_error("planned resident prefix is no longer reusable");
        }
        if (shared_state != nullptr &&
            (!shared_state->identity || shared_state->identity->prefix_identity() == nullptr ||
             !qwen3_5::detail::prefix_matches(prompt, shared_state->identity->ledger(),
                                              *shared_state->identity->prefix_identity(),
                                              request_plan.reuse_base))) {
            throw std::logic_error("planned shared prefix is no longer reusable");
        }
        if (request_plan.reuse == ReusePath::SharedStablePrefix &&
            (!request_plan.selected_checkpoint ||
             request_plan.selected_checkpoint->kind !=
                 runtime::CheckpointKind::SharedStablePrefix ||
             request_plan.selected_checkpoint->frontier != shared_state->frontier ||
             request_plan.selected_checkpoint->ordinal != 0)) {
            throw std::logic_error("planned shared-prefix checkpoint is unavailable");
        }
        if (is_rewrite_checkpoint_restore(request_plan.reuse) &&
            (!source_state->rewrite_checkpoint.valid ||
             source_state->rewrite_checkpoint.frontier != request_plan.reuse_base ||
             request_plan.reuse != restore_path(source_state->rewrite_checkpoint.kind))) {
            throw std::logic_error("planned rewrite checkpoint is unavailable");
        }
        if (request_plan.reuse == ReusePath::PrivateLongAnchor &&
            (!request_plan.selected_checkpoint ||
             request_plan.selected_checkpoint->kind != runtime::CheckpointKind::LongAnchor ||
             std::none_of(source_state->long_anchors.begin(), source_state->long_anchors.end(),
                          [&](const LongAnchorCheckpoint& anchor) {
                              return anchor.frontier ==
                                         request_plan.selected_checkpoint->frontier &&
                                     anchor.ordinal == request_plan.selected_checkpoint->ordinal &&
                                     state_store->valid(anchor.state);
                          }))) {
            throw std::logic_error("planned long-anchor checkpoint is unavailable");
        }
        if (request_plan.rewrite_disposition == RewriteCheckpointDisposition::RetainExisting &&
            (!prompt.identity.rewrite_checkpoint || source_state == nullptr ||
             !can_retain_rewrite_checkpoint(prompt, *prompt.identity.rewrite_checkpoint,
                                            *source_state, request_plan.reuse,
                                            request_plan.reuse_base))) {
            throw std::logic_error("planned rewrite checkpoint retention is unavailable");
        }
        if (request_plan.rewrite_disposition ==
                RewriteCheckpointDisposition::ReplaceAtCommittedFrontier &&
            (!prompt.identity.rewrite_checkpoint ||
             std::none_of(request_plan.capture_groups.begin(), request_plan.capture_groups.end(),
                          [&](const CaptureGroup& group) {
                              return group.rewrite &&
                                     *group.rewrite == prompt.identity.rewrite_checkpoint->kind &&
                                     group.frontier == prompt.identity.rewrite_checkpoint->frontier;
                          }))) {
            throw std::logic_error("planned rewrite checkpoint capture is invalid");
        }
        for (const CaptureGroup& group : request_plan.capture_groups) {
            const bool base_shared_promotion = group.frontier == request_plan.reuse_base &&
                                               group.shared && !group.rewrite && !group.long_anchor;
            if (!group.identity ||
                (group.frontier <= request_plan.reuse_base && !base_shared_promotion) ||
                group.frontier > prompt_tokens ||
                group.identity->shortlist_key.frontier != group.frontier ||
                group.identity->prefix_identity() == nullptr ||
                !qwen3_5::detail::prefix_matches(prompt, group.identity->ledger(),
                                                 *group.identity->prefix_identity(),
                                                 group.frontier)) {
                throw std::logic_error("planned capture identity is invalid");
            }
        }

        if (request.prefill) {
            throw std::logic_error("free request lane retained prefill bookkeeping");
        }
        if (request_plan.vision) {
            std::vector<bool> used(prompt.media_payloads.size(), false);
            for (const VisionUseSpan& use : request_plan.vision->uses) {
                if (use.prepared_item_index >= used.size()) {
                    throw std::logic_error("Vision plan references a missing media payload");
                }
                used[use.prepared_item_index] = true;
            }
            for (std::size_t index = 0; index < used.size(); ++index) {
                if (!used[index]) {
                    prompt.media_payloads[index].reset();
                    continue;
                }
            }
            VisionPrefillPlan& vision      = *request_plan.vision;
            const std::uint32_t first_item = vision.uses.front().prepared_item_index;
            if (!vision.control_plan) {
                throw std::logic_error("Vision suffix plan has no prepared metadata");
            }
            auto control = std::make_shared<qwen3_5::VisionControl>(
                qwen3_5::build_vision_control(prompt, *vision.control_plan, first_item));
            for (VisionUseSpan& use : vision.uses) {
                if (use.prepared_item_index < first_item) {
                    throw std::logic_error("Vision suffix item order changed during admission");
                }
                use.control_index = use.prepared_item_index - first_item;
                if (use.control_index >= control->items.size()) {
                    throw std::logic_error("Vision suffix control does not cover a planned item");
                }
            }
            vision.control = std::move(control);
            vision.control_plan.reset();
        }
        if (prompt.has_media() && !request_plan.vision) { prompt.release_all_media_payloads(); }

        materialization_ledger_.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        materialization_identity_.assign(prompt);
        materialization_prefix_digests_.assign(prompt);

        const std::uint32_t initial_mtp_extent =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min({draft_window,
                            request_plan.summary.effective_output_tokens > 1
                                ? request_plan.summary.effective_output_tokens - 2
                                : 0U,
                            capacity - prompt_tokens > 0 ? capacity - prompt_tokens - 1 : 0U})
                : 0U;
        RequestControl::Prefill prefill{
            .prompt             = std::move(prompt),
            .vision_plan        = std::move(request_plan.vision),
            .vision             = nullptr,
            .capture_groups     = std::move(request_plan.capture_groups),
            .base               = request_plan.reuse_base,
            .cursor             = request_plan.reuse_base,
            .prompt_tokens      = prompt_tokens,
            .initial_mtp_extent = initial_mtp_extent,
            .elapsed_seconds    = 0.0,
            .prepare_mtp        = request_plan.prepare_mtp,
            .reuse              = request_plan.reuse,
            .mtp_bridge         = request_plan.mtp_bridge,
        };
        // 请求的预填充簿记在这里就建立（还没开始跑），物品级的视觉会话也一并建好。游标先停在 reuse_base：
        // 前缀部分靠复用，从复用点之后才开始真正算。
        request.prefill.emplace(std::move(prefill));
        if (request.prefill->vision_plan) {
            if (!workspace_plan.vision) {
                throw std::logic_error("Vision prefill has no startup workspace plan");
            }
            request.prefill->vision = std::make_unique<execution::VisionPrefillSession>(
                device, parameters,
                DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
                *workspace_plan.vision, request.prefill->prompt, *request.prefill->vision_plan,
                vision_handoff_peak_bytes);
        }
        request.prefill->elapsed_seconds =
            std::chrono::duration<double>(Clock::now() - host_started).count();
        static_assert(std::is_nothrow_move_constructible_v<MaterializationTransaction>);
        if (transaction.root_continuation_index && !transaction.root_waiting_for_victim) {
            ContinuationSlot& destination =
                continuation_slots[*transaction.root_continuation_index];
            if (destination.role != ContinuationSlotRole::Free) {
                throw std::logic_error("materialization continuation destination changed");
            }
            destination.role = ContinuationSlotRole::ReservedMaterialization;
        }
        // 事务正式开工。递增 resource_revision 是必须的：从这一刻起物理世界多了一批"已预订但还没用"
        // 的资源，之前算出的任何压力方案与封印计划都以旧版本为准，必须作废。
        advance_resource_revision();
        context_transaction_.emplace<MaterializationTransaction>(std::move(transaction));
        return runtime::ContextTransactionReserveStatus::Reserved;
    } catch (...) {
        release_materialization_staging(transaction);
        throw;
    }
}

// 事务的退场：把这次动作已经做出去的东西收回来——先中止还没走完的压力动作（其中已提交到传输流的要
// 先等流结束），再归还事务持有的全部物理预订（地址空间、状态槽、设备页预留、Host 区间）。这是
// "构造失败"与"中途取消/失败"共用的那条路径，所以它必须能容忍事务停在任何进度上。
void ProgramImpl::release_materialization_staging(
    MaterializationTransaction& transaction) noexcept {
    const std::uint32_t lane = transaction.destination.value;
    if (lane < max_concurrency && requests[lane].lifecycle == Lifecycle::Empty) {
        requests[lane].prefill.reset();
    }
    for (std::size_t position = transaction.shared_pressure_cursor;
         position < transaction.shared_pressure.size(); ++position) {
        MaterializationTransaction::PressureWork& work = transaction.shared_pressure[position];
        if (work.submitted) {
            try {
                context_completion_.synchronize();
            } catch (...) { std::terminate(); }
        }
        abort_pressure_work(work);
    }

    for (std::size_t position = transaction.pressure_cursor; position < transaction.pressure.size();
         ++position) {
        MaterializationTransaction::PressureWork& work = transaction.pressure[position];
        if (work.submitted) {
            try {
                context_completion_.synchronize();
            } catch (...) { std::terminate(); }
        }
        abort_pressure_work(work);
    }

    abort_materialization_transfers(transaction);
    transaction.backend_retained_tail_backup.reset();
    transaction.text_retained_tail_backup.reset();
    transaction.backend_retained_tail.reset();
    transaction.text_retained_tail.reset();
    transaction.backend_prefix_fork.reset();
    transaction.text_prefix_fork.reset();
    transaction.backend_source_restore_reservation.reset();
    transaction.text_source_restore_reservation.reset();
    transaction.backend_activation.reset();
    transaction.text_activation.reset();
    if (transaction.root_backend_address && backend_kv_addresses) {
        (void)backend_kv_addresses->release(*transaction.root_backend_address);
        transaction.root_backend_address.reset();
    }
    if (transaction.root_text_address && text_kv_addresses) {
        (void)text_kv_addresses->release(*transaction.root_text_address);
        transaction.root_text_address.reset();
    }
    if (transaction.state_fork_destination) {
        if (state_store) { (void)state_store->release(*transaction.state_fork_destination); }
        transaction.state_fork_destination.reset();
    }
    for (std::size_t index = 0; index < transaction.reserved_state_count; ++index) {
        if (state_store) { (void)state_store->release(transaction.reserved_states[index]); }
        transaction.reserved_states[index] = {};
    }
    transaction.reserved_state_count = 0;

    if (transaction.root_continuation_index) {
        const std::uint32_t index = *transaction.root_continuation_index;
        if (index < continuation_capacity &&
            continuation_slots[index].role == ContinuationSlotRole::ReservedMaterialization) {
            release_continuation_slot_best_effort(index);
        }
        transaction.root_continuation_index.reset();
    }
    transaction.prepared                       = false;
    transaction.prefix_tail_submitted          = false;
    transaction.retained_tail_backup_submitted = false;
    transaction.prefix_forks_ready             = false;
    materialization_ledger_.clear();
    materialization_identity_.clear();
    materialization_prefix_digests_.clear();
}

// "吃掉来源"（ConsumeToActive）的语义：新请求直接继承来源的落脚点，来源序列本身退化成"只到
// reuse_base"。于是要把 reuse_base 之后的一切从来源上摘掉——状态镜像、长锚点、改写 checkpoint、
// KV 尾段——而 reuse_base 之前的必须原样留下（正是新请求要复用的那部分）。
//
// 这里的每一步都在释放"原本会被算作来源独占资源"的东西，所以最后要拿结果跟候选当初承诺的
// demand.final_removed 对账：对不上说明释放的范围与承诺不符，必须报出来而不是默默接受。
void ProgramImpl::prepare_consumed_source(MaterializationTransaction& transaction) {
    if (transaction.source_prepared || !transaction.plan || transaction.plan->impl_ == nullptr) {
        throw std::logic_error("materialization source preparation state is invalid");
    }
    transaction.source_prepared           = true;
    const AdmissionCandidateImpl& details = *transaction.plan->impl_;
    if (!transaction.has_source ||
        details.source_mode != runtime::PrivateSourceMode::ConsumeToActive) {
        return;
    }
    if (transaction.source_index >= continuation_capacity ||
        continuation_slots[transaction.source_index].role != ContinuationSlotRole::Catalogued ||
        continuation_slots[transaction.source_index].generation != transaction.source_generation) {
        throw std::logic_error("materialization source changed before dependency release");
    }
    SequenceState& source = continuation_states[transaction.source_index];
    if (!source.kv || details.reuse == ReusePath::Root ||
        details.reuse == ReusePath::SharedStablePrefix) {
        throw std::logic_error("consumed materialization source is incomplete");
    }

    const detail::PhysicalResources before = owner_exclusive_resources(source);
    const auto retained_state              = [&](StateImageHandle handle) {
        if (source.endpoint_valid && source.state.read == handle) { return true; }
        if (source.rewrite_state && *source.rewrite_state == handle) { return true; }
        return std::any_of(
            source.long_anchors.begin(), source.long_anchors.end(),
            [&](const LongAnchorCheckpoint& anchor) { return anchor.state == handle; });
    };
    const auto release_if_unreferenced = [&](StateImageHandle handle) {
        if (!state_store->valid(handle) || retained_state(handle) ||
            state_store->checkpoint_references(handle) != 0) {
            return;
        }
        if (!state_store->release(handle)) {
            throw std::logic_error("superseded source StateImage remained pinned");
        }
    };

    // 只有"不再被别处引用"的镜像才真能释放：端点在用、改写点在用、某个长锚点在用，或者还有别的
    // checkpoint 引用着它，都要留着。释放顺序是从后往前摘锚点，这样被摘掉的不会影响还在的。
    if (source.endpoint_valid && source.execution_frontier > details.reuse_base) {
        const StateImageHandle endpoint = source.state.read;
        source.endpoint_valid           = false;
        source.state                    = {};
        source.tail_hidden              = {};
        source.tail_hidden_valid        = false;
        release_if_unreferenced(endpoint);
    }
    for (std::size_t index = source.long_anchors.size(); index != 0; --index) {
        LongAnchorCheckpoint& anchor = source.long_anchors[index - 1U];
        if (anchor.frontier <= details.reuse_base) { continue; }
        const StateImageHandle state = anchor.state;
        state_store->release_checkpoint_reference(state);
        source.long_anchors.erase(source.long_anchors.begin() +
                                  static_cast<std::ptrdiff_t>(index - 1U));
        release_if_unreferenced(state);
    }
    if (details.reuse == ReusePath::PrivateEndpoint &&
        details.rewrite_disposition != RewriteCheckpointDisposition::RetainExisting &&
        source.rewrite_state) {
        const StateImageHandle rewrite = *source.rewrite_state;
        state_store->release_checkpoint_reference(rewrite);
        source.rewrite_state.reset();
        source.rewrite_checkpoint        = {};
        source.rewrite_checkpoint_hidden = {};
        release_if_unreferenced(rewrite);
    }

    struct TruncateTarget {
        KVAddressSpaceStore* addresses = nullptr;
        LogicalKVPageStore* pages      = nullptr;
        KVAddressSpaceHandle address;
        std::uint32_t frontier        = 0;
        bool prefix_fork              = false;
        bool releases_stale_host_tail = false;
    };

    std::array<TruncateTarget, 2> targets{};
    std::size_t target_count = 0;
    targets[target_count++]  = TruncateTarget{
         .addresses   = text_kv_addresses.get(),
         .pages       = text_kv_pages.get(),
         .address     = source.kv->text,
         .frontier    = details.reuse_base,
         .prefix_fork = details.text_prefix_fork_required,
    };
    if (source.kv->backend) {
        targets[target_count++] = TruncateTarget{
            .addresses   = backend_kv_addresses.get(),
            .pages       = backend_kv_pages.get(),
            .address     = *source.kv->backend,
            .frontier    = backend_frontier_at(speculative_backend, details.reuse_base),
            .prefix_fork = details.backend_prefix_fork_required,
        };
    }

    // KV 侧的截断有两种形态，取决于新请求要怎么用这段前缀：走 CoW（prefix_fork）时源序列的页不动，
    // 只把地址空间缩到 frontier；否则是破坏性截断，真把页交回去。破坏性截断前要先处理"尾页里超出
    // frontier 的那半页"——如果 Host 上存着不同列数的旧副本，得先把它一并释放，否则 Host 与设备两侧
    // 对同一页的记录会不一致。
    std::array<HostKVPageReplicaRelease, 2> host_tail_releases{};
    std::size_t host_tail_release_count = 0;
    for (TruncateTarget& target : std::span(targets.data(), target_count)) {
        if (target.prefix_fork) {
            if (!target.addresses->can_truncate_inactive_prefix(target.address, target.frontier)) {
                throw std::logic_error("COW source KV suffix is not releasable");
            }
            continue;
        }
        const std::uint32_t target_pages = kv_pages_for_frontier(target.frontier);
        if (target_pages != 0) {
            const LogicalKVPageHandle tail =
                target.addresses->logical_page(target.address, target_pages - 1U);
            const std::uint32_t columns =
                target.frontier -
                (target_pages - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            target.releases_stale_host_tail = columns != target.pages->committed_columns(tail) &&
                                              target.pages->host_resident(tail);
            if (target.releases_stale_host_tail) {
                if (!host_kv_extents || host_tail_release_count == host_tail_releases.size()) {
                    throw std::logic_error("stale source Host KV tail is not releasable");
                }
                host_tail_releases[host_tail_release_count++] =
                    HostKVPageReplicaRelease{.pages = target.pages, .page = tail};
            }
        }
        if (!target.addresses->can_destructive_truncate_inactive(target.address, target.frontier,
                                                                 target.releases_stale_host_tail)) {
            throw std::logic_error("consumed source KV is not destructively truncatable");
        }
    }
    if (host_tail_release_count != 0) {
        const std::span<const HostKVPageReplicaRelease> releases(host_tail_releases.data(),
                                                                 host_tail_release_count);
        if (!host_kv_extents->release_page_replicas(releases)) {
            throw std::logic_error("stale source Host KV tails cannot be released atomically");
        }
    }
    for (TruncateTarget& target : std::span(targets.data(), target_count)) {
        if (target.prefix_fork) {
            target.addresses->truncate_inactive_prefix(target.address, target.frontier);
        } else {
            target.addresses->destructive_truncate_inactive(target.address, target.frontier);
        }
        target.addresses->set_checkpoint_requirement(target.address, target.frontier);
    }
    // 三个后端各自的 KV 进度也要跟着退回 reuse_base：地址空间被截了，进度还停在原处就会让后续按
    // "已经算过的位置"去读已经不存在的页。
    source.text_kv_valid = details.reuse_base;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        source.mtp_kv_valid = backend_frontier_at(speculative_backend, details.reuse_base);
    } else if (is_masked_draft_backend(speculative_backend)) {
        source.dflash_context_frontier = details.reuse_base;
    }
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
    refresh_state_views(source);

    const detail::PhysicalResources after   = owner_exclusive_resources(source);
    const detail::PhysicalResources removed = checked_resource_difference(before, after);
    (void)checked_resource_difference(details.demand.final_removed, removed);
}

// 设备侧的准备：压力阶段已经收尾、牺牲者都已释放之后才进来。这里做三件事，顺序上互相嵌套——
//   1) 状态镜像：按来源与复用路径决定"继承同一份（fork）"还是"另开一份（分配）"，其中 Host-only 的
//      来源还要先把镜像搬回设备；所有设备槽位都是"先预订、再激活/搬运、最后结算或中止"。
//   2) KV 地址空间：根请求新建一条，复用来源的则把来源的地址空间"激活"到本 lane；随后把激活前仍缺的
//      设备页从 Host 副本补回来（记成恢复清单，交给后面的传输阶段真正拷贝）。
//   3) 状态搬移：需要恢复的那一份状态在这里就发起到传输流上。
// 走前缀 fork（CoW）的分支不在此列——它的目标是另一条地址空间，交给 prepare_prefix_forks。
void ProgramImpl::prepare_materialization(MaterializationTransaction& transaction) {
    if (transaction.prepared || !transaction.plan ||
        transaction.destination.value >= max_concurrency ||
        !requests[transaction.destination.value].prefill || !transaction.source_prepared) {
        throw std::logic_error("materialization preparation state is invalid");
    }
    for (std::size_t victim = 0; victim < transaction.victim_count; ++victim) {
        if (!transaction.victim_released[victim]) {
            throw std::logic_error("materialization preparation has an unreleased victim");
        }
    }

    const auto prepare_started            = Clock::now();
    const AdmissionCandidateImpl& details = *transaction.plan->impl_;
    const detail::PhysicalDemand& demand  = details.demand;
    const std::uint32_t lane              = transaction.destination.value;
    if (transaction.has_source &&
        (transaction.source_index >= continuation_capacity ||
         continuation_slots[transaction.source_index].role != ContinuationSlotRole::Catalogued ||
         continuation_slots[transaction.source_index].generation !=
             transaction.source_generation)) {
        throw std::logic_error("materialization source changed during capacity preparation");
    }
    if (transaction.has_shared_source &&
        (transaction.shared_source_index >= shared_prefix_capacity ||
         shared_prefix_slots[transaction.shared_source_index].role !=
             SharedPrefixSlotRole::Catalogued ||
         shared_prefix_slots[transaction.shared_source_index].generation !=
             transaction.shared_source_generation)) {
        throw std::logic_error("materialization shared source changed during capacity preparation");
    }
    SequenceState* source_state =
        transaction.has_source ? &continuation_states[transaction.source_index] : nullptr;
    SharedPrefixState* shared_state = transaction.has_shared_source
                                          ? &shared_prefix_states[transaction.shared_source_index]
                                          : nullptr;
    // state_count 是这次要给设备预留的状态槽数，直接取自候选当初承诺的量。下面几种情形会各自减一
    // 并单独处理，因为它们的槽位有特殊来路（Host 恢复要占一格、fork 的目的地要另开、Both 分裂要一格）。
    std::uint32_t state_count       = demand.reservation_added.device.state_slots;
    std::optional<StateImageHandle> host_state_restore;
    std::optional<StateImageHandle> host_state_fork_destination;
    if (source_state != nullptr || shared_state != nullptr) {
        const StateImageHandle state =
            source_state != nullptr
                ? selected_state(*source_state, details.reuse, details.selected_checkpoint)
                : shared_state->state;
        const StateReplicaResidency residency = state_store->residency(state);
        // "来源存在"是 StateImageStore（镜像库）的事实，不看资源计数：一份合法分配可以被 private 与
        // shared 两个检查点别名，此时它对任何一方都不是独占资源。
        if (state_store->role(state) != StateImageRole::CheckpointImmutable ||
            residency == StateReplicaResidency::None) {
            throw std::logic_error("materialization source has no published StateImage replica");
        }
        // ConsumeToActive + 需要 fork：来源槽位马上就要变成本 lane 的活动状态，而镜像还得换一份新身份，
        // 否则两条 lane 会共享同一份可变状态。
        const bool consuming_fork =
            source_state != nullptr &&
            details.source_mode == runtime::PrivateSourceMode::ConsumeToActive &&
            details.state_fork_required;
        // 走法一：Host-only。设备上还没有这份镜像，得先把 Host 副本搬回来；搬运在函数末尾才发起，
        // 这里先占掉一格预订并定好目的地。
        if (residency == StateReplicaResidency::HostOnly) {
            host_state_restore = state;
            if (state_count == 0) {
                throw std::logic_error("Host StateImage restore has no Device reservation");
            }
            --state_count;
            // Retain 与 consuming_fork 都是"本 lane 另拿一份身份"：Retain 留下原身份继续给来源用，
            // consuming_fork 则是把来源身份整个接管过来。只预订逻辑身份——物理内容由 Host 副本补上。
            if (details.source_mode == runtime::PrivateSourceMode::Retain || consuming_fork) {
                std::optional<StateImageHandle> destination =
                    state_store->reserve_logical_destination();
                if (!destination) { throw std::bad_alloc(); }
                if (consuming_fork) {
                    transaction.state_fork_destination = *destination;
                } else {
                    transaction.reserved_states[transaction.reserved_state_count++] = *destination;
                }
                host_state_fork_destination = *destination;
            }
        } else if (consuming_fork) {
            // 走法二：镜像已在设备上且来源要被本 lane 接管——直接把来源镜像的身份 Fork 给本 lane，
            // 免去一次设备内拷贝。
            if (state_count == 0) {
                throw std::logic_error("StateImage Fork has no Device reservation");
            }
            --state_count;
            transaction.state_fork_destination = state_store->reserve_destination();
            if (!transaction.state_fork_destination) { throw std::bad_alloc(); }
        } else if (source_state != nullptr &&
                   details.source_mode == runtime::PrivateSourceMode::Retain &&
                   residency == StateReplicaResidency::Both) {
            // 走法三：Both + Retain——来源同时有设备与 Host 两份副本，本 lane 只需要设备那一份，于是把
            // 这次分配"分裂"成两个身份：来源继续持有 Host 副本，新身份持有设备副本，不额外拷贝。
            if (state_count == 0) {
                throw std::logic_error("Both StateImage split has no active destination");
            }
            --state_count;
            std::optional<StateImageHandle> destination =
                state_store->reserve_logical_destination();
            if (!destination) { throw std::bad_alloc(); }
            transaction.reserved_states[transaction.reserved_state_count++] = *destination;
            transaction.split_state_identity                                = true;
        }
    }
    // 剩下的槽位没有特殊来路，按需预订即可。reserved_states 的容量是构造期就切好的，这里只断言"没有超账"。
    if (state_count > transaction.reserved_states.size() - transaction.reserved_state_count) {
        throw std::logic_error("materialization state reservation exceeds the active contract");
    }
    for (std::uint32_t index = 0; index < state_count; ++index) {
        std::optional<StateImageHandle> state = state_store->reserve_destination();
        if (!state) { throw std::bad_alloc(); }
        transaction.reserved_states[transaction.reserved_state_count++] = *state;
    }
    // 根请求（无任何来源）：镜像直接"激活为清零"，第一次前向从空状态开始。它的槽位在容量预留阶段就被
    // 占定，必须是清单里的第一个。
    if (!transaction.has_source && !transaction.has_shared_source) {
        if (!transaction.root_continuation_index || transaction.root_waiting_for_victim ||
            continuation_slots[*transaction.root_continuation_index].role !=
                ContinuationSlotRole::ReservedMaterialization ||
            transaction.reserved_state_count == 0) {
            throw std::logic_error("root materialization destination is not reserved");
        }
        state_store->activate_reset(transaction.reserved_states[0], device.stream);
    }

    // 接下来是 KV 地址空间。地址空间描述符与其中的页是两回事：这里只决定"本 lane 用哪条地址空间、
    // 要不要新建一条"，页级别的激活与补页在下面。
    KVAddressSpaceHandle text_address;
    std::optional<KVAddressSpaceHandle> backend_address;
    // Retain 的来源仍需持有它原来的地址空间；ConsumeToActive 则是把来源那条整个交给本 lane（详见下面的
    // activation）。
    const bool retained_source = (source_state != nullptr || shared_state != nullptr) &&
                                 details.source_mode == runtime::PrivateSourceMode::Retain;
    if (source_state != nullptr || shared_state != nullptr) {
        const SequenceKVBundle* source_kv = source_state != nullptr
                                                ? (source_state->kv ? &*source_state->kv : nullptr)
                                                : (shared_state->kv ? &*shared_state->kv : nullptr);
        if (source_kv == nullptr) {
            throw std::logic_error("materialization source has no KV address space");
        }
        text_address    = source_kv->text;
        backend_address = source_kv->backend;
        // 来源要保留（Retain）或要走前缀 fork 时，本 lane 需要一条新的、尚未激活的地址空间；否则直接
        // 沿用来源那条（靠下面的 activation 把它挂到本 lane 名下）。
        if (retained_source || details.text_prefix_fork_required) {
            transaction.root_text_address = text_kv_addresses->create_inactive();
            if (!transaction.root_text_address) {
                throw std::logic_error("Text KV prefix-fork destination is unavailable");
            }
        }
        if (backend_address && (retained_source || details.backend_prefix_fork_required)) {
            transaction.root_backend_address = backend_kv_addresses->create_inactive();
            if (!transaction.root_backend_address) {
                throw std::logic_error("Backend KV prefix-fork destination is unavailable");
            }
        }
    } else {
        // 根请求：新建地址空间。它必须自带 Text KV，否则下面与 entitlement 的对账会失败。
        transaction.root_text_address = text_kv_addresses->create_inactive();
        if (!transaction.root_text_address) {
            throw std::logic_error("root Text KV address descriptor is unavailable");
        }
        text_address = *transaction.root_text_address;
        if (details.backend_kv_page_entitlement != 0) {
            if (!backend_kv_addresses) {
                throw std::logic_error("root Backend KV store is unavailable");
            }
            transaction.root_backend_address = backend_kv_addresses->create_inactive();
            if (!transaction.root_backend_address) {
                throw std::logic_error("root Backend KV address descriptor is unavailable");
            }
            backend_address = *transaction.root_backend_address;
        }
    }
    if (details.text_kv_page_entitlement == 0 ||
        backend_address.has_value() != (details.backend_kv_page_entitlement != 0)) {
        throw std::logic_error("materialization KV addresses do not match their entitlements");
    }

    // 激活的边界就是复用点 reuse_base：把地址空间里 0..frontier 这一段划给本 lane，边界内缺的设备页
    // 之后再从 Host 副本补回来。后端 KV 的进度比 Text KV 差一格——MTP 在后端缓存里记的是"下一个"
    // 位置。
    if (source_state != nullptr || shared_state != nullptr) {
        transaction.text_activation_frontier = details.reuse_base;
        if (backend_address) {
            transaction.backend_activation_frontier =
                speculative_backend == SpeculativeBackend::Mtp && details.reuse_base != 0
                    ? details.reuse_base - 1U
                    : details.reuse_base;
        }
    }

    const bool text_prefix_fork =
        (source_state != nullptr || shared_state != nullptr) && details.text_prefix_fork_required;
    const bool backend_prefix_fork = (source_state != nullptr || shared_state != nullptr) &&
                                     details.backend_prefix_fork_required;
    // 两条互斥的路：要走前缀 fork 的，本 lane 的目标是另一条地址空间，这里只留一个空页预订位，等
    // prepare_prefix_forks 把尾页 CoW 完成后再填；不走 fork 的，直接对目标地址空间做 activation——
    // 登记 lane 的活跃引用，并声明它允许持有多少设备页（entitlement）。
    if (text_prefix_fork) {
        transaction.text_source_restore_reservation.emplace(
            text_kv_pages->physical_pool().make_empty_reservation());
    } else {
        const KVAddressSpaceHandle activation_address =
            retained_source ? *transaction.root_text_address : text_address;
        transaction.text_activation.emplace(text_kv_addresses->prepare_activation(
            activation_address, details.text_kv_page_entitlement, static_cast<std::int32_t>(lane),
            transaction.text_activation_frontier));
    }
    if (backend_address && backend_prefix_fork) {
        transaction.backend_source_restore_reservation.emplace(
            backend_kv_pages->physical_pool().make_empty_reservation());
    } else if (backend_address) {
        const KVAddressSpaceHandle activation_address =
            retained_source ? *transaction.root_backend_address : *backend_address;
        transaction.backend_activation.emplace(backend_kv_addresses->prepare_activation(
            activation_address, details.backend_kv_page_entitlement,
            static_cast<std::int32_t>(lane), transaction.backend_activation_frontier));
    }

    // 把激活边界内"设备上还缺的页"整理成一份恢复清单：只登记（逻辑页 → Host extent 里的副本位置 →
    // 目标设备页句柄），真正的拷贝留给 enqueue_materialization_transfers。
    const auto prepare_kv_restores =
        [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages, KVAddressSpaceHandle address,
            std::optional<std::uint32_t> activation_frontier, bool source_reservation,
            DeviceKVPageReservation& reservation,
            std::vector<MaterializationTransaction::KVRestorePage>& restores,
            std::vector<DeviceKVPageHandle>& destinations) {
            const std::uint32_t mapped = activation_frontier
                                             ? kv_pages_for_frontier(*activation_frontier)
                                             : addresses.mapped_pages(address);
            if (mapped > addresses.mapped_pages(address)) {
                throw std::logic_error("KV activation frontier exceeds address membership");
            }
            std::uint32_t missing = 0;
            for (std::uint32_t page = 0; page < mapped; ++page) {
                if (!pages.device_resident(addresses.logical_page(address, page))) { ++missing; }
            }
            // fork 分支的空预订位在这里才长到实际缺口大小；activation 分支的预订位由地址空间自己持有。
            if (source_reservation) {
                pages.physical_pool().resize_reservation(reservation, missing);
            }
            for (std::uint32_t page = 0; page < mapped; ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                if (pages.device_resident(logical)) { continue; }
                if (!pages.host_resident(logical) || !host_kv_extents) {
                    throw std::logic_error("checkpoint KV page has no restorable replica");
                }
                const HostKVPageReplica replica = pages.host_replica(logical);
                const DeviceKVPageHandle destination =
                    pages.reserve_device_replica(logical, reservation);
                restores.push_back(MaterializationTransaction::KVRestorePage{
                    .logical     = logical,
                    .extent      = replica.extent,
                    .extent_page = replica.page_offset,
                });
                destinations.push_back(destination);
            }
        };
    DeviceKVPageReservation& text_restore_reservation =
        text_prefix_fork ? *transaction.text_source_restore_reservation
                         : text_kv_addresses->page_reservation(*transaction.text_activation);
    prepare_kv_restores(*text_kv_addresses, *text_kv_pages, text_address,
                        transaction.text_activation_frontier, text_prefix_fork,
                        text_restore_reservation, transaction.text_restores,
                        transaction.text_restore_destinations);
    if (backend_address) {
        DeviceKVPageReservation& backend_restore_reservation =
            backend_prefix_fork
                ? *transaction.backend_source_restore_reservation
                : backend_kv_addresses->page_reservation(*transaction.backend_activation);
        prepare_kv_restores(*backend_kv_addresses, *backend_kv_pages, *backend_address,
                            transaction.backend_activation_frontier, backend_prefix_fork,
                            backend_restore_reservation, transaction.backend_restores,
                            transaction.backend_restore_destinations);
    }
    // Host 镜像的搬回在这里真正发起到传输流上：目的地若是 fork 出来的新身份就走 host_fork（一份 Host
    // 副本变成两份），否则是普通的 H2D。计时按 State 类记账，掩码留给后面的观测汇总。
    if (host_state_restore) {
        start_context_transfer_timer(runtime::ContextResourceClass::State);
        std::optional<StateImageTransfer> restore =
            host_state_fork_destination
                ? state_store->begin_host_fork(*host_state_restore, *host_state_fork_destination,
                                               device.transfer_stream)
                : state_store->begin_host_to_device(*host_state_restore, device.transfer_stream);
        if (!restore) { throw std::bad_alloc(); }
        transaction.state_restore.emplace(std::move(*restore));
        stop_context_transfer_timer(runtime::ContextResourceClass::State);
        transaction.transfer_timer_mask |=
            1U << context_resource_index(runtime::ContextResourceClass::State);
    }
    // 到这里"设备侧的预订"全部就位：镜像槽位、地址空间、恢复清单、状态搬移都已登记，只等传输阶段把它们
    // 变成真实数据。prepared 之后本函数不会重入（上面已断言）。
    transaction.prepared = true;
    requests[lane].prefill->elapsed_seconds +=
        std::chrono::duration<double>(Clock::now() - prepare_started).count();
}

// 前缀 fork（写时复制）：来源地址空间原封不动，本 lane 新建一条地址空间，把 0..frontier 的**整页**
// 共享过去（引用计数 +1 而已，不拷贝数据）；只有 frontier 落在页中间时，那一页尾页必须真复制一份，
// 让两条地址空间各自持有——否则两边写同一页会互相污染。若计划还要求"留一份来源尾页的副本再释放来源
// 的设备页"（retained tail release），要先把尾页备份到 Host（若尚无 Host 副本）。
//
// 这里把"发起的活"和"完成的活"分开：尾页复制是异步发到传输流上的，发完本函数就带着
// prefix_tail_submitted 退出，等发布阶段同步、确认之后再回来收尾（置 prefix_forks_ready）。因此它从
// 两个地方被调用——没有任何传输要发时（enqueue 的收尾），以及传输发布之后（publish 的收尾）。
void ProgramImpl::prepare_prefix_forks(MaterializationTransaction& transaction) {
    if (!transaction.plan || transaction.plan->impl_ == nullptr ||
        (transaction.has_source == transaction.has_shared_source) ||
        (transaction.has_source && transaction.source_index >= continuation_capacity) ||
        (transaction.has_shared_source &&
         transaction.shared_source_index >= shared_prefix_capacity) ||
        transaction.prefix_forks_ready || transaction.prefix_tail_submitted) {
        throw std::logic_error("prefix fork preparation is invalid");
    }
    const AdmissionCandidateImpl& details = *transaction.plan->impl_;
    if ((!details.text_prefix_fork_required && !details.backend_prefix_fork_required) ||
        (details.text_prefix_fork_required &&
         (!transaction.root_text_address || transaction.text_prefix_fork)) ||
        (details.backend_prefix_fork_required &&
         (!transaction.root_backend_address || transaction.backend_prefix_fork))) {
        throw std::logic_error("planned prefix fork destinations are incomplete");
    }
    const SequenceKVBundle* source_kv =
        transaction.has_source ? (continuation_states[transaction.source_index].kv
                                      ? &*continuation_states[transaction.source_index].kv
                                      : nullptr)
                               : (shared_prefix_states[transaction.shared_source_index].kv
                                      ? &*shared_prefix_states[transaction.shared_source_index].kv
                                      : nullptr);
    if (source_kv == nullptr || !transaction.text_activation_frontier) {
        throw std::logic_error("prefix fork source is incomplete");
    }
    if (transaction.text_source_restore_reservation &&
        transaction.text_source_restore_reservation->pages() != 0) {
        throw std::logic_error("retained Text KV restores are incomplete");
    }
    if (transaction.backend_source_restore_reservation &&
        transaction.backend_source_restore_reservation->pages() != 0) {
        throw std::logic_error("retained Backend KV restores are incomplete");
    }
    // 要释放来源尾页的设备副本，就必须先保证它还有别的落脚点：没有 Host 副本的话，在这里为它预订一块
    // Host extent（拷贝动作在发布阶段做）。同时确认这一页此刻确实只被本 fork 引用、没有别的写者。
    const auto prepare_retained_tail_backup = [&](KVAddressSpaceStore& addresses,
                                                  LogicalKVPageStore& pages,
                                                  KVPrefixForkReservation& fork, bool staged,
                                                  std::optional<LogicalKVPageHandle>& retained_tail,
                                                  std::optional<HostKVExtentReservation>& backup) {
        if (!staged) { return; }
        const LogicalKVPageHandle tail = addresses.prefix_fork_tail_logical_source(fork);
        if (pages.address_references(tail) != 1 || !pages.device_resident(tail) ||
            pages.writer_references(tail) != 0) {
            throw std::logic_error("retained KV tail changed before staged release");
        }
        retained_tail = tail;
        if (pages.host_resident(tail)) { return; }
        if (host_kv_extents == nullptr) {
            throw std::logic_error("retained KV tail has no Host extent store");
        }
        const std::array membership{tail};
        std::optional<HostKVExtentReservation> reserved =
            host_kv_extents->prepare(pages, membership);
        if (!reserved) { throw std::bad_alloc(); }
        backup.emplace(std::move(*reserved));
    };
    // 预订位到此交棒：不再需要"从 Host 恢复来源页"，改由 fork 自己的预订承担本 lane 要新增的页。
    bool copied_tail = false;
    if (details.text_prefix_fork_required) {
        transaction.text_source_restore_reservation.reset();
        transaction.text_prefix_fork.emplace(text_kv_addresses->prepare_prefix_fork(
            source_kv->text, *transaction.root_text_address, *transaction.text_activation_frontier,
            details.text_kv_page_entitlement,
            static_cast<std::int32_t>(transaction.destination.value),
            details.text_retained_tail_release));
        prepare_retained_tail_backup(
            *text_kv_addresses, *text_kv_pages, *transaction.text_prefix_fork,
            details.text_retained_tail_release, transaction.text_retained_tail,
            transaction.text_retained_tail_backup);
        // 边界落在页中间：这一页要被两条地址空间同时引用，必须复制一份（写时复制的唯一一次真拷贝）。
        if (*transaction.text_activation_frontier % static_cast<std::uint32_t>(kPagedKVPageSize) !=
            0) {
            start_context_transfer_timer(runtime::ContextResourceClass::MainKV);
            text_kv_pages->physical_pool().copy_page(
                text_kv_addresses->prefix_fork_tail_source(*transaction.text_prefix_fork),
                text_kv_addresses->prefix_fork_tail_destination(*transaction.text_prefix_fork),
                device.transfer_stream);
            stop_context_transfer_timer(runtime::ContextResourceClass::MainKV);
            transaction.transfer_timer_mask |=
                1U << context_resource_index(runtime::ContextResourceClass::MainKV);
            ++transaction.operations.partial_tail_cow_pages;
            copied_tail = true;
        }
    }

    if (details.backend_prefix_fork_required) {
        if (!transaction.root_backend_address || !transaction.backend_activation_frontier) {
            throw std::logic_error("Backend KV prefix-fork destination is incomplete");
        }
        if (!source_kv->backend) {
            throw std::logic_error("Backend KV prefix-fork source is unavailable");
        }
        transaction.backend_source_restore_reservation.reset();
        transaction.backend_prefix_fork.emplace(backend_kv_addresses->prepare_prefix_fork(
            *source_kv->backend, *transaction.root_backend_address,
            *transaction.backend_activation_frontier, details.backend_kv_page_entitlement,
            static_cast<std::int32_t>(transaction.destination.value),
            details.backend_retained_tail_release));
        prepare_retained_tail_backup(
            *backend_kv_addresses, *backend_kv_pages, *transaction.backend_prefix_fork,
            details.backend_retained_tail_release, transaction.backend_retained_tail,
            transaction.backend_retained_tail_backup);
        // 后端 KV 同理：尾页在页中间时也要复制一份。
        if (*transaction.backend_activation_frontier %
                static_cast<std::uint32_t>(kPagedKVPageSize) !=
            0) {
            start_context_transfer_timer(runtime::ContextResourceClass::BackendKV);
            backend_kv_pages->physical_pool().copy_page(
                backend_kv_addresses->prefix_fork_tail_source(*transaction.backend_prefix_fork),
                backend_kv_addresses->prefix_fork_tail_destination(
                    *transaction.backend_prefix_fork),
                device.transfer_stream);
            stop_context_transfer_timer(runtime::ContextResourceClass::BackendKV);
            transaction.transfer_timer_mask |=
                1U << context_resource_index(runtime::ContextResourceClass::BackendKV);
            ++transaction.operations.partial_tail_cow_pages;
            copied_tail = true;
        }
    }

    // 尾页复制是异步的：登记一个完成点，并把本事务标成"传输未结算"，进度机下一轮会先把它同步掉、发布，
    // 再回到这里走后半段（备份/释放尾页）。没有拷贝可发时就直接宣布前缀就绪。
    if (copied_tail) {
        context_completion_.record(device.transfer_stream);
        transaction.prefix_tail_submitted = true;
        transaction.transfer_submitted    = true;
    } else {
        transaction.prefix_forks_ready = true;
    }
}

// 把 prepare_materialization 登记好的恢复清单真正发到传输流上（KV 的 H2D 补页；状态的 Host 搬回在
// prepare 阶段就发过了）。相邻且在同一 Host extent 里连续的页会被合并成一次拷贝——这是纯粹的批处理。
// 发完登记完成点；一张拷贝都没发时，说明本事务的活只剩前缀 fork，直接转交给 prepare_prefix_forks。
void ProgramImpl::enqueue_materialization_transfers(MaterializationTransaction& transaction) {
    if (!transaction.prepared || transaction.transfer_submitted) {
        throw std::logic_error("materialization transfer batch is not enqueueable");
    }
    const auto enqueue_kv =
        [&](LogicalKVPageStore& pages,
            const std::vector<MaterializationTransaction::KVRestorePage>& restores,
            const std::vector<DeviceKVPageHandle>& destinations,
            runtime::ContextResourceClass resource) {
            if (restores.size() != destinations.size()) {
                throw std::logic_error("KV restore bookkeeping is not row aligned");
            }
            if (restores.empty()) { return; }
            start_context_transfer_timer(resource);
            // 把 Host extent 里位置连续的相邻页并成一段，逐段投递。
            std::size_t begin = 0;
            while (begin < restores.size()) {
                std::size_t end = begin + 1;
                while (end < restores.size() && restores[end].extent == restores[begin].extent &&
                       restores[end].extent_page == restores[end - 1].extent_page + 1U) {
                    ++end;
                }
                const HostKVAllocationConstView source =
                    host_kv_extents->view(restores[begin].extent)
                        .subview(restores[begin].extent_page,
                                 static_cast<std::uint32_t>(end - begin));
                pages.physical_pool().copy_from_host(
                    source,
                    std::span<const DeviceKVPageHandle>(destinations.data() + begin, end - begin),
                    device.transfer_stream);
                begin = end;
            }
            stop_context_transfer_timer(resource);
            transaction.transfer_timer_mask |= 1U << context_resource_index(resource);
        };
    enqueue_kv(*text_kv_pages, transaction.text_restores, transaction.text_restore_destinations,
               runtime::ContextResourceClass::MainKV);
    if (!transaction.backend_restores.empty()) {
        enqueue_kv(*backend_kv_pages, transaction.backend_restores,
                   transaction.backend_restore_destinations,
                   runtime::ContextResourceClass::BackendKV);
    }
    // 有拷贝在飞才需要登记完成点；一张都没有（或只剩前缀 fork）就直接进入下一步。
    const bool any = transaction.state_restore.has_value() || !transaction.text_restores.empty() ||
                     !transaction.backend_restores.empty();
    if (any) {
        context_completion_.record(device.transfer_stream);
        transaction.transfer_submitted = true;
    } else if (transaction.plan && transaction.plan->impl_ &&
               (transaction.plan->impl_->text_prefix_fork_required ||
                transaction.plan->impl_->backend_prefix_fork_required)) {
        prepare_prefix_forks(transaction);
    }
}

// 传输完成后补记观测：计时器已经在发起拷贝时开关过，这里把"哪一类资源、哪个方向、多大工作量、多少页"
// 汇总成对外的观测行。掩码保证每一类资源只记一次；各分支按本事务实际做过的事（尾页 CoW / 尾页备份 /
// 恢复搬运）挑对应的方向与工作量模型。
void ProgramImpl::record_materialization_transfer_observations(
    MaterializationTransaction& transaction) {
    if (!transaction.transfer_submitted || !context_completion_.ready()) {
        throw std::logic_error("materialization transfer observation is not complete");
    }
    const auto record = [&](runtime::ContextResourceClass resource,
                            runtime::ContextTransferDirection direction, TransferWork transfer_work,
                            std::uint32_t pages) {
        const std::uint8_t bit = static_cast<std::uint8_t>(1U << context_resource_index(resource));
        if ((transaction.transfer_timer_mask & bit) == 0) { return; }
        transaction.transfer_observations.push_back(
            context_transfer_observation(resource, direction, transfer_work, pages));
        transaction.transfer_timer_mask &= static_cast<std::uint8_t>(~bit);
    };
    const auto host_layout = [](const LogicalKVPageStore& pages) {
        return plan_host_kv_page_layout(pages.physical_pool().geometry());
    };
    const auto restore_copy_runs = [](const auto& restores, const auto& destinations,
                                      const LogicalKVPageStore& pages) {
        if (restores.size() != destinations.size()) {
            throw std::logic_error("KV restore observation is not row aligned");
        }
        std::uint32_t runs = 0;
        std::size_t begin  = 0;
        while (begin < restores.size()) {
            std::size_t end = begin + 1U;
            while (end < restores.size() && restores[end].extent == restores[begin].extent &&
                   restores[end].extent_page == restores[end - 1U].extent_page + 1U) {
                ++end;
            }
            runs += pages.physical_pool().contiguous_run_count(
                std::span<const DeviceKVPageHandle>(destinations.data() + begin, end - begin));
            begin = end;
        }
        return runs;
    };
    if (transaction.prefix_tail_submitted) {
        if (transaction.text_prefix_fork && transaction.text_prefix_fork->needs_tail_copy()) {
            record(runtime::ContextResourceClass::MainKV,
                   runtime::ContextTransferDirection::DeviceToDevice,
                   plan_device_kv_copy_work(host_layout(*text_kv_pages), 1), 1);
        }
        if (backend_kv_pages && transaction.backend_prefix_fork &&
            transaction.backend_prefix_fork->needs_tail_copy()) {
            record(runtime::ContextResourceClass::BackendKV,
                   runtime::ContextTransferDirection::DeviceToDevice,
                   plan_device_kv_copy_work(host_layout(*backend_kv_pages), 1), 1);
        }
        return;
    }
    if (transaction.retained_tail_backup_submitted) {
        if (transaction.text_retained_tail_backup) {
            record(runtime::ContextResourceClass::MainKV,
                   runtime::ContextTransferDirection::DeviceToHost,
                   plan_host_kv_transfer_work(host_layout(*text_kv_pages), 1, 1), 1);
        }
        if (backend_kv_pages && transaction.backend_retained_tail_backup) {
            record(runtime::ContextResourceClass::BackendKV,
                   runtime::ContextTransferDirection::DeviceToHost,
                   plan_host_kv_transfer_work(host_layout(*backend_kv_pages), 1, 1), 1);
        }
        return;
    }
    if (transaction.state_restore) {
        record(runtime::ContextResourceClass::State,
               runtime::ContextTransferDirection::HostToDevice,
               state_image_transfer_work(host_state_images->layout()), 0);
    }
    record(runtime::ContextResourceClass::MainKV, runtime::ContextTransferDirection::HostToDevice,
           plan_host_kv_transfer_work(host_layout(*text_kv_pages),
                                      static_cast<std::uint32_t>(transaction.text_restores.size()),
                                      restore_copy_runs(transaction.text_restores,
                                                        transaction.text_restore_destinations,
                                                        *text_kv_pages)),
           static_cast<std::uint32_t>(transaction.text_restores.size()));
    if (backend_kv_pages) {
        record(runtime::ContextResourceClass::BackendKV,
               runtime::ContextTransferDirection::HostToDevice,
               plan_host_kv_transfer_work(
                   host_layout(*backend_kv_pages),
                   static_cast<std::uint32_t>(transaction.backend_restores.size()),
                   restore_copy_runs(transaction.backend_restores,
                                     transaction.backend_restore_destinations, *backend_kv_pages)),
               static_cast<std::uint32_t>(transaction.backend_restores.size()));
    }
}

// 传输完成后的"落定"。它本身也是分段的：每处理完一段，可能又发起新的传输（尾页 CoW 之后的备份），
// 于是带着 transfer_submitted 退出，等下一轮再进来接着做。顺序是——
//   尾页 CoW 已提交 → 备份尾页到 Host（若有需要）→ 发布备份并释放来源尾页 → 前缀就绪；
//   否则 → 发布状态镜像与 KV 恢复页（预订变正式）→ 若还要前缀 fork 就转交 prepare_prefix_forks。
void ProgramImpl::publish_materialization_transfers(MaterializationTransaction& transaction) {
    record_materialization_transfer_observations(transaction);
    // 尾页备份：把设备上的尾页拷进预先订好的 Host extent。这一段同样是"发完就退出"。
    const auto enqueue_retained_tail_backups = [&]() {
        bool submitted     = false;
        const auto enqueue = [&](LogicalKVPageStore& pages,
                                 std::optional<HostKVExtentReservation>& backup,
                                 runtime::ContextResourceClass resource) {
            if (!backup) { return; }
            if (host_kv_extents == nullptr || host_kv_extents->page_count(*backup) != 1) {
                throw std::logic_error("retained KV tail Host reservation changed");
            }
            std::array<DeviceKVPageHandle, 1> source{};
            host_kv_extents->device_sources(*backup, source);
            start_context_transfer_timer(resource);
            pages.physical_pool().copy_to_host(source, host_kv_extents->writable_view(*backup),
                                               device.transfer_stream);
            stop_context_transfer_timer(resource);
            transaction.transfer_timer_mask |= 1U << context_resource_index(resource);
            submitted = true;
        };
        enqueue(*text_kv_pages, transaction.text_retained_tail_backup,
                runtime::ContextResourceClass::MainKV);
        if (backend_kv_pages) {
            enqueue(*backend_kv_pages, transaction.backend_retained_tail_backup,
                    runtime::ContextResourceClass::BackendKV);
        }
        if (submitted) {
            context_completion_.record(device.transfer_stream);
            transaction.retained_tail_backup_submitted = true;
            transaction.transfer_submitted             = true;
        }
        return submitted;
    };
    // 来源尾页的最终处置：备份先落进 Host extent，来源页的引用关系结算完，才允许丢掉它的设备副本。
    // 顺序不能颠倒——先释放设备页再发布备份，中间任何失败都会丢掉数据。
    const auto publish_retained_tail_releases = [&]() {
        if (!transaction.plan || transaction.plan->impl_ == nullptr) {
            throw std::logic_error("retained KV tail release lost its admission plan");
        }
        const AdmissionCandidateImpl& details = *transaction.plan->impl_;
        const auto publish = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                 std::optional<KVPrefixForkReservation>& fork, bool staged,
                                 std::optional<LogicalKVPageHandle>& retained_tail,
                                 std::optional<HostKVExtentReservation>& backup) {
            if (!staged) {
                if (retained_tail || backup) {
                    throw std::logic_error("unstaged KV prefix fork owns a retained tail release");
                }
                return;
            }
            if (!fork || !retained_tail ||
                addresses.prefix_fork_tail_logical_source(*fork) != *retained_tail) {
                throw std::logic_error("staged KV prefix-fork tail identity changed");
            }
            if (backup) {
                if (host_kv_extents == nullptr) {
                    throw std::logic_error("retained KV tail Host store disappeared");
                }
                (void)host_kv_extents->publish(std::move(*backup));
                backup.reset();
            }
            addresses.settle_prefix_fork_tail_source(*fork);
            if (!pages.drop_device_replica(*retained_tail)) {
                throw std::logic_error("retained KV tail Device replica is not releasable");
            }
            addresses.complete_prefix_fork_after_tail_release(*fork);
            retained_tail.reset();
        };
        publish(*text_kv_addresses, *text_kv_pages, transaction.text_prefix_fork,
                details.text_retained_tail_release, transaction.text_retained_tail,
                transaction.text_retained_tail_backup);
        if (details.backend_retained_tail_release) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("staged Backend KV tail store is unavailable");
            }
            publish(*backend_kv_addresses, *backend_kv_pages, transaction.backend_prefix_fork, true,
                    transaction.backend_retained_tail, transaction.backend_retained_tail_backup);
        } else if (transaction.backend_retained_tail || transaction.backend_retained_tail_backup) {
            throw std::logic_error("unstaged Backend KV tail release was prepared");
        }
        transaction.prefix_forks_ready = true;
    };
    // 尾页 CoW 已同步：接着要么去备份尾页（新一段传输），要么直接做尾页释放收尾。
    if (transaction.prefix_tail_submitted) {
        transaction.prefix_tail_submitted = false;
        transaction.transfer_submitted    = false;
        if (enqueue_retained_tail_backups()) { return; }
        publish_retained_tail_releases();
        return;
    }
    if (transaction.retained_tail_backup_submitted) {
        transaction.retained_tail_backup_submitted = false;
        transaction.transfer_submitted             = false;
        publish_retained_tail_releases();
        return;
    }
    // 常规路径：状态镜像与 KV 恢复页从"预订"转为"已发布"。状态搬运在 prepare 阶段就发起了，这里只是
    // 承认它；第二个参数 true = 保留来源侧的副本，也就是搬回设备之后 Host 上那份检查点继续留着。
    if (transaction.state_restore) {
        state_store->publish_transfer(std::move(*transaction.state_restore), true);
        transaction.state_restore.reset();
        transaction.state_restored = true;
    }
    // 恢复页同样在这里转正式：发布之后它们才算"设备上有一份可用的副本"，此前只是预订+拷贝。
    for (const MaterializationTransaction::KVRestorePage& restore : transaction.text_restores) {
        text_kv_pages->publish_device_replica(restore.logical);
    }
    for (const MaterializationTransaction::KVRestorePage& restore : transaction.backend_restores) {
        backend_kv_pages->publish_device_replica(restore.logical);
    }
    transaction.text_restores.clear();
    transaction.text_restore_destinations.clear();
    transaction.backend_restores.clear();
    transaction.backend_restore_destinations.clear();
    transaction.transfer_submitted = false;
    if (transaction.plan && transaction.plan->impl_ &&
        (transaction.plan->impl_->text_prefix_fork_required ||
         transaction.plan->impl_->backend_prefix_fork_required)) {
        prepare_prefix_forks(transaction);
    }
}

// 传输阶段的撤销。此刻可能还有拷贝在飞——先把传输流同步干净（不留下写了一半的页），把已经发生的传输补
// 记成观测，再把所有**尚未发布**的目标退回：状态搬运中止，KV 恢复页交还预订位。已经发布的东西不能撤，
// 所以这里只处理"还在预订态"的对象；任何一步失败都是逻辑错误，直接 terminate。
void ProgramImpl::abort_materialization_transfers(
    MaterializationTransaction& transaction) noexcept {
    try {
        if (transaction.transfer_submitted) {
            context_completion_.synchronize();
            record_materialization_transfer_observations(transaction);
        }
        // 状态搬运的中止由镜像库自己保证：它知道这次搬运占据的是哪一格。
        if (transaction.state_restore) {
            state_store->abort_transfer(std::move(*transaction.state_restore));
            transaction.state_restore.reset();
        }
        // 恢复页的预订位就藏在"激活"或 fork 的预订位里，取出来按行退回。
        if (transaction.text_activation || transaction.text_source_restore_reservation) {
            DeviceKVPageReservation& reservation =
                transaction.text_source_restore_reservation
                    ? *transaction.text_source_restore_reservation
                    : text_kv_addresses->page_reservation(*transaction.text_activation);
            for (const MaterializationTransaction::KVRestorePage& restore :
                 transaction.text_restores) {
                text_kv_pages->abort_device_replica(restore.logical, reservation);
            }
        }
        if (transaction.backend_activation || transaction.backend_source_restore_reservation) {
            DeviceKVPageReservation& reservation =
                transaction.backend_source_restore_reservation
                    ? *transaction.backend_source_restore_reservation
                    : backend_kv_addresses->page_reservation(*transaction.backend_activation);
            for (const MaterializationTransaction::KVRestorePage& restore :
                 transaction.backend_restores) {
                backend_kv_pages->abort_device_replica(restore.logical, reservation);
            }
        }
    } catch (...) { std::terminate(); }
    // 清单清空、计时掩码归零：本事务此后不再有传输阶段，观测也已经补记过了。
    transaction.text_restores.clear();
    transaction.text_restore_destinations.clear();
    transaction.backend_restores.clear();
    transaction.backend_restore_destinations.clear();
    transaction.transfer_timer_mask = 0;
    transaction.transfer_submitted  = false;
}

// 压力工作的"建账"：把计划里早已定好的动作（owner 选择出的每一条）翻译成本次事务的私有记账行。
// 这里不碰任何物理资源，只做两件事——把页区间展开成具体的逻辑页句柄，并校验区间确实落在该地址空间内。
// 纯驱逐（evicts_continuation）不需要这份账：它整条 owner 一起走，没有按页的搬运。
void ProgramImpl::prepare_pressure_bookkeeping(MaterializationTransaction::PressureWork& work) {
    work.state_changes.clear();
    work.main_kv_changes.clear();
    work.backend_kv_changes.clear();
    if (work.option.evicts_continuation) { return; }

    work.state_changes.resize(work.option.state_changes.size());

    const SequenceState* sequence =
        work.shared_owner ? nullptr : &continuation_states[work.continuation_index];
    const SharedPrefixState* shared =
        work.shared_owner ? &shared_prefix_states[work.continuation_index] : nullptr;
    const SequenceKVBundle* kv = sequence != nullptr ? (sequence->kv ? &*sequence->kv : nullptr)
                                                     : (shared->kv ? &*shared->kv : nullptr);
    if (kv == nullptr) { throw std::logic_error("pressure owner has no KV address space"); }

    const auto prepare =
        [&](KVAddressSpaceStore* addresses, LogicalKVPageStore* pages,
            std::optional<KVAddressSpaceHandle> address,
            std::span<const qwen3_5::detail::PressureKVDecision> actions,
            std::vector<MaterializationTransaction::PressureWork::KVChangeWork>& changes) {
            changes.reserve(actions.size());
            for (const qwen3_5::detail::PressureKVDecision& action : actions) {
                changes.emplace_back();
                MaterializationTransaction::PressureWork::KVChangeWork& change = changes.back();
                if (action.kind == qwen3_5::detail::PressureKVDecisionKind::None) {
                    throw std::logic_error("pressure KV action has no operation kind");
                }
                if (addresses == nullptr || pages == nullptr || !address) {
                    throw std::logic_error("pressure KV action has no typed address space");
                }
                const std::uint32_t mapped = addresses->mapped_pages(*address);
                if (action.page_count == 0 || action.begin_page > mapped ||
                    action.page_count > mapped - action.begin_page) {
                    throw std::logic_error("pressure KV action range is invalid");
                }
                change.pages.reserve(action.page_count);
                for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
                    change.pages.push_back(
                        addresses->logical_page(*address, action.begin_page + offset));
                }
                // DemoteToHost 要往 Host 搬，提前把"设备侧来源页句柄"的位置留好（真正的来源在准备
                // 传输时由 Host extent 预订结果填进 change.sources）。
                if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DemoteToHost) {
                    change.sources.resize(action.page_count);
                }
            }
        };
    prepare(text_kv_addresses.get(), text_kv_pages.get(), kv->text, work.option.main_kv_changes,
            work.main_kv_changes);
    prepare(backend_kv_addresses.get(), backend_kv_pages.get(), kv->backend,
            work.option.backend_kv_changes, work.backend_kv_changes);
}

// 第一阶段的落地：只做"删了不用搬"的那部分动作——丢 Host 副本、丢检查点。它们不依赖任何拷贝，所以
// 在拷贝开始之前就能提交；也正因如此，事务中途取消时这部分已经生效、不再回滚（见 progress 里的取消
// 说明）。记账只累加到 committed_delta，最终还要和计划承诺的 effect 对账。
void ProgramImpl::publish_pressure_host_releases(MaterializationTransaction::PressureWork& work) {
    detail::PhysicalDelta delta;
    if (work.option.evicts_continuation || work.completed || work.submitted) { return; }
    const bool valid_owner =
        work.shared_owner ? (work.continuation_index < shared_prefix_capacity &&
                             shared_prefix_slots[work.continuation_index].role ==
                                 SharedPrefixSlotRole::Catalogued &&
                             shared_prefix_slots[work.continuation_index].generation ==
                                 work.continuation_generation &&
                             shared_prefix_states[work.continuation_index].active_references == 0)
                          : (work.continuation_index < continuation_capacity &&
                             continuation_slots[work.continuation_index].role ==
                                 ContinuationSlotRole::Catalogued &&
                             continuation_slots[work.continuation_index].generation ==
                                 work.continuation_generation);
    if (!valid_owner || work.option.shared_owner != work.shared_owner) {
        throw std::logic_error("pressure Host release owner changed before publication");
    }
    SequenceState* sequence =
        work.shared_owner ? nullptr : &continuation_states[work.continuation_index];
    SharedPrefixState* shared =
        work.shared_owner ? &shared_prefix_states[work.continuation_index] : nullptr;

    // 丢检查点只对私有（continuation）owner 成立：共享前缀的检查点不属于任何单条请求。
    if (!work.option.dropped_checkpoints.empty() && !work.checkpoint_drop_published) {
        if (sequence == nullptr) {
            throw std::logic_error("checkpoint drop targets a shared pressure owner");
        }
        for (const runtime::CheckpointRef checkpoint : work.option.dropped_checkpoints) {
            publish_checkpoint_drop(*sequence, checkpoint);
        }
        delta.removed =
            checked_resource_sum(delta.removed, work.option.checkpoint_drop_effect.removed);
        delta.added = checked_resource_sum(delta.added, work.option.checkpoint_drop_effect.added);
        work.checkpoint_drop_published = true;
        work.mutation_published        = true;
        const bool pure_drop           = work.option.state_changes.empty() &&
                               work.option.main_kv_changes.empty() &&
                               work.option.backend_kv_changes.empty();
        if (pure_drop) {
            work.committed_delta = delta;
            work.completed       = true;
            return;
        }
    }

    if (work.state_changes.size() != work.option.state_changes.size()) {
        throw std::logic_error("pressure State bookkeeping is not action aligned");
    }
    for (std::size_t index = 0; index < work.option.state_changes.size(); ++index) {
        const qwen3_5::detail::PressureStateDecision action = work.option.state_changes[index];
        auto& change                                        = work.state_changes[index];
        if (!pressure_state_drops_host(action) || change.host_released) { continue; }
        const std::optional<StateImageHandle> state =
            pressure_state_source(action, sequence, shared);
        if (!state || !state_store->drop_host_replica(*state)) {
            throw std::logic_error("pressure Host State duplicate is no longer releasable");
        }
        change.host_released    = true;
        work.mutation_published = true;
        ++delta.removed.host.state_slots;
    }

    const SequenceKVBundle* kv = sequence != nullptr ? (sequence->kv ? &*sequence->kv : nullptr)
                                                     : (shared->kv ? &*shared->kv : nullptr);
    if (kv == nullptr) { throw std::logic_error("pressure Host release owner has no KV bundle"); }
    const auto release_kv = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                KVAddressSpaceHandle address,
                                const qwen3_5::detail::PressureKVDecision& action,
                                MaterializationTransaction::PressureWork::KVChangeWork& change) {
        if (action.kind != qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate ||
            change.host_released) {
            return;
        }
        const std::uint32_t mapped = addresses.mapped_pages(address);
        if (action.begin_page > mapped || action.page_count > mapped - action.begin_page ||
            change.pages.size() != action.page_count) {
            throw std::logic_error("pressure Host KV release region changed");
        }
        for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
            if (change.pages[offset] !=
                addresses.logical_page(address, action.begin_page + offset)) {
                throw std::logic_error("pressure Host KV release membership changed");
            }
        }
        if (!host_kv_extents || !host_kv_extents->release_page_replicas(pages, change.pages)) {
            throw std::logic_error("pressure Host KV duplicates are no longer releasable");
        }
        const std::size_t page_stride =
            &pages == text_kv_pages.get() ? text_host_kv_page_stride : backend_host_kv_page_stride;
        if (action.page_count != 0 &&
            page_stride > std::numeric_limits<std::size_t>::max() / action.page_count) {
            throw std::overflow_error("pressure Host KV release size overflow");
        }
        const std::size_t bytes = page_stride * static_cast<std::size_t>(action.page_count);
        if (bytes > std::numeric_limits<std::size_t>::max() - delta.removed.host.kv_bytes) {
            throw std::overflow_error("pressure Host KV release sum overflow");
        }
        delta.removed.host.kv_bytes += bytes;
        change.host_released    = true;
        work.mutation_published = true;
    };
    if (work.main_kv_changes.size() != work.option.main_kv_changes.size() ||
        work.backend_kv_changes.size() != work.option.backend_kv_changes.size()) {
        throw std::logic_error("pressure KV bookkeeping is not action aligned");
    }
    for (std::size_t index = 0; index < work.option.main_kv_changes.size(); ++index) {
        release_kv(*text_kv_addresses, *text_kv_pages, kv->text, work.option.main_kv_changes[index],
                   work.main_kv_changes[index]);
    }
    if (!work.option.backend_kv_changes.empty()) {
        if (!kv->backend || !backend_kv_addresses || !backend_kv_pages) {
            throw std::logic_error("pressure Host Backend KV release has no typed store");
        }
        for (std::size_t index = 0; index < work.option.backend_kv_changes.size(); ++index) {
            release_kv(*backend_kv_addresses, *backend_kv_pages, *kv->backend,
                       work.option.backend_kv_changes[index], work.backend_kv_changes[index]);
        }
    }
    work.committed_delta.removed =
        checked_resource_sum(work.committed_delta.removed, delta.removed);
    work.committed_delta.added = checked_resource_sum(work.committed_delta.added, delta.added);
}

// 第二阶段：把"要搬运"的动作变成真正的拷贝。进度机按资源类分三轮调用它（State → MainKV →
// BackendKV），每次只处理 resource 这一类；这样计时与观测天然按类分开。每轮都要重新校验 owner 与
// 页区间——上一轮的删除可能已经改变了世界。DemoteToHost 在这里预订 Host extent 并发出 D2H 拷贝；
// 两类 Drop 只做校验，真正的删除留到发布阶段，以保证"数据先有落脚点，再删原件"。
void ProgramImpl::prepare_pressure_work(MaterializationTransaction::PressureWork& work,
                                        runtime::ContextResourceClass resource) {
    const bool valid_owner =
        work.shared_owner ? (work.continuation_index < shared_prefix_capacity &&
                             shared_prefix_slots[work.continuation_index].role ==
                                 SharedPrefixSlotRole::Catalogued &&
                             shared_prefix_slots[work.continuation_index].generation ==
                                 work.continuation_generation &&
                             shared_prefix_states[work.continuation_index].active_references == 0)
                          : (work.continuation_index < continuation_capacity &&
                             continuation_slots[work.continuation_index].role ==
                                 ContinuationSlotRole::Catalogued &&
                             continuation_slots[work.continuation_index].generation ==
                                 work.continuation_generation);
    if (work.completed || !valid_owner || work.option.shared_owner != work.shared_owner) {
        throw std::logic_error("pressure work source changed before transfer");
    }
    if (work.option.evicts_continuation) { return; }
    SequenceState* sequence =
        work.shared_owner ? nullptr : &continuation_states[work.continuation_index];
    SharedPrefixState* shared =
        work.shared_owner ? &shared_prefix_states[work.continuation_index] : nullptr;
    if (work.state_changes.size() != work.option.state_changes.size()) {
        throw std::logic_error("pressure State bookkeeping is not action aligned");
    }
    // State 类：按动作发起 D2H 搬运（每个动作最多一份状态镜像）。
    if (resource == runtime::ContextResourceClass::State) {
        for (std::size_t index = 0; index < work.option.state_changes.size(); ++index) {
            const qwen3_5::detail::PressureStateDecision action = work.option.state_changes[index];
            auto& change                                        = work.state_changes[index];
            if (!pressure_state_demotes(action)) { continue; }
            if (change.transfer) {
                throw std::logic_error("pressure State transfer was prepared more than once");
            }
            const std::optional<StateImageHandle> source =
                pressure_state_source(action, sequence, shared);
            if (!source) { throw std::logic_error("pressure State transfer has no source"); }
            std::optional<StateImageTransfer> transfer =
                state_store->begin_device_to_host(*source, device.transfer_stream);
            if (!transfer) { throw std::bad_alloc(); }
            change.transfer.emplace(std::move(*transfer));
        }
    }

    const auto prepare_kv = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                KVAddressSpaceHandle address,
                                const qwen3_5::detail::PressureKVDecision& action,
                                MaterializationTransaction::PressureWork::KVChangeWork& change) {
        if (action.page_count == 0) { return; }
        if (action.kind == qwen3_5::detail::PressureKVDecisionKind::None) {
            throw std::logic_error("pressure KV action has no operation kind");
        }
        if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate &&
            change.host_released) {
            return;
        }
        const std::uint32_t mapped = addresses.mapped_pages(address);
        if (action.begin_page > mapped || action.page_count > mapped - action.begin_page ||
            change.pages.size() != action.page_count) {
            throw std::logic_error("pressure KV region changed before transfer");
        }
        for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
            const LogicalKVPageHandle logical =
                addresses.logical_page(address, action.begin_page + offset);
            // 动作本身规定了当前应有的副本状态：DemoteToHost 只对"设备有、Host 没有"的页有意义；
            // 两类 Drop 则要求 Host 副本已经存在（要删的东西得先在那里）。
            const bool host_resident = pages.host_resident(logical);
            const bool valid_residency =
                action.kind == qwen3_5::detail::PressureKVDecisionKind::DemoteToHost
                    ? !host_resident
                    : host_resident;
            const bool removes_device =
                action.kind != qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate;
            if (!pages.device_resident(logical) || pages.writer_references(logical) != 0 ||
                pages.source_pins(logical) != 0 || !valid_residency ||
                (removes_device && addresses.has_active_reference(logical))) {
                throw std::logic_error("pressure KV replica changed before transfer");
            }
            if (change.pages[offset] != logical) {
                throw std::logic_error("pressure KV membership changed before transfer");
            }
        }
        if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate) {
            if (!host_kv_extents) { throw std::logic_error("Host KV extent store is unavailable"); }
            if (!host_kv_extents->can_release_page_replicas(pages, change.pages)) {
                throw std::logic_error("pressure Host KV replicas are no longer releasable");
            }
            return;
        }
        if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DropDeviceDuplicate) { return; }
        // 剩下的就是 DemoteToHost：先订好 Host 落脚点，再把每一页的设备来源拷过去。
        if (!host_kv_extents) { throw std::logic_error("Host KV extent store is unavailable"); }
        std::optional<HostKVExtentReservation> reserved =
            host_kv_extents->prepare(pages, change.pages);
        if (!reserved) { throw std::bad_alloc(); }
        if (change.sources.size() != change.pages.size()) {
            throw std::logic_error("pressure KV source backing was not prepared");
        }
        host_kv_extents->device_sources(*reserved, change.sources);
        pages.physical_pool().copy_to_host(
            change.sources, host_kv_extents->writable_view(*reserved), device.transfer_stream);
        change.backup.emplace(std::move(*reserved));
    };
    const SequenceKVBundle* kv = sequence != nullptr ? (sequence->kv ? &*sequence->kv : nullptr)
                                                     : (shared->kv ? &*shared->kv : nullptr);
    if (kv == nullptr) { throw std::logic_error("pressure owner has no KV address space"); }
    if (resource == runtime::ContextResourceClass::MainKV) {
        if (work.main_kv_changes.size() != work.option.main_kv_changes.size()) {
            throw std::logic_error("pressure Main KV bookkeeping is not action aligned");
        }
        for (std::size_t index = 0; index < work.option.main_kv_changes.size(); ++index) {
            prepare_kv(*text_kv_addresses, *text_kv_pages, kv->text,
                       work.option.main_kv_changes[index], work.main_kv_changes[index]);
        }
    }
    if (resource == runtime::ContextResourceClass::BackendKV &&
        !work.option.backend_kv_changes.empty()) {
        if (!kv->backend || !backend_kv_addresses || !backend_kv_pages) {
            throw std::logic_error("pressure owner has no Backend KV address space");
        }
        if (work.backend_kv_changes.size() != work.option.backend_kv_changes.size()) {
            throw std::logic_error("pressure Backend KV bookkeeping is not action aligned");
        }
        for (std::size_t index = 0; index < work.option.backend_kv_changes.size(); ++index) {
            prepare_kv(*backend_kv_addresses, *backend_kv_pages, *kv->backend,
                       work.option.backend_kv_changes[index], work.backend_kv_changes[index]);
        }
    }
    // 本轮有没有真的发出东西：有则事务要等完成点，进度机会停在"拷贝在飞"。
    work.submitted = std::any_of(work.state_changes.begin(), work.state_changes.end(),
                                 [](const auto& change) { return change.transfer.has_value(); }) ||
                     std::any_of(work.main_kv_changes.begin(), work.main_kv_changes.end(),
                                 [](const auto& change) { return change.backup.has_value(); }) ||
                     std::any_of(work.backend_kv_changes.begin(), work.backend_kv_changes.end(),
                                 [](const auto& change) { return change.backup.has_value(); });
}

// 第三阶段：把拷贝出来的副本与随后的删除一起落定。顺序仍然是"先有落脚点，再删原件"——备份先进
// Host extent，再丢设备副本。这里是 noexcept + terminate 的收尾路径：走到这一步前面的准备都成功了，
// 剩下的失败意味着账本已经和物理世界对不上，继续跑没有意义。
void ProgramImpl::publish_pressure_work(MaterializationTransaction::PressureWork& work) noexcept {
    try {
        if (work.option.evicts_continuation || work.completed) { std::terminate(); }
        SequenceState* sequence =
            work.shared_owner ? nullptr : &continuation_states[work.continuation_index];
        SharedPrefixState* shared =
            work.shared_owner ? &shared_prefix_states[work.continuation_index] : nullptr;
        if (work.state_changes.size() != work.option.state_changes.size()) { std::terminate(); }
        for (std::size_t index = 0; index < work.option.state_changes.size(); ++index) {
            const qwen3_5::detail::PressureStateDecision action = work.option.state_changes[index];
            auto& change                                        = work.state_changes[index];
            const std::optional<StateImageHandle> source =
                pressure_state_source(action, sequence, shared);
            if (!source) { std::terminate(); }
            // 搬出去的（DemoteToHost）在此转为"Host 侧持有"：第二个参数 false = 不留来源副本，即设备侧
            // 那一份随发布一起交出。
            if (change.transfer) {
                state_store->publish_transfer(std::move(*change.transfer), false);
                change.transfer.reset();
                work.mutation_published = true;
            } else if (!change.host_released) {
                // 没搬过、纯删副本的动作：按方向决定删 Host 还是删设备那一份。
                if (pressure_state_drops_host(action)
                        ? !state_store->drop_host_replica(*source)
                        : !state_store->drop_device_replica(*source)) {
                    std::terminate();
                }
                work.mutation_published = true;
            }
        }

        const auto publish_kv =
            [&](LogicalKVPageStore& pages, const qwen3_5::detail::PressureKVDecision& action,
                MaterializationTransaction::PressureWork::KVChangeWork& change) {
                // DropHostDuplicate 不搬运，所以不该持有备份；直接释放 Host 副本。
                if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate) {
                    if (change.host_released) { return; }
                    if (!host_kv_extents || change.backup) { std::terminate(); }
                    if (!host_kv_extents->release_page_replicas(pages, change.pages)) {
                        std::terminate();
                    }
                    work.mutation_published = true;
                    return;
                }
                // 先把落脚点（Host extent）转正，再丢设备副本。
                if (change.backup) {
                    if (!host_kv_extents) { std::terminate(); }
                    (void)host_kv_extents->publish(std::move(*change.backup));
                    change.backup.reset();
                }
                for (const LogicalKVPageHandle page : change.pages) {
                    if (!pages.drop_device_replica(page)) { std::terminate(); }
                }
                work.mutation_published = true;
            };
        if (work.main_kv_changes.size() != work.option.main_kv_changes.size() ||
            work.backend_kv_changes.size() != work.option.backend_kv_changes.size()) {
            std::terminate();
        }
        // spill_pages 只统计真正降级（DemoteToHost）的页数，供对外操作计数用；两类 Drop 不计。
        for (std::size_t index = 0; index < work.option.main_kv_changes.size(); ++index) {
            publish_kv(*text_kv_pages, work.option.main_kv_changes[index],
                       work.main_kv_changes[index]);
            if (work.option.main_kv_changes[index].kind ==
                qwen3_5::detail::PressureKVDecisionKind::DemoteToHost) {
                work.spill_pages += work.main_kv_changes[index].pages.size();
            }
        }
        if (!work.option.backend_kv_changes.empty()) {
            if (!backend_kv_pages) { std::terminate(); }
            for (std::size_t index = 0; index < work.option.backend_kv_changes.size(); ++index) {
                publish_kv(*backend_kv_pages, work.option.backend_kv_changes[index],
                           work.backend_kv_changes[index]);
                if (work.option.backend_kv_changes[index].kind ==
                    qwen3_5::detail::PressureKVDecisionKind::DemoteToHost) {
                    work.spill_pages += work.backend_kv_changes[index].pages.size();
                }
            }
        }
        work.submitted = false;
        work.completed = true;
    } catch (...) { std::terminate(); }
}

// 压力工作的撤销：与其它撤销一样，只退"还没提交的东西"——搬运中的状态镜像交还，Host extent 预订位
// 释放（预订本身就是临时的），已经提交给 HostReleases 阶段的删除不回滚。
void ProgramImpl::abort_pressure_work(MaterializationTransaction::PressureWork& work) noexcept {
    try {
        if (work.completed) { return; }
        for (auto& change : work.state_changes) {
            if (change.transfer) {
                state_store->abort_transfer(std::move(*change.transfer));
                change.transfer.reset();
            }
        }
        for (auto& change : work.main_kv_changes) { change.backup.reset(); }
        for (auto& change : work.backend_kv_changes) { change.backup.reset(); }
        work.state_changes.clear();
        work.main_kv_changes.clear();
        work.backend_kv_changes.clear();
        work.submitted = false;
    } catch (...) { std::terminate(); }
}

// 整条驱逐一个私有牺牲者。它不是原子的：槽位身份（索引+代际）可能在这期间变了，那就什么都不做，
// 返回空结果让调用方按"已经不由我们负责"处理。唯一特殊的一处是根请求自己的槽位——如果这条正是它
// 等的那个，释放之后要立刻改回 ReservedMaterialization，把它交还给正在等待的根事务。
ProgramImpl::PhysicalReleaseResult
ProgramImpl::release_materialization_victim(MaterializationTransaction& transaction,
                                            std::size_t position) {
    PhysicalReleaseResult out;
    if (position >= transaction.victim_count || transaction.victim_released[position]) {
        return out;
    }
    const std::uint32_t index      = transaction.victim_indices[position];
    const std::uint64_t generation = transaction.victim_generations[position];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Catalogued ||
        continuation_slots[index].generation != generation) {
        return out;
    }
    if (!can_release_continuation_slot_strict(index)) {
        throw std::logic_error("materialization victim is not strictly releasable");
    }

    out.delta.removed = owner_exclusive_resources(continuation_states[index]);
    release_continuation_slot_strict(index);
    if (transaction.root_waiting_for_victim && transaction.root_continuation_index == index) {
        continuation_slots[index].role      = ContinuationSlotRole::ReservedMaterialization;
        transaction.root_waiting_for_victim = false;
    }
    transaction.victim_released[position] = true;
    out.status                            = runtime::ConsumeStatus::Consumed;
    return out;
}

// 物化事务的推进机：Runtime 反复调用它，每一轮都推进到下一个"稳定点"就返回（要么 InProgress，要么
// 终态）。整条流水线是四段串起来的，顺序不能换：
//   1) 压力阶段（五相：HostReleases → CopyPreparation → CopiesInFlight → CopyPublication → Committed）。
//      先删不搬的（Host 副本/检查点），再按资源类发起 D2H 降级拷回，等拷贝落地后才真正删设备副本，
//      最后把一份份 work 的结果汇总成对外的回执。
//   2) 来源收尾 prepare_consumed_source：退 KV 进度、截断来源地址空间。
//   3) 传输阶段：发起恢复搬运、等完成、发布（预订转正）。
//   4) 设备侧准备 prepare_materialization + enqueue，最后 start_request —— 全流程唯一的物理发布点。
//
// 不可抢占：取消只在**阶段边界**被检查。已经提交的物理删除不回滚（也回滚不了），所以取消检查一律
// 紧跟在"一个阶段刚刚完成"之后，且此后不再发起新的阶段。中止时必须补齐所有回执——Runtime 依赖它们
// 归还逻辑目录能力。
MaterializationResult
ProgramImpl::progress_materialization_transaction(runtime::CancellationFlagView cancellation) {
    MaterializationResult out;
    MaterializationTransaction* transaction_ptr =
        std::get_if<MaterializationTransaction>(&context_transaction_);
    if (transaction_ptr == nullptr || transaction_ptr->terminal) {
        throw std::logic_error("Program has no progressable context transaction");
    }
    MaterializationTransaction& transaction = *transaction_ptr;
    PressureTransition& pressure_transition = transaction.pressure_transition;
    // 把各 work 的降级页数汇总到事务级计数，并就地归零（同一份 work 不会被统计两次）。饱和加法：
    // 计数溢出不值得让事务失败。
    const auto collect_pressure_operations  = [&](MaterializationTransaction::PressureWork& work) {
        if (work.spill_pages > std::numeric_limits<std::uint64_t>::max() -
                                   transaction.operations.pressure_spill_pages) {
            transaction.operations.pressure_spill_pages = std::numeric_limits<std::uint64_t>::max();
        } else {
            transaction.operations.pressure_spill_pages += work.spill_pages;
        }
        work.spill_pages = 0;
    };
    // 回执的两种形态。Retain(保留)：这个 owner 还在目录里，回执要带上它的最新摘要（模式/处置写清楚是"保留"）。
    const auto retain_private_result = [&](auto& result, const SequenceState& state) {
        if (!result.final_summary) {
            throw std::logic_error("private acknowledgement backing was not reserved");
        }
        using Result = std::remove_cvref_t<decltype(result)>;
        if constexpr (std::is_same_v<Result, MaterializationSourceResult>) {
            result.mode = runtime::PrivateSourceMode::Retain;
        } else {
            result.disposition = runtime::VictimDisposition::Retained;
        }
        populate_continuation_summary(state, *result.final_summary);
    };
    // Evicted(驱逐)：整个 owner 已经不在目录里，摘要没有意义，只留"已经提交"的标记。
    const auto evict_private_result = [&](MaterializationVictimResult& result) {
        result.disposition        = runtime::VictimDisposition::Evicted;
        result.pressure_committed = true;
        result.final_summary.reset();
    };
    // 牺牲者的回执：走到这里仍未被"物理释放"的（例如只压了它的 Host 副本，槽位还在目录里），要按
    // "保留"出具摘要；pressure_committed 直接取该 work 有没有真的动过物理资源。
    const auto complete_victim_acknowledgement = [&]() {
        for (std::size_t position = 0; position < transaction.victim_count; ++position) {
            if (transaction.victim_released[position]) { continue; }
            const std::uint32_t index      = transaction.victim_indices[position];
            const std::uint64_t generation = transaction.victim_generations[position];
            if (index >= continuation_capacity ||
                continuation_slots[index].role != ContinuationSlotRole::Catalogued ||
                continuation_slots[index].generation != generation) {
                throw std::logic_error("unmodified pressure claim is unavailable");
            }
            retain_private_result(transaction.pressure_results[position],
                                  continuation_states[index]);
            const MaterializationTransaction::PressureWork& work = transaction.pressure[position];
            transaction.pressure_results[position].pressure_committed = work.mutation_published;
        }
        out.victims = std::move(transaction.pressure_results);
    };
    // 私有来源的两种归途：ConsumeToActive 且已发布时，来源槽位已经变成这条 lane 的活动状态，本事务
    // 不再对外出具来源回执；否则来源必须仍然在目录里（身份未变），并带上最新摘要作为保留凭据。
    const auto complete_source_acknowledgement = [&](bool published) {
        if (!transaction.has_source) { return; }
        if (published && transaction.source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
            out.source.emplace(MaterializationSourceResult{
                .mode = runtime::PrivateSourceMode::ConsumeToActive,
            });
            return;
        }
        if (transaction.source_index >= continuation_capacity ||
            continuation_slots[transaction.source_index].role != ContinuationSlotRole::Catalogued ||
            continuation_slots[transaction.source_index].generation !=
                transaction.source_generation) {
            throw std::logic_error("retained materialization source is unavailable");
        }
        SequenceState& source = continuation_states[transaction.source_index];
        if (!transaction.source_result) {
            throw std::logic_error("materialization source backing was not reserved");
        }
        retain_private_result(*transaction.source_result, source);
        out.source.emplace(std::move(*transaction.source_result));
    };
    const auto complete_shared_source_acknowledgement = [&](bool published) {
        if (!transaction.has_shared_source) { return; }
        if (transaction.shared_source_index >= shared_prefix_capacity ||
            shared_prefix_slots[transaction.shared_source_index].role !=
                SharedPrefixSlotRole::Catalogued ||
            shared_prefix_slots[transaction.shared_source_index].generation !=
                transaction.shared_source_generation) {
            throw std::logic_error("retained materialization shared source is unavailable");
        }
        const SharedPrefixState& source = shared_prefix_states[transaction.shared_source_index];
        if (!transaction.shared_source_result) {
            throw std::logic_error("materialization shared-source backing was not reserved");
        }
        transaction.shared_source_result->final_summary = shared_prefix_summary(source);
        out.shared_source.emplace(std::move(*transaction.shared_source_result));
        // 共享来源不会因本事务消失，它至少还被本 lane 引用着——引用数归零说明账错了。
        if (published && out.shared_source->final_summary->active_references == 0) {
            throw std::logic_error("published shared source lost its active reference");
        }
    };
    const auto complete_shared_victim_acknowledgement = [&]() {
        for (std::size_t position = 0; position < transaction.shared_victim_count; ++position) {
            if (transaction.shared_victim_released[position]) { continue; }
            const std::uint32_t index      = transaction.shared_victim_indices[position];
            const std::uint64_t generation = transaction.shared_victim_generations[position];
            if (index >= shared_prefix_capacity ||
                shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
                shared_prefix_slots[index].generation != generation) {
                throw std::logic_error("unmodified shared pressure claim is unavailable");
            }
            transaction.shared_pressure_results[position] = MaterializationSharedVictimResult{
                .owner              = transaction.shared_pressure_results[position].owner,
                .disposition        = runtime::VictimDisposition::Retained,
                .pressure_committed = transaction.shared_pressure[position].mutation_published,
                .final_summary      = shared_prefix_summary(shared_prefix_states[index]),
            };
        }
        out.shared_victims = std::move(transaction.shared_pressure_results);
    };
    // 中止：先把暂存（预订、拷贝）退掉，再标记终态，最后补齐所有的回执——即使这一趟什么都没做成，
    // Runtime 也必须有完整的凭据去归还逻辑目录能力。
    const auto abort_transaction = [&]() {
        release_materialization_staging(transaction);
        transaction.terminal      = true;
        out.status                = runtime::ContextTransactionStatus::Aborted;
        out.transfer_observations = std::move(transaction.transfer_observations);
        out.operations            = transaction.operations;
        complete_source_acknowledgement(false);
        complete_shared_source_acknowledgement(false);
        complete_victim_acknowledgement();
        complete_shared_victim_acknowledgement();
    };

    if (cancellation.requested()) { transaction.cancel_pending = true; }

    // 第一相：只做"删了不用搬"的部分。共享牺牲者（前缀目录里的）先走，私有牺牲者随后。
    if (pressure_transition.phase == PressureTransitionPhase::HostReleases) {
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
        for (std::size_t position = 0; position < transaction.shared_victim_count; ++position) {
            MaterializationTransaction::PressureWork& work = transaction.shared_pressure[position];
            // 纯驱逐：整条 owner 一次性放掉（严格释放，不做任何隐式丢弃），并要求计划当初承诺的
            // "只减不增"仍然成立。
            if (work.option.evicts_continuation) {
                const std::uint32_t index      = transaction.shared_victim_indices[position];
                const std::uint64_t generation = transaction.shared_victim_generations[position];
                if (index >= shared_prefix_capacity ||
                    shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
                    shared_prefix_slots[index].generation != generation ||
                    shared_prefix_states[index].active_references != 0) {
                    throw std::logic_error("shared pressure victim changed before release");
                }
                const detail::PhysicalResources exclusive =
                    owner_exclusive_resources(shared_prefix_states[index]);
                if (work.option.effect.added != detail::PhysicalResources{}) {
                    throw std::logic_error("shared pressure eviction changed after reservation");
                }
                if (!can_release_shared_prefix_state(index, SharedPrefixSlotRole::Catalogued)) {
                    throw std::logic_error("shared pressure victim is not strictly releasable");
                }
                const detail::PhysicalResources released =
                    release_shared_prefix_state_strict(index, SharedPrefixSlotRole::Catalogued);
                if (released != exclusive) {
                    throw std::logic_error("shared pressure eviction acknowledgement is invalid");
                }
                work.committed_delta    = detail::PhysicalDelta{.removed = released};
                work.completed          = true;
                work.mutation_published = true;
                transaction.shared_pressure_results[position] = MaterializationSharedVictimResult{
                    .owner              = transaction.shared_pressure_results[position].owner,
                    .disposition        = runtime::VictimDisposition::Evicted,
                    .pressure_committed = true,
                };
                transaction.shared_victim_released[position] = true;
            } else {
                publish_pressure_host_releases(work);
            }
        }
        // 私有牺牲者同理：要么整条驱逐，要么只发布不搬运的 Host 侧删除。
        for (std::size_t position = 0; position < transaction.victim_count; ++position) {
            MaterializationTransaction::PressureWork& work = transaction.pressure[position];
            if (work.option.evicts_continuation) {
                const PhysicalReleaseResult released =
                    release_materialization_victim(transaction, position);
                if (released.status != runtime::ConsumeStatus::Consumed ||
                    released.delta.added != detail::PhysicalResources{} ||
                    work.option.effect.added != detail::PhysicalResources{}) {
                    throw std::logic_error("materialization eviction changed after reservation");
                }
                work.committed_delta    = released.delta;
                work.completed          = true;
                work.mutation_published = true;
                evict_private_result(transaction.pressure_results[position]);
            } else {
                publish_pressure_host_releases(work);
                if (work.completed) {
                    SequenceState& victim = continuation_states[work.continuation_index];
                    retain_private_result(transaction.pressure_results[position], victim);
                    transaction.pressure_results[position].pressure_committed = true;
                    transaction.victim_released[position]                     = true;
                }
            }
        }
        // 阶段边界。上面这些删除已经提交、无法回滚，因此从这里开始取消是"干净"的——后面还没有发起
        // 任何拷贝。取消检查一律照这个模式放在阶段收尾处。
        pressure_transition.phase = PressureTransitionPhase::CopyPreparation;
        if (cancellation.requested()) { transaction.cancel_pending = true; }
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
    }

    // 对账：实际减去/增加的量必须与计划当初承诺的一致（不一致说明账错了）。随后把 work 的记账直接记成
    // 承诺值，后续所有对账都以计划为准。
    const auto complete_pressure_delta = [&](MaterializationTransaction::PressureWork& work) {
        (void)checked_resource_difference(work.option.effect.removed, work.committed_delta.removed);
        (void)checked_resource_difference(work.option.effect.added, work.committed_delta.added);
        work.committed_delta = work.option.effect;
    };

    // 还没走完的 work：共享的在前，私有的在后（与发起顺序一致）。
    const auto for_each_pending_pressure = [&](auto&& callback) {
        for (MaterializationTransaction::PressureWork& work : transaction.shared_pressure) {
            if (!work.completed) { callback(work); }
        }
        for (MaterializationTransaction::PressureWork& work : transaction.pressure) {
            if (!work.completed) { callback(work); }
        }
    };

    // 第二相：发起降级拷贝。按资源类分三轮（State → MainKV → BackendKV），顺序固定——计时与观测都
    // 按这个次序切分。某一类完全没有 D2H 需求时不开计时器，也就不会产生观测行。
    if (pressure_transition.phase == PressureTransitionPhase::CopyPreparation) {
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }

        constexpr std::array pressure_resources{
            runtime::ContextResourceClass::State,
            runtime::ContextResourceClass::MainKV,
            runtime::ContextResourceClass::BackendKV,
        };
        try {
            for (const runtime::ContextResourceClass resource : pressure_resources) {
                bool has_copy = false;
                for_each_pending_pressure(
                    [&](const MaterializationTransaction::PressureWork& work) {
                        has_copy =
                            has_copy ||
                            std::any_of(
                                work.option.transfer_requirements.begin(),
                                work.option.transfer_requirements.end(),
                                [&](const auto& requirement) {
                                    return requirement.resource == resource &&
                                           requirement.direction ==
                                               runtime::ContextTransferDirection::DeviceToHost;
                                });
                    });
                if (has_copy) { start_context_transfer_timer(resource); }
                try {
                    for_each_pending_pressure([&](MaterializationTransaction::PressureWork& work) {
                        prepare_pressure_work(work, resource);
                    });
                } catch (...) {
                    if (has_copy) { stop_context_transfer_timer(resource); }
                    throw;
                }
                if (!has_copy) { continue; }
                stop_context_transfer_timer(resource);
                const std::size_t resource_index = context_resource_index(resource);
                pressure_transition.timer_mask |= static_cast<std::uint8_t>(1U << resource_index);
                for_each_pending_pressure(
                    [&](const MaterializationTransaction::PressureWork& work) {
                        for (const runtime::ContextTransferRequirement& requirement :
                             work.option.transfer_requirements) {
                            if (requirement.resource != resource ||
                                requirement.direction !=
                                    runtime::ContextTransferDirection::DeviceToHost) {
                                continue;
                            }
                            TransferWork& total = pressure_transition.transfer_work[resource_index];
                            total.payload_bytes =
                                requirement.work.payload_bytes >
                                        std::numeric_limits<std::uint64_t>::max() -
                                            total.payload_bytes
                                    ? std::numeric_limits<std::uint64_t>::max()
                                    : total.payload_bytes + requirement.work.payload_bytes;
                            const std::uint64_t operations =
                                static_cast<std::uint64_t>(total.copy_operations) +
                                requirement.work.copy_operations;
                            total.copy_operations =
                                operations > std::numeric_limits<std::uint32_t>::max()
                                    ? std::numeric_limits<std::uint32_t>::max()
                                    : static_cast<std::uint32_t>(operations);
                            const std::uint64_t pages =
                                static_cast<std::uint64_t>(
                                    pressure_transition.transfer_pages[resource_index]) +
                                requirement.page_count;
                            pressure_transition.transfer_pages[resource_index] =
                                pages > std::numeric_limits<std::uint32_t>::max()
                                    ? std::numeric_limits<std::uint32_t>::max()
                                    : static_cast<std::uint32_t>(pages);
                            if (resource == runtime::ContextResourceClass::State) {
                                pressure_transition.state_images =
                                    requirement.units > std::numeric_limits<std::uint64_t>::max() -
                                                            pressure_transition.state_images
                                        ? std::numeric_limits<std::uint64_t>::max()
                                        : pressure_transition.state_images + requirement.units;
                            }
                        }
                    });
            }
        } catch (...) {
            // 失败时可能已经有拷贝在飞：先把流同步干净，再逐个退回，然后原样抛出（由上层决定事务去留）。
            (void)cudaStreamSynchronize(device.transfer_stream);
            for_each_pending_pressure(
                [&](MaterializationTransaction::PressureWork& work) { abort_pressure_work(work); });
            throw;
        }

        bool copies_submitted = false;
        for_each_pending_pressure([&](const MaterializationTransaction::PressureWork& work) {
            copies_submitted = copies_submitted || work.submitted;
        });
        // 有东西在飞就停在等完成点；什么都没发就直接进入发布。
        pressure_transition.phase = copies_submitted ? PressureTransitionPhase::CopiesInFlight
                                                     : PressureTransitionPhase::CopyPublication;
        if (copies_submitted) {
            context_completion_.record(device.transfer_stream);
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
    }

    // 第三相：只等完成点，不做任何事。
    if (pressure_transition.phase == PressureTransitionPhase::CopiesInFlight) {
        if (!context_completion_.ready()) {
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
        pressure_transition.phase = PressureTransitionPhase::CopyPublication;
    }
    if (transaction.cancel_pending) {
        // 此刻 D2H 的目的地仍然是私有预订。等流同步、把这些预订退掉之后，外部能看到的只有
        // HostReleases 那一相已经提交的删除。
        abort_transaction();
        return out;
    }

    // 第四相：拷贝已经落地，这才真正删掉设备副本，并给每个 work 出回执。共享的在前，私有的在后，
    // 游标在最后才推进到末尾——中途抛异常时剩下的 work 仍然"未完成"，可被撤销路径统一处理。
    if (pressure_transition.phase == PressureTransitionPhase::CopyPublication) {
        for (std::size_t position = 0; position < transaction.shared_pressure.size(); ++position) {
            MaterializationTransaction::PressureWork& work = transaction.shared_pressure[position];
            if (work.completed) { continue; }
            publish_pressure_work(work);
            collect_pressure_operations(work);
            const std::uint32_t index = transaction.shared_victim_indices[position];
            transaction.shared_pressure_results[position] = MaterializationSharedVictimResult{
                .owner              = transaction.shared_pressure_results[position].owner,
                .disposition        = runtime::VictimDisposition::Retained,
                .pressure_committed = true,
                .final_summary      = shared_prefix_summary(shared_prefix_states[index]),
            };
            complete_pressure_delta(work);
            transaction.shared_victim_released[position] = true;
        }
        transaction.shared_pressure_cursor = transaction.shared_pressure.size();

        for (std::size_t position = 0; position < transaction.pressure.size(); ++position) {
            MaterializationTransaction::PressureWork& work = transaction.pressure[position];
            if (work.completed) { continue; }
            publish_pressure_work(work);
            collect_pressure_operations(work);
            retain_private_result(transaction.pressure_results[position],
                                  continuation_states[work.continuation_index]);
            transaction.pressure_results[position].pressure_committed = true;
            complete_pressure_delta(work);
            transaction.victim_released[position] = true;
        }
        transaction.pressure_cursor = transaction.pressure.size();

        constexpr std::array pressure_resources{
            runtime::ContextResourceClass::State,
            runtime::ContextResourceClass::MainKV,
            runtime::ContextResourceClass::BackendKV,
        };
        // 把压力阶段的计时按类转成对外观测行（方向固定是 D2H）。
        for (const runtime::ContextResourceClass resource : pressure_resources) {
            const std::size_t index = context_resource_index(resource);
            const std::uint8_t bit  = static_cast<std::uint8_t>(1U << index);
            if ((pressure_transition.timer_mask & bit) == 0) { continue; }
            transaction.transfer_observations.push_back(context_transfer_observation(
                resource, runtime::ContextTransferDirection::DeviceToHost,
                pressure_transition.transfer_work[index], pressure_transition.transfer_pages[index],
                pressure_transition.state_images));
        }
        pressure_transition.timer_mask = 0;
        pressure_transition.phase      = PressureTransitionPhase::Committed;
        if (cancellation.requested()) { transaction.cancel_pending = true; }
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
    }
    if (pressure_transition.phase != PressureTransitionPhase::Committed) {
        throw std::logic_error("materialization pressure transition did not reach a stable phase");
    }

    // 压力阶段整体提交之后，才收尾来源：退 KV 进度、截断来源地址空间。它必须排在压力之后——被压掉
    // 的空间正是来源可以收缩的前提。
    if (!transaction.source_prepared) {
        prepare_consumed_source(transaction);
        if (cancellation.requested()) { transaction.cancel_pending = true; }
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
    }

    // 传输阶段：等完成点 → 发布。发布本身可能又发起新传输（尾页备份），所以发布后还要再看一眼。
    if (transaction.transfer_submitted) {
        if (!context_completion_.ready()) {
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
        publish_materialization_transfers(transaction);
        if (transaction.transfer_submitted) {
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
    }

    if (transaction.cancel_pending) {
        abort_transaction();
        return out;
    }

    // 设备侧准备 + 发起拷贝：这一对是紧挨着的（准备只登记预订，紧接着就把拷贝发出去）。
    if (!transaction.prepared) {
        prepare_materialization(transaction);
        enqueue_materialization_transfers(transaction);
        if (transaction.transfer_submitted) {
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
    }
    if (cancellation.requested()) {
        abort_transaction();
        return out;
    }

    // 全流程唯一的物理发布点。走到这里所有物理资源都已就位，start_request 把它们交给 lane 正式执行；
    // 逻辑目录能力仍归 ResourceManager，它要在这份终态结果校验通过之后才会认领。发布失败则把暂存退掉，
    // 让事务保持"未提交"。
    try {
        out.published.emplace(start_request(transaction));
        materialization_ledger_.clear();
        materialization_identity_.clear();
        materialization_prefix_digests_.clear();
    } catch (...) {
        release_materialization_staging(transaction);
        throw;
    }
    transaction.terminal      = true;
    out.status                = runtime::ContextTransactionStatus::Published;
    out.transfer_observations = std::move(transaction.transfer_observations);
    out.operations            = transaction.operations;
    complete_source_acknowledgement(true);
    complete_shared_source_acknowledgement(true);
    complete_victim_acknowledgement();
    complete_shared_victim_acknowledgement();
    return out;
}

// 对外只有这一个推进入口：按 context_transaction_ 当前握着哪种事务分发。两种事务的实现是同构的——都
// 是自己内部的相位机，都只返回"进行中"或终态；这里统一把非法状态挡在门外。
ContextTransactionProgress
ProgramImpl::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    const auto terminal_or_pending =
        []<class Result>(Result&& result) -> ContextTransactionProgress {
        if (result.status == runtime::ContextTransactionStatus::InProgress) {
            return runtime::ContextTransactionInProgress{};
        }
        if (result.status != runtime::ContextTransactionStatus::Published &&
            result.status != runtime::ContextTransactionStatus::Aborted) {
            throw std::logic_error("context transaction returned an invalid status");
        }
        return ContextTransactionProgress(std::forward<Result>(result));
    };
    return std::visit(
        [&](auto& transaction) -> ContextTransactionProgress {
            using Transaction = std::decay_t<decltype(transaction)>;
            if constexpr (std::is_same_v<Transaction, std::monostate>) {
                throw std::logic_error("Program has no progressable context transaction");
            } else if constexpr (std::is_same_v<Transaction, MaterializationTransaction>) {
                return terminal_or_pending(progress_materialization_transaction(cancellation));
            } else {
                return terminal_or_pending(progress_active_capture_transaction(cancellation));
            }
        },
        context_transaction_);
}

// 终态事务在推进之后才被回收：先在这里确认它确实到了终态，再把它从 variant 里卸掉。回收之所以不能由
// 推进函数自己做，是因为调用方还没读走结果。
void ProgramImpl::finalize_context_transaction() noexcept {
    const bool terminal = std::visit(
        [](const auto& transaction) {
            using T = std::decay_t<decltype(transaction)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return false;
            } else if constexpr (std::is_same_v<T, ActiveCaptureTransaction>) {
                return transaction.published;
            } else {
                return transaction.terminal;
            }
        },
        context_transaction_);
    if (terminal) { context_transaction_.emplace<std::monostate>(); }
}

// 是否已有未终结的事务：一个新事务想要开始，前提是这里为 false。
bool ProgramImpl::has_context_transaction() const noexcept {
    return !std::holds_alternative<std::monostate>(context_transaction_);
}


} // namespace ninfer::models::qwen3_5::detail
