#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>

// ============================================================================
// context.cpp —— Program 的状态与账本底座：句柄校验、lane/槽位簿记、资源账、状态镜像与 KV 的生命周期
//
// 这个文件里几乎没有算法，全是对"世界现在是什么样"的读写与核算：谁是活的、谁占着多少、一份镜像或一段
// KV 被谁引用、放手时到底能释放什么。上层（planning/ 与 transactions/）的所有判断都建立在这些读数上，
// 所以这里的函数回答的都是同一类问题——事实是什么，而不是接下来该做什么。
//
// 与 program_impl.h 私有实现编号的对应关系：本文件实现第 2)、3)、4)、5)、8)、12) 段，外加第 11) 段的
// 两个边角动作和预填推进入口；文件内部顺序也与编号一致——
//   校验谓词 → lane 与续跑槽 → 资源账 → 来源选择 → KV 前缀数法 → 对外摘要 → 预填入口与 lane 清场
//   → 状态镜像的生命周期 → KV 地址空间的映射与放手 → 两个边角动作
//
// 三条贯穿全文件的纪律：
//   · **两套账，别混用**：逐 owner 的账只算"拆掉这个 owner 真能释放的"（别名共享的分配不算它的），
//     全局占用 physical_occupancy() 才是世界总量。
//   · **放手分档**：每个释放动作都有 strict / best_effort 两副面孔。strict 要求"放手之后一切干净"，
//     做不到就返回 false 或 terminate；best_effort 只保证"别崩、账别漏"，只配用在错误路径上。
//   · **矛盾即违约**：两套账对不上、引用数与实际持有对不上、地址空间与束不一致——一律抛异常或
//     terminate，绝不就地猜一个数。这里是账本的底座，没有人能替一个坏掉的账本兜底。
// ============================================================================
namespace ninfer::models::qwen3_5::detail {

// ============================================================================
// 校验谓词（program_impl.h 第 2) 段）
//
// 一律是"所有权 → 代次 → 角色/生命周期"三连检查，全部 noexcept：它们要能在任何时刻被调用，包括句柄
// 早已作废之后。判 false 表示"这个句柄现在不代表任何东西"，而不是"大概没问题"。
// ============================================================================

// 序列句柄：lane 代次对得上、该 lane 的续跑槽仍是 Active（也就是这条请求确实活着）、且生命周期落在
// "仍然活着"的四种里（Prefilling / Active / Pending / Finishable）——Empty 不算。
bool ProgramImpl::valid_sequence(SequenceHandle handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t lane = ContractAccess::lane(handle).value;
    if (lane >= max_concurrency || ContractAccess::epoch(handle) != lane_epochs[lane]) {
        return false;
    }
    if (active_continuations[lane] >= continuation_capacity ||
        continuation_slots[active_continuations[lane]].role != ContinuationSlotRole::Active) {
        return false;
    }
    const Lifecycle lifecycle = requests[lane].lifecycle;
    return lifecycle == Lifecycle::Prefilling || lifecycle == Lifecycle::Active ||
           lifecycle == Lifecycle::Pending || lifecycle == Lifecycle::Finishable;
}

// 续跑句柄：槽位索引在界内、代次相等、角色为 Catalogued——只有已经发布出去（可能正被别的请求拿去
// 复用）的续跑才有句柄在世上流通。
bool ProgramImpl::valid_continuation(const ContinuationHandle& handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t index = ContractAccess::index(handle);
    return index < continuation_capacity &&
           ContractAccess::epoch(handle) == continuation_slots[index].generation &&
           continuation_slots[index].role == ContinuationSlotRole::Catalogued;
}

// 共享前缀句柄：同上，槽位角色必须是 Catalogued——还在事务期（ReservedCapture / ReservedReplacement）
// 的那份前缀不许被外面引用。
bool ProgramImpl::valid_shared_prefix(const SharedPrefixHandle& handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t index = ContractAccess::index(handle);
    return index < shared_prefix_capacity &&
           ContractAccess::epoch(handle) == shared_prefix_slots[index].generation &&
           shared_prefix_slots[index].role == SharedPrefixSlotRole::Catalogued;
}

// 捕获票据：除三连检查外还要求它是**当前**那一张（id 与 pending_capture_offer 相等）且预填游标正好
// 停在这个捕获点的前沿上——票据只在"捕获点已经走到、等着裁决"的那一刻有效，挪一步就作废。
bool ProgramImpl::valid_capture_offer(const CaptureOffer& offer) const noexcept {
    if (ContractAccess::owner(offer) != this) { return false; }
    const std::uint32_t lane = ContractAccess::lane(offer).value;
    if (lane >= max_concurrency || ContractAccess::epoch(offer) != lane_epochs[lane] ||
        (requests[lane].lifecycle != Lifecycle::Prefilling &&
         requests[lane].lifecycle != Lifecycle::Active) ||
        !requests[lane].prefill) {
        return false;
    }
    const RequestControl::Prefill& prefill = *requests[lane].prefill;
    return prefill.pending_capture_offer != 0 &&
           prefill.pending_capture_offer == ContractAccess::id(offer) &&
           prefill.next_capture < prefill.capture_groups.size() &&
           prefill.cursor == prefill.capture_groups[prefill.next_capture].frontier;
}

// 这个续跑槽此刻是不是被某个进行中的物化**掐着**：作为来源被借用，或作为还没释放的牺牲者被记账，
// 都算掐着。Runtime 靠它回答"这个空闲槽现在能不能被别人拿走"——不是问句柄有没有效，而是问所有权有
// 没有悬在半空。
bool ProgramImpl::materialization_pins(std::uint32_t index,
                                       std::uint64_t generation) const noexcept {
    const MaterializationTransaction* transaction_ptr =
        std::get_if<MaterializationTransaction>(&context_transaction_);
    if (transaction_ptr == nullptr) { return false; }
    const MaterializationTransaction& transaction = *transaction_ptr;
    if (transaction.has_source && transaction.source_index == index &&
        transaction.source_generation == generation) {
        return true;
    }
    for (std::size_t victim = 0; victim < transaction.victim_count; ++victim) {
        if (!transaction.victim_released[victim] && transaction.victim_indices[victim] == index &&
            transaction.victim_generations[victim] == generation) {
            return true;
        }
    }
    return false;
}

// 未结算批次的凭据性：owner、事务号必须等于当前那个唯一的 pending 事务，行数与每行的 lane / 代次 /
// 生命周期（必须是 Pending）都要对得上。批次一旦被 commit 或 abort 消化掉，这份凭据立刻失效。
bool ProgramImpl::valid_pending(const PendingBatch& pending) const noexcept {
    if (ContractAccess::owner(pending) != this || !pending_transaction_ ||
        ContractAccess::transaction(pending) != pending_transaction_->id) {
        return false;
    }
    const auto rows = ContractAccess::rows(pending);
    if (rows.size() != pending_transaction_->size) { return false; }
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (!valid_sequence(rows[row]) ||
            ContractAccess::lane(rows[row]).value != pending_transaction_->lanes[row] ||
            ContractAccess::epoch(rows[row]) != pending_transaction_->epochs[row] ||
            requests[pending_transaction_->lanes[row]].lifecycle != Lifecycle::Pending) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// lane 与续跑槽的簿记（program_impl.h 第 8) 段的槽位部分）
//
// 两套生命周期在这里交汇：lane 是请求的槽位（靠代次一次性作废），续跑槽是序列状态的槽位（靠角色与
// 代次管理）。一条 lane 在使用期间绑定一个 Active 的续跑槽；续跑被发布出去（Catalogued）或退休之后，
// 这层绑定就断了。
// ============================================================================

// 整条 lane 作废的那个闸门：只推进代次，于是这条 lane 上所有句柄与票据同时失效，不需要逐个注销。
// 代次 0 被跳过——那是"从没作废过"的哨兵值。
void ProgramImpl::invalidate_lane(std::uint32_t lane) noexcept {
    if (lane >= max_concurrency) { return; }
    ++lane_epochs[lane];
    if (lane_epochs[lane] == 0) { ++lane_epochs[lane]; }
}

// lane → 它当前绑定的续跑状态（两个重载只差 const）。lane 没有 Active 绑定就抛：那是调用方用错了
// lane，不是可以猜一个的情况。
SequenceState& ProgramImpl::active_sequence(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("active lane is out of range"); }
    const std::uint32_t index = active_continuations[lane];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Active) {
        throw std::logic_error("active lane has no continuation binding");
    }
    return continuation_states[index];
}

const SequenceState& ProgramImpl::active_sequence(std::uint32_t lane) const {
    if (lane >= max_concurrency) { throw std::out_of_range("active lane is out of range"); }
    const std::uint32_t index = active_continuations[lane];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Active) {
        throw std::logic_error("active lane has no continuation binding");
    }
    return continuation_states[index];
}

// 领一个空闲续跑槽并直接占成 Active（分配与绑定是一步，免得中间被人抢走）。
std::optional<std::uint32_t> ProgramImpl::allocate_continuation_slot() noexcept {
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role == ContinuationSlotRole::Free) {
            continuation_slots[index].role = ContinuationSlotRole::Active;
            return index;
        }
    }
    return std::nullopt;
}

// 严格放手一个续跑槽的先决条件：槽位已目录化（没有活跃绑定）、两种 KV 地址空间都"放手后正好干净"、
// 每一份状态镜像的引用数都恰好等于这条序列自己持有的数量（多一个引用就说明别人还在用，放不掉）。
// 判 false 就是"现在不能严格放手"，由调用方决定怎么办。
bool ProgramImpl::can_release_continuation_slot_strict(std::uint32_t index) const {
    if (index >= continuation_capacity || !state_store || !text_kv_addresses || !text_kv_pages ||
        continuation_slots[index].role != ContinuationSlotRole::Catalogued) {
        return false;
    }
    const SequenceState& sequence = continuation_states[index];
    if (sequence.state.fork_pending || !sequence.shared_prefix_references.empty() || !sequence.kv ||
        !text_kv_addresses->can_release(sequence.kv->text)) {
        return false;
    }
    if (sequence.kv->backend) {
        if (!backend_kv_addresses || !backend_kv_pages ||
            !backend_kv_addresses->can_release(*sequence.kv->backend)) {
            return false;
        }
    }

    const auto validate_state = [&](StateImageHandle handle, bool release_object) {
        if (!state_store->valid(handle)) { return false; }
        const std::uint32_t owned = owned_checkpoint_references(sequence, handle);
        const std::uint32_t total = state_store->checkpoint_references(handle);
        if (owned > total ||
            (owned != 0 && state_store->role(handle) != StateImageRole::CheckpointImmutable)) {
            return false;
        }
        return !release_object || total != owned ||
               state_store->can_release_after_checkpoint_references(handle, owned);
    };
    const auto repeated_before_anchor = [&](std::size_t anchor_index, StateImageHandle handle) {
        if (handle == sequence.state.read || handle == sequence.state.write ||
            (sequence.rewrite_state && handle == *sequence.rewrite_state)) {
            return true;
        }
        for (std::size_t prior = 0; prior < anchor_index; ++prior) {
            if (sequence.long_anchors[prior].state == handle) { return true; }
        }
        return false;
    };

    if (sequence.endpoint_valid) {
        if (!validate_state(sequence.state.read, !sequence.state.read_has_external_owner() ||
                                                     sequence.state.read == sequence.state.write)) {
            return false;
        }
        if (sequence.state.write != sequence.state.read &&
            !validate_state(sequence.state.write, true)) {
            return false;
        }
    } else if (state_store->valid(sequence.state.read) ||
               state_store->valid(sequence.state.write) || sequence.state.borrows_read()) {
        return false;
    }
    if (sequence.rewrite_state && *sequence.rewrite_state != sequence.state.read &&
        *sequence.rewrite_state != sequence.state.write &&
        !validate_state(*sequence.rewrite_state, true)) {
        return false;
    }
    for (std::size_t anchor = 0; anchor < sequence.long_anchors.size(); ++anchor) {
        const StateImageHandle handle = sequence.long_anchors[anchor].state;
        if (!repeated_before_anchor(anchor, handle) && !validate_state(handle, true)) {
            return false;
        }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool repeated = handle == sequence.state.read || handle == sequence.state.write ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            repeated = repeated || anchor.state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    return true;
}

// 严格放手：前提在 can_ 里已经查过，这里再查一遍只是为了不满足时 terminate——走到这个函数就意味
// 调用方已经承诺过"能放手"，食言属于内部违约。顺序是先把物理资源放干净（KV → 状态）再退休槽位。
void ProgramImpl::release_continuation_slot_strict(std::uint32_t index) noexcept {
    try {
        if (!can_release_continuation_slot_strict(index)) { std::terminate(); }
    } catch (...) { std::terminate(); }
    SequenceState& sequence = continuation_states[index];
    release_sequence_kv_strict(sequence);
    release_sequence_state_strict(sequence);
    retire_continuation_slot(index);
}

// 错误路径上的版本：不检查前提、不承诺结果可用，只保证"该还的都还一遍、槽位一定退休"。
void ProgramImpl::release_continuation_slot_best_effort(std::uint32_t index) noexcept {
    if (index >= continuation_capacity ||
        continuation_slots[index].role == ContinuationSlotRole::Free) {
        return;
    }
    SequenceState& sequence = continuation_states[index];
    release_active_shared_references(sequence);
    release_sequence_kv(sequence);
    release_sequence_state(sequence);
    retire_continuation_slot(index);
}

// 槽位退休：把这条记录代表的一切清成零——两条前沿、账本、前缀身份与摘要、三种 KV 推进度、草稿、
// 各类有效标志、重建配方；解除所有指向它的 lane 绑定；推进代次，于是所有旧句柄一并作废。
// 它只负责"这条记录不再代表任何东西"，物理资源该由调用方在此之前放干净。
void ProgramImpl::retire_continuation_slot(std::uint32_t index) noexcept {
    if (index >= continuation_capacity) { std::terminate(); }
    SequenceState& sequence     = continuation_states[index];
    sequence.execution_frontier = 0;
    sequence.ledger_frontier    = 0;
    sequence.ledger.clear();
    sequence.prefix_identity.clear();
    sequence.prefix_digests.clear();
    sequence.rope_delta              = 0;
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
    sequence.mtp_draft_count         = 0;
    sequence.tail_hidden_valid       = false;
    sequence.endpoint_valid          = false;
    sequence.rewrite_checkpoint      = {};
    sequence.rebuild_work            = {};
    sequence.rebuild_tail_begin      = 0;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (active_continuations[lane] == index) {
            active_continuations[lane] = continuation_capacity;
        }
    }
    ContinuationSlot& slot = continuation_slots[index];
    slot.role              = ContinuationSlotRole::Free;
    if (++slot.generation == 0) { ++slot.generation; }
}

// ============================================================================
// 资源账（program_impl.h 第 3) 段与第 6) 段）
//
// 同一个世界有两种数法，读的时候先分清是哪一种：
//   · owner 的账（owner_exclusive_resources / sequence_exclusive_state_resources）：只算"拆掉这个
//     owner 真的会释放的"——别名共享的分配不算，别人的东西不能算成自己的；
//   · 全局占用（physical_occupancy）：真实总量，是准入与压力判断的基准。
// 缺口（deficit）与峰值可行性都建立在全局占用之上。
// ============================================================================

// 逐份状态镜像数出这条序列独占的部分：主绑定（read / write）、改写检查点、预占的目的地、全部长锚点。
// 顺手校验两处结构性不变式——read/write 要么都有要么都没有；借来的 read 必须正处在 fork 中且与 write
// 不是同一份。借来的源属于别人，不算在这条序列头上。
detail::PhysicalResources
ProgramImpl::sequence_exclusive_state_resources(const SequenceState& sequence) const {
    if (!state_store) {
        throw std::logic_error("sequence StateImage resources have no physical store");
    }
    detail::PhysicalResources out;
    std::array<StateImageHandle, 4> states{};
    std::uint32_t state_count = 0;
    const auto add_state      = [&](StateImageHandle handle) {
        if (!state_store->valid(handle)) {
            throw std::logic_error("sequence owner has a stale StateImage");
        }
        if (!state_exclusive_to_sequence(sequence, handle)) { return; }
        for (std::uint32_t index = 0; index < state_count; ++index) {
            if (states[index] == handle) { return; }
        }
        states[state_count++]                 = handle;
        const StateReplicaResidency residency = state_store->residency(handle);
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.device.state_slots;
        }
        if (residency == StateReplicaResidency::HostOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.host.state_slots;
        }
    };
    const bool has_read_state  = sequence.state.read.valid();
    const bool has_write_state = sequence.state.write.valid();
    if (has_read_state != has_write_state) {
        throw std::logic_error("sequence owner has a partial primary StateImage pair");
    }
    if (sequence.state.borrows_read() &&
        (!sequence.state.fork_pending || sequence.state.read == sequence.state.write)) {
        throw std::logic_error("sequence has an invalid borrowed StateImage source");
    }
    if (has_read_state) {
        if (!sequence.state.borrows_read() || sequence.state.read == sequence.state.write) {
            add_state(sequence.state.read);
        }
        add_state(sequence.state.write);
    }
    if (sequence.rewrite_state) { add_state(*sequence.rewrite_state); }
    if (sequence.reserved_state) { add_state(*sequence.reserved_state); }
    for (std::size_t anchor_index = 0; anchor_index < sequence.long_anchors.size();
         ++anchor_index) {
        const StateImageHandle handle = sequence.long_anchors[anchor_index].state;
        if (!state_store->valid(handle)) {
            throw std::logic_error("sequence owner has a stale long-anchor StateImage");
        }
        if (!state_exclusive_to_sequence(sequence, handle)) { continue; }
        bool seen = false;
        for (std::uint32_t index = 0; index < std::min<std::uint32_t>(state_count, states.size());
             ++index) {
            if (states[index] == handle) { seen = true; }
        }
        for (std::size_t prior = 0; !seen && prior < anchor_index; ++prior) {
            if (sequence.long_anchors[prior].state == handle) { seen = true; }
        }
        if (seen) { continue; }
        const StateReplicaResidency residency = state_store->residency(handle);
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.device.state_slots;
        }
        if (residency == StateReplicaResidency::HostOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.host.state_slots;
        }
    }
    return out;
}

// 在状态账之上再加两种 KV 地址空间。两条数法上的讲究：
//   · 只数**引用数为 1** 的页：还被别的地址引用的页拆掉这个 owner 也释放不了，不算它的效果；
//   · 活跃地址还要把"预留了额度但还没映射成页"的部分算上——那份额度拆掉就能还给池子，是实实在在的
//     占用（resize 之后的预留正是靠这一点被计入压力）。
detail::PhysicalResources
ProgramImpl::owner_exclusive_resources(const SequenceState& sequence) const {
    if (!state_store || !text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("sequence owner resources have no physical stores");
    }
    detail::PhysicalResources out = sequence_exclusive_state_resources(sequence);

    {
        if (!sequence.kv) { throw std::logic_error("sequence owner has no KV address bundle"); }
        const auto add_kv = [&](const KVAddressSpaceStore& addresses,
                                const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                std::uint32_t& device_pages) {
            if (!addresses.valid(address)) { throw std::logic_error("stale KV address space"); }
            for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                // A shared logical page contributes to aggregate occupancy once. Releasing this
                // address cannot free either replica while another address still references it,
                // so it is not part of this owner's exact transition effect.
                if (pages.address_references(logical) > 1) { continue; }
                if (pages.device_resident(logical)) { ++device_pages; }
                if (pages.host_resident(logical)) {
                    if (!host_kv_extents) {
                        throw std::logic_error("missing Host KV extent store");
                    }
                    const HostKVPageReplica& replica = pages.host_replica(logical);
                    const std::size_t stride =
                        host_kv_extents->view(replica.extent).layout().page_stride;
                    if (stride > std::numeric_limits<std::size_t>::max() - out.host.kv_bytes) {
                        throw std::overflow_error("resident Host KV byte count overflow");
                    }
                    out.host.kv_bytes += stride;
                }
            }
            if (addresses.active(address)) {
                const std::uint32_t mapped      = addresses.mapped_pages(address);
                const std::uint32_t entitlement = addresses.entitlement(address);
                if (entitlement < mapped ||
                    entitlement - mapped >
                        std::numeric_limits<std::uint32_t>::max() - device_pages) {
                    throw std::logic_error("owner active KV entitlement is inconsistent");
                }
                device_pages += entitlement - mapped;
            }
        };
        add_kv(*text_kv_addresses, *text_kv_pages, sequence.kv->text, out.device.main_kv_pages);
        if (sequence.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("missing Backend KV stores");
            }
            add_kv(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                   out.device.backend_kv_pages);
        }
    }
    return out;
}

// 共享前缀一侧同理，只是"独占"的判据不同：状态镜像只有在这份共享前缀是**唯一**持有者（引用数为 1）
// 时才算它的——共享前缀的镜像常常同时被别的检查点引用着，那种情况下拆掉它一份镜像也不会消失。
detail::PhysicalResources
ProgramImpl::owner_exclusive_resources(const SharedPrefixState& shared) const {
    if (!state_store || !text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("shared owner resources have no physical stores");
    }
    detail::PhysicalResources out;
    {
        if (!shared.kv || !shared.identity || !state_store->valid(shared.state)) {
            throw std::logic_error("shared prefix has incomplete resident physical state");
        }
        if (state_store->checkpoint_references(shared.state) == 0) {
            throw std::logic_error("shared prefix StateImage has no checkpoint reference");
        }
        const StateReplicaResidency residency = state_store->residency(shared.state);
        if (state_store->checkpoint_references(shared.state) == 1) {
            if (residency == StateReplicaResidency::DeviceOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.device.state_slots;
            }
            if (residency == StateReplicaResidency::HostOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.host.state_slots;
            }
        }
        const auto add_kv = [&](const KVAddressSpaceStore& addresses,
                                const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                std::uint32_t& device_pages) {
            if (!addresses.valid(address)) { throw std::logic_error("stale shared KV address"); }
            for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                if (pages.address_references(logical) != 1) { continue; }
                if (pages.device_resident(logical)) { ++device_pages; }
                if (pages.host_resident(logical)) {
                    if (!host_kv_extents) {
                        throw std::logic_error("missing Host KV extent store");
                    }
                    const HostKVPageReplica& replica = pages.host_replica(logical);
                    const std::size_t stride =
                        host_kv_extents->view(replica.extent).layout().page_stride;
                    if (stride > std::numeric_limits<std::size_t>::max() - out.host.kv_bytes) {
                        throw std::overflow_error("shared Host KV byte count overflow");
                    }
                    out.host.kv_bytes += stride;
                }
            }
        };
        add_kv(*text_kv_addresses, *text_kv_pages, shared.kv->text, out.device.main_kv_pages);
        if (shared.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("missing shared Backend KV stores");
            }
            add_kv(*backend_kv_addresses, *backend_kv_pages, *shared.kv->backend,
                   out.device.backend_kv_pages);
        }
    }
    return out;
}

// 世界的真实总量：活着（非 Empty）的 lane 数、state_store 的设备/Host 占用、两种 KV 池的（已分配 +
// 已预留）页数、Host KV arena 的字节数。noexcept 是有意的——各处兜底路径都要读它。
detail::PhysicalResources ProgramImpl::physical_occupancy() const noexcept {
    detail::PhysicalResources out;
    for (const RequestControl& request : requests) {
        if (request.lifecycle != Lifecycle::Empty) { ++out.device.active_lanes; }
    }
    if (state_store) {
        out.device.state_slots = state_store->device_occupied();
        out.host.state_slots   = state_store->host_occupied();
    }
    if (text_kv_pages) {
        const DeviceKVPagePool& pool = text_kv_pages->physical_pool();
        out.device.main_kv_pages     = pool.allocated_pages() + pool.reserved_pages();
    }
    if (backend_kv_pages) {
        const DeviceKVPagePool& pool = backend_kv_pages->physical_pool();
        out.device.backend_kv_pages  = pool.allocated_pages() + pool.reserved_pages();
    }
    if (host_kv_arena) { out.host.kv_bytes = host_kv_arena->occupied_bytes(); }
    return out;
}

// 裸缺口：(当前占用 + 候选的峰值增量) − 容量，逐维度取正数部分。含义是"这个候选要落地还差多少"。
detail::PhysicalResources
ProgramImpl::materialization_deficit(const ResourceCandidateState& admission) const {
    // 压力是相对这个候选的**真实峰值**而言的。把每个维度都当成稀缺会直接禁掉"从设备降到 Host"这种
    // 动作——哪怕 Host 明明还有余量。
    const detail::PhysicalResources required =
        checked_resource_sum(physical_occupancy(), admission.demand.physical_peak_additional);
    return positive_resource_difference(required, admission_capacity());
}

// 引导缺口：先把这份压力结局折回候选的恒等峰值（加上压力吃掉的、减去压力释放出来的），再数缺口。
// 与裸缺口是两套数法，别混用。
detail::PhysicalResources
ProgramImpl::guided_materialization_deficit(const ResourceCandidateState& admission,
                                            const detail::PhysicalDelta& pressure) const {
    // 压力作用在候选的完整峰值上，而不是它已经截断过的缺口上。否则把一次降级动作直接作用在"Host 缺口
    // 为 0"的状态上，会在 arena 明明还有余量时凭空造出 Host 压力，把启发式引向没必要的破坏。
    const detail::PhysicalResources projected_peak = positive_resource_difference(
        checked_resource_sum(admission.demand.physical_peak_additional, pressure.added),
        pressure.removed);
    const detail::PhysicalResources required =
        checked_resource_sum(physical_occupancy(), projected_peak);
    return positive_resource_difference(required, admission_capacity());
}

// 逐维度判"这份峰值还装得下吗"。用减法（used <= capacity − added）而不是加法，避免两个数加起来溢出。
bool ProgramImpl::physical_peak_fits(detail::PhysicalResources peak) const noexcept {
    const detail::PhysicalResources occupied = physical_occupancy();
    const detail::PhysicalResources limits   = admission_capacity();
    const auto fits_u32 = [](std::uint32_t used, std::uint32_t added, std::uint32_t capacity) {
        return added <= capacity && used <= capacity - added;
    };
    const auto fits_size = [](std::size_t used, std::size_t added, std::size_t capacity) {
        return added <= capacity && used <= capacity - added;
    };
    return fits_u32(occupied.device.active_lanes, peak.device.active_lanes,
                    limits.device.active_lanes) &&
           fits_u32(occupied.device.state_slots, peak.device.state_slots,
                    limits.device.state_slots) &&
           fits_u32(occupied.device.main_kv_pages, peak.device.main_kv_pages,
                    limits.device.main_kv_pages) &&
           fits_u32(occupied.device.backend_kv_pages, peak.device.backend_kv_pages,
                    limits.device.backend_kv_pages) &&
           fits_u32(occupied.host.state_slots, peak.host.state_slots, limits.host.state_slots) &&
           fits_size(occupied.host.kv_bytes, peak.host.kv_bytes, limits.host.kv_bytes);
}

// ============================================================================
// 来源选择（program_impl.h 第 4) 段）
//
// 物化要回答"这次复用哪一份状态镜像"。答案牵动三件事——选谁、这份镜像剩下的引用够不够（不够就得 fork
// 一份新的）、改写检查点还保不保得住——三者必须一起算，单独看任何一个都会得出对不上的账。
// ============================================================================

// 按复用路径挑来源镜像：端点用 read，改写回滚用 rewrite_state，长锚点按 (frontier, ordinal) 精确定位。
// 挑不到就抛——路径与状态不匹配是违约，这里没有"退而求其次"。
StateImageHandle
ProgramImpl::selected_state(const SequenceState& sequence, ReusePath reuse,
                            std::optional<runtime::CheckpointRef> checkpoint) const {
    if (reuse == ReusePath::PrivateEndpoint) {
        if (!sequence.endpoint_valid || !state_store->valid(sequence.state.read)) {
            throw std::logic_error("private endpoint StateImage is stale");
        }
        return sequence.state.read;
    }
    if (is_rewrite_checkpoint_restore(reuse) && sequence.rewrite_state &&
        state_store->valid(*sequence.rewrite_state)) {
        return *sequence.rewrite_state;
    }
    if (reuse == ReusePath::PrivateLongAnchor) {
        if (!checkpoint || checkpoint->kind != runtime::CheckpointKind::LongAnchor) {
            throw std::logic_error("long-anchor materialization has no selected checkpoint");
        }
        const auto anchor = std::find_if(sequence.long_anchors.begin(), sequence.long_anchors.end(),
                                         [&](const LongAnchorCheckpoint& candidate) {
                                             return candidate.frontier == checkpoint->frontier &&
                                                    candidate.ordinal == checkpoint->ordinal;
                                         });
        if (anchor != sequence.long_anchors.end() && state_store->valid(anchor->state)) {
            return anchor->state;
        }
    }
    throw std::logic_error("materialization path has no selected StateImage");
}

// 这次复用会**消耗掉**被选镜像的几个引用：改写检查点被移交出去算一个，早于复用点的长锚点被顶掉也各
// 算一个（前沿还在复用点之后的长锚点则保留）。要算出"用完还剩几个引用、要不要 fork"，就得先知道消耗
// 多少——这也是它与 selected_state 必须成对使用的原因。
std::uint32_t
ProgramImpl::selected_state_consumed_references(const SequenceState& sequence, ReusePath reuse,
                                                RewriteCheckpointDisposition rewrite_disposition,
                                                std::optional<runtime::CheckpointRef> checkpoint,
                                                std::uint32_t reuse_base) const {
    const StateImageHandle selected   = selected_state(sequence, reuse, checkpoint);
    std::uint32_t consumed_references = 0;
    if (is_rewrite_checkpoint_restore(reuse) &&
        rewrite_disposition != RewriteCheckpointDisposition::RetainExisting) {
        if (!sequence.rewrite_state || *sequence.rewrite_state != selected) {
            throw std::logic_error("selected rewrite StateImage is unavailable");
        }
        consumed_references = 1;
    } else if (reuse == ReusePath::PrivateEndpoint &&
               rewrite_disposition != RewriteCheckpointDisposition::RetainExisting &&
               sequence.rewrite_state && *sequence.rewrite_state == selected) {
        consumed_references = 1;
    }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        if (anchor.frontier > reuse_base && anchor.state == selected) {
            if (consumed_references == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("consumed StateImage reference inventory overflow");
            }
            ++consumed_references;
        }
    }
    const std::uint32_t references = state_store->checkpoint_references(selected);
    if (consumed_references > references) {
        throw std::logic_error("selected StateImage reference inventory is inconsistent");
    }
    return consumed_references;
}

// 要不要为这次复用开一份新镜像：引用总数正好等于要被消耗掉的数量，说明用完这份镜像就归我们了，可以
// 原地接手；只要还多出引用，就说明别人（别的检查点、别的请求）也指着它，必须 fork。这就是"要不要
// fork"的全部依据。
bool ProgramImpl::selected_state_requires_fork(const SequenceState& sequence, ReusePath reuse,
                                               RewriteCheckpointDisposition rewrite_disposition,
                                               std::optional<runtime::CheckpointRef> checkpoint,
                                               std::uint32_t reuse_base) const {
    const StateImageHandle selected = selected_state(sequence, reuse, checkpoint);
    return state_store->checkpoint_references(selected) !=
           selected_state_consumed_references(sequence, reuse, rewrite_disposition, checkpoint,
                                              reuse_base);
}

// 现有改写检查点能不能原样留着。三道关：检查点存在且镜像还有效、检查点前沿与这条序列的前缀账本确实
// 对得上（账本可能已经被改写改过）、以及前沿关系成立——前沿相同直接可留；否则只有"这次复用正好从它
// 接上"（复用点等于检查点前沿且新前沿不超过它）时才允许留——那种情况下它仍是合法的回滚点。
bool ProgramImpl::can_retain_rewrite_checkpoint(const PreparedPromptData& prompt,
                                                const RewriteCheckpointSpec& desired,
                                                const SequenceState& sequence, ReusePath reuse,
                                                std::uint32_t reuse_base) const {
    if (!sequence.rewrite_checkpoint.valid || !sequence.rewrite_state ||
        !state_store->valid(*sequence.rewrite_state) ||
        !qwen3_5::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                         sequence.rewrite_checkpoint.frontier)) {
        return false;
    }
    if (sequence.rewrite_checkpoint.frontier == desired.frontier) { return true; }
    return is_rewrite_checkpoint_restore(reuse) &&
           sequence.rewrite_checkpoint.frontier == reuse_base && desired.frontier <= reuse_base;
}

// ============================================================================
// KV 前缀数法与对外摘要（program_impl.h 第 5) 段）
//
// 前半是几个"数页数"的小函数：常驻多少、共享多少、缺多少、尾部要不要 CoW；后半是把内部账本翻译成对外
// 摘要。两半的关系是硬性的——外部拿到的摘要必须与内部账本同源，否则外部会按错的账做决定。所以这些
// 数字只在这里数一次，别在别处再数一遍。
// ============================================================================

// 到 frontier 为止，设备上**常驻**的页数（注意不是成员页数，也不是额度——成员页可以只在 Host 上）。
std::uint32_t ProgramImpl::device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                  KVAddressSpaceHandle address,
                                                  std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t resident = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        if (pages.device_resident(addresses.logical_page(address, page))) { ++resident; }
    }
    return resident;
}

// 到 frontier 为止，与别人共享（引用数 > 1）的页数。非对齐的尾页不算：它随后会被复制成私有页，不具
// 备"共享"的语义。
std::uint32_t ProgramImpl::shared_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                  KVAddressSpaceHandle address,
                                                  std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t shared = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        if (pages.address_references(addresses.logical_page(address, page)) <= 1) { continue; }
        if (page + 1U == required && frontier % static_cast<std::uint32_t>(kPagedKVPageSize) != 0) {
            continue;
        }
        ++shared;
    }
    return shared;
}

// 共享**且**设备常驻的页数：共享里真正省下设备页的那一部分（共享但只存在于 Host 的页不省设备）。
std::uint32_t ProgramImpl::shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                         KVAddressSpaceHandle address,
                                                         std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t resident = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        const LogicalKVPageHandle logical = addresses.logical_page(address, page);
        if (pages.address_references(logical) > 1 && pages.device_resident(logical)) { ++resident; }
    }
    return resident;
}

// 前沿落在一页中间时，那半页是否需要私有化：要么它与别人共享（不能就地改它），要么它根本不在设备上。
// 页对齐的前沿没有尾页，自然不需要。
bool ProgramImpl::partial_tail_cow_required(const KVAddressSpaceStore& addresses,
                                            KVAddressSpaceHandle address,
                                            std::uint32_t frontier) const {
    if (frontier == 0 || frontier % static_cast<std::uint32_t>(kPagedKVPageSize) == 0) {
        return false;
    }
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    const LogicalKVPageHandle tail = addresses.logical_page(address, required - 1U);
    return pages.address_references(tail) > 1 || !pages.device_resident(tail);
}

// 共享但不在设备上的页数——要靠搬运补齐的那部分（恢复或复用另一个 owner 的缓存时，正是这些页需要
// 从 Host 搬回设备）。
std::uint32_t
ProgramImpl::missing_shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                   KVAddressSpaceHandle address,
                                                   std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t missing = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        const LogicalKVPageHandle logical = addresses.logical_page(address, page);
        if (pages.address_references(logical) > 1 && !pages.device_resident(logical)) { ++missing; }
    }
    return missing;
}

// 到 frontier 为止 Host 侧属于这个 owner 的字节数：共享页（引用数 > 1）和不在 Host 上的页都不算，
// 需要 CoW 的尾页也不算（那页的旧副本马上就会失效）。最后一类是要害：列数与前沿不一致的尾页说明它
// 经历过破坏性私有改写，内容 epoch 已经变了，旧的 Host 副本不能再算进活跃额度。noexcept——估量用的
// 数字，数不出来就报 0，不该让调用方因此失败。
std::size_t ProgramImpl::host_kv_prefix_bytes(const KVAddressSpaceStore& addresses,
                                              KVAddressSpaceHandle address,
                                              std::uint32_t frontier) const noexcept {
    if (!host_kv_extents) { return 0; }
    try {
        const LogicalKVPageStore& pages =
            (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
        const std::uint32_t required_pages = kv_pages_for_frontier(frontier);
        if (required_pages > addresses.mapped_pages(address)) { return 0; }
        std::size_t bytes = 0;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (pages.address_references(logical) > 1) { continue; }
            if (!pages.host_resident(logical)) { continue; }
            if (page + 1U == required_pages &&
                frontier % static_cast<std::uint32_t>(kPagedKVPageSize) != 0 &&
                partial_tail_cow_required(addresses, address, frontier)) {
                continue;
            }
            const std::uint32_t begin = page * static_cast<std::uint32_t>(kPagedKVPageSize);
            const std::uint32_t selected_columns =
                std::min(static_cast<std::uint32_t>(kPagedKVPageSize), frontier - begin);
            if (selected_columns != pages.committed_columns(logical)) {
                // 破坏性私有改写会让这页尾页的内容 epoch 变掉，旧的 Host 副本不能再留在活跃额度里。
                continue;
            }
            const std::size_t stride =
                host_kv_extents->view(pages.host_replica(logical).extent).layout().page_stride;
            if (stride > std::numeric_limits<std::size_t>::max() - bytes) { return 0; }
            bytes += stride;
        }
        return bytes;
    } catch (...) { return 0; }
}

// 把内部的一份检查点翻译成对外摘要：身份短名单键、状态常驻形态、需要多少 KV、重建配方。
// shortlist_key 里的 identity_tag 由后端 / proposal_head / KV 存储形态拼成——摘要只在同一套配置下
// 才可比，换了后端就不是同一个身份。后端前沿按后端类型换算（MTP 少一，DFlash 与主前沿相同，普通
// 解码为 0），这是"这份检查点的 KV 需求"里唯一随后端变的一项。
qwen3_5::CheckpointSummary
ProgramImpl::checkpoint_summary(const SequenceState& sequence, runtime::CheckpointRef checkpoint,
                                StateImageHandle state, runtime::PrefillWork rebuild_work) const {
    if (!sequence.kv) { throw std::logic_error("checkpoint summary has no KV address space"); }
    if (checkpoint.frontier == 0) {
        throw std::logic_error("checkpoint summary has an empty frontier");
    }
    if (!state_store->valid(state)) {
        throw std::logic_error("checkpoint summary has a stale StateImage");
    }
    const StateReplicaResidency state_location = state_store->residency(state);
    runtime::ReplicaResidency residency        = runtime::ReplicaResidency::DeviceOnly;
    if (state_location == StateReplicaResidency::HostOnly) {
        residency = runtime::ReplicaResidency::HostOnly;
    } else if (state_location == StateReplicaResidency::Both) {
        residency = runtime::ReplicaResidency::Both;
    } else if (state_location != StateReplicaResidency::DeviceOnly) {
        throw std::logic_error("checkpoint StateImage has no published replica");
    }
    const std::uint32_t backend_frontier =
        speculative_backend == SpeculativeBackend::Mtp      ? checkpoint.frontier - 1U
        : speculative_backend == SpeculativeBackend::DFlash ? checkpoint.frontier
                                                            : 0U;
    const std::uint32_t identity_tag = static_cast<std::uint32_t>(speculative_backend) |
                                       (static_cast<std::uint32_t>(proposal_head) << 8U) |
                                       (static_cast<std::uint32_t>(kv_storage) << 16U);
    return qwen3_5::CheckpointSummary{
        .ref   = checkpoint,
        .scope = runtime::CheckpointScope::Private,
        .shortlist_key =
            {
                .digests      = sequence.prefix_digests.at(checkpoint.frontier),
                .frontier     = checkpoint.frontier,
                .identity_tag = identity_tag,
            },
        .state_residency = residency,
        .required_kv =
            {
                .main_frontier    = checkpoint.frontier,
                .backend_frontier = backend_frontier,
                .main_pages       = kv_pages_for_frontier(checkpoint.frontier),
                .backend_pages    = kv_pages_for_frontier(backend_frontier),
            },
        .rebuild_work = validated_rebuild_work(rebuild_work, checkpoint.frontier),
    };
}

// 一条续跑对外报告的**全部**检查点：端点 + 改写检查点 + 所有长锚点。这里只是摘要的壳，
// 真正逐项填的是下面那个 populate（同一份填充逻辑还要被增量刷新的路径复用）。
qwen3_5::ContinuationSummary
ProgramImpl::continuation_summary(const SequenceState& sequence) const {
    qwen3_5::ContinuationSummary summary;
    summary.long_anchors.reserve(sequence.long_anchors.size());
    populate_continuation_summary(sequence, summary);
    return summary;
}

// 逐项填充：端点（read 镜像 + 执行前沿）、改写检查点、长锚点按 ordinal 顺序；写完再校验至少存在
// 一份检查点——一条私有续跑却一份检查点都没有属于违约。最后顺手报告这份续跑自己是不是还活着：
// active_references 看它是否仍绑在 Active 槽上（不绑定就填 0，是正常状态，不是错误）。
void ProgramImpl::populate_continuation_summary(const SequenceState& sequence,
                                                qwen3_5::ContinuationSummary& summary) const {
    validate_long_anchor_ordinals(sequence.long_anchors,
                                  context_cache.max_long_anchors_per_continuation.value_or(0));
    if (summary.long_anchors.capacity() < sequence.long_anchors.size()) {
        throw std::logic_error("continuation summary backing was not reserved");
    }
    summary.endpoint.reset();
    summary.rewrite.reset();
    summary.long_anchors.clear();
    summary.active_references = 0;
    if (sequence.endpoint_valid) {
        const runtime::CheckpointRef endpoint{
            .kind     = runtime::CheckpointKind::SessionEndpoint,
            .frontier = sequence.execution_frontier,
        };
        runtime::PrefillWork endpoint_work = sequence.rebuild_work;
        summary.endpoint =
            checkpoint_summary(sequence, endpoint, sequence.state.read, endpoint_work);
    }
    if (sequence.rewrite_checkpoint.valid) {
        if (!sequence.rewrite_state) {
            throw std::logic_error("rewrite checkpoint has no StateImage");
        }
        const runtime::CheckpointRef rewrite{
            .kind     = checkpoint_kind(sequence.rewrite_checkpoint.kind),
            .frontier = sequence.rewrite_checkpoint.frontier,
        };
        summary.rewrite = checkpoint_summary(sequence, rewrite, *sequence.rewrite_state,
                                             sequence.rewrite_checkpoint.rebuild_work);
    }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        summary.long_anchors.push_back(
            checkpoint_summary(sequence,
                               runtime::CheckpointRef{.kind = runtime::CheckpointKind::LongAnchor,
                                                      .frontier = anchor.frontier,
                                                      .ordinal  = anchor.ordinal},
                               anchor.state, anchor.rebuild_work));
    }
    if (!summary.endpoint && !summary.rewrite && summary.long_anchors.empty()) {
        throw std::logic_error("private continuation has no checkpoint");
    }
    const auto* begin = continuation_states.data();
    const auto* end   = begin + continuation_capacity;
    if (&sequence >= begin && &sequence < end) {
        const std::size_t index = static_cast<std::size_t>(&sequence - begin);
        summary.active_references =
            continuation_slots[index].role == ContinuationSlotRole::Active ? 1U : 0U;
    }
}

// 共享前缀的对外摘要：与私有检查点同构的一份描述，外加被引用计数。与 checkpoint_summary 的两点
// 差别：身份直接取自共享身份，不需要再拼 identity_tag；两种前沿（主 KV 与后端 KV）各自独立，
// 因为它们本来就可能不同步。
qwen3_5::SharedPrefixSummary
ProgramImpl::shared_prefix_summary(const SharedPrefixState& shared) const {
    if (!shared.kv || !shared.identity || shared.frontier == 0 ||
        !state_store->valid(shared.state)) {
        throw std::logic_error("shared-prefix summary source is incomplete");
    }
    const StateReplicaResidency state_location = state_store->residency(shared.state);
    runtime::ReplicaResidency residency        = runtime::ReplicaResidency::DeviceOnly;
    if (state_location == StateReplicaResidency::HostOnly) {
        residency = runtime::ReplicaResidency::HostOnly;
    } else if (state_location == StateReplicaResidency::Both) {
        residency = runtime::ReplicaResidency::Both;
    } else if (state_location != StateReplicaResidency::DeviceOnly) {
        throw std::logic_error("shared-prefix StateImage has no published replica");
    }
    return qwen3_5::SharedPrefixSummary{
        .checkpoint =
            {
                .ref =
                    {
                        .kind     = runtime::CheckpointKind::SharedStablePrefix,
                        .frontier = shared.frontier,
                    },
                .scope           = runtime::CheckpointScope::Shared,
                .shortlist_key   = shared.identity->shortlist_key,
                .state_residency = residency,
                .required_kv =
                    {
                        .main_frontier    = shared.frontier,
                        .backend_frontier = shared.backend_frontier,
                        .main_pages       = kv_pages_for_frontier(shared.frontier),
                        .backend_pages    = kv_pages_for_frontier(shared.backend_frontier),
                    },
                .rebuild_work = validated_rebuild_work(shared.rebuild_work, shared.frontier),
            },
        .active_references = shared.active_references,
    };
}

// 预填推进一步（流程本体在 prefill.cpp，这里只做前置检查、进度包装与失败收场）：要求这条序列确实
// 在 Prefilling、句柄有效、且没有未结算的事务在飞。失败时先把这条 lane 收拾干净再原样抛出——失败
// 的序列不能继续挂在账上；清场耗掉的 Host 时间也计进去，否则计时账会对不上。
PrefillProgress ProgramImpl::advance_prefill(SequenceHandle sequence,
                                             runtime::ExecutionTiming* failed_timing) {
    if (pending_transaction_ || !valid_sequence(sequence)) {
        throw std::logic_error("prefill sequence capability is invalid");
    }
    const std::uint32_t lane = ContractAccess::lane(sequence).value;
    if (requests[lane].lifecycle != Lifecycle::Prefilling) {
        throw std::logic_error("prefill advance requires a prefilling sequence");
    }
    try {
        runtime::PrefillStepResult step = advance_prefill_raw(lane, failed_timing);
        if (failed_timing != nullptr) { *failed_timing += step.timing; }
        return wrap_prefill(lane, std::move(step));
    } catch (...) {
        const Clock::time_point cleanup_started = Clock::now();
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        if (failed_timing != nullptr) {
            failed_timing->post_host_ns += elapsed_ns(cleanup_started);
        }
        throw;
    }
}

// ================================================================================================
// lane 清场（program_impl.h 第 8) 段的清场部分）
//
// 两档放手，用途不同：strict 是**有序收场**——先把前提全查一遍（can_clear_lane_strict），任何一条
// 不成立就报告失败、什么都不动，成功后才按依赖顺序逐个释放；best_effort 是**错误路径上的兜底**，
// 只求别崩、别漏账，不保证所有资源都能真的还回去。
//
// 顺序不能乱：先摘共享前缀引用（别人还等着这次放手把引用计数降下来），再放 KV 地址，最后放状态
// 镜像与槽位——每个前置步骤都依赖前一步已经交还了东西。
// ================================================================================================

// 清场前置检查：能不能干净地把这条 lane 放掉。逐项确认——续跑槽位确实处于 Active、两套 KV 地址都
// 能"停用后释放"、共享前缀的引用计数够扣（同一份前缀被引用多次时要逐次累加核对）、状态镜像的引用
// 账都归位（未结算的 fork 要么能中止、要么已经妥当），且每个待释放的镜像都确实是"释放后没人再引用"
// 的状态。这里只查不动，任何异常都当作"不能清"。
bool ProgramImpl::can_clear_lane_strict(const SequenceState& sequence) const {
    const auto* begin = continuation_states.data();
    const auto* end   = begin + continuation_capacity;
    if (&sequence < begin || &sequence >= end || !state_store || !text_kv_addresses ||
        !text_kv_pages || !sequence.kv) {
        return false;
    }
    const std::uint32_t continuation = static_cast<std::uint32_t>(&sequence - begin);
    if (continuation_slots[continuation].role != ContinuationSlotRole::Active ||
        !text_kv_addresses->can_release_after_deactivate(sequence.kv->text) ||
        (sequence.kv->backend &&
         (!backend_kv_addresses || !backend_kv_pages ||
          !backend_kv_addresses->can_release_after_deactivate(*sequence.kv->backend)))) {
        return false;
    }

    for (std::size_t position = 0; position < sequence.shared_prefix_references.size();
         ++position) {
        const std::uint32_t index = sequence.shared_prefix_references[position];
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued) {
            return false;
        }
        const std::uint32_t required = static_cast<std::uint32_t>(std::count(
            sequence.shared_prefix_references.begin(),
            sequence.shared_prefix_references.begin() + static_cast<std::ptrdiff_t>(position + 1U),
            index));
        if (shared_prefix_states[index].active_references < required) { return false; }
    }

    if (!state_store->valid(sequence.state.read) || !state_store->valid(sequence.state.write) ||
        (sequence.state.fork_pending &&
         (!sequence.state.borrows_read() ||
          !state_store->can_abort_fork(sequence.state.read, sequence.state.write)))) {
        return false;
    }
    enum class ForkEndpoint : std::uint8_t { None, Source, Destination };
    const auto validate_state = [&](StateImageHandle handle, bool release_object,
                                    ForkEndpoint fork_endpoint = ForkEndpoint::None) {
        if (!state_store->valid(handle)) { return false; }
        const std::uint32_t owned = owned_checkpoint_references(sequence, handle);
        const std::uint32_t total = state_store->checkpoint_references(handle);
        if (owned > total ||
            (owned != 0 && state_store->role(handle) != StateImageRole::CheckpointImmutable &&
             fork_endpoint != ForkEndpoint::Destination)) {
            return false;
        }
        if (!release_object || total != owned) { return true; }
        if (fork_endpoint == ForkEndpoint::Source) {
            return state_store->can_release_source_after_fork_abort(sequence.state.read,
                                                                    sequence.state.write, owned);
        }
        if (fork_endpoint == ForkEndpoint::Destination) {
            return state_store->can_release_destination_after_fork_abort(
                sequence.state.read, sequence.state.write, owned);
        }
        return state_store->can_release_after_checkpoint_references(handle, owned);
    };
    const auto duplicates_binding = [&](StateImageHandle handle) {
        return handle == sequence.state.read || handle == sequence.state.write;
    };

    if (!validate_state(sequence.state.read,
                        !sequence.state.read_has_external_owner() ||
                            sequence.state.read == sequence.state.write,
                        sequence.state.fork_pending ? ForkEndpoint::Source : ForkEndpoint::None)) {
        return false;
    }
    if (sequence.state.write != sequence.state.read &&
        !validate_state(sequence.state.write, true,
                        sequence.state.fork_pending ? ForkEndpoint::Destination
                                                    : ForkEndpoint::None)) {
        return false;
    }
    if (sequence.rewrite_state && !duplicates_binding(*sequence.rewrite_state) &&
        !validate_state(*sequence.rewrite_state, true)) {
        return false;
    }
    for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
        const StateImageHandle handle = sequence.long_anchors[index].state;
        bool repeated                 = duplicates_binding(handle) ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (std::size_t prior = 0; !repeated && prior < index; ++prior) {
            repeated = sequence.long_anchors[prior].state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool repeated                 = duplicates_binding(handle) ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            repeated = repeated || anchor.state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    return true;
}

// 清场第一步：把这条序列对共享前缀的活跃引用全部摘掉。strict 的意思是**这里不该出现意外**——
// 槽位不在 Catalogued 角色或引用计数已经为 0，都是账目违约，直接 terminate 而不是将就往下走。
// （best-effort 的对应物是文件后段的 release_active_shared_references。）
void ProgramImpl::release_active_shared_references_strict(SequenceState& sequence) noexcept {
    for (const std::uint32_t index : sequence.shared_prefix_references) {
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
            shared_prefix_states[index].active_references == 0) {
            std::terminate();
        }
        --shared_prefix_states[index].active_references;
    }
    sequence.shared_prefix_references.clear();
}

// 有序清场：先问 can_clear_lane_strict（查不过就返回 false，调用方自己决定怎么办），通过了才依次
// 摘共享引用 → 放 KV → 放状态镜像 → 退休槽位，最后把请求控制块复位成"空 lane 且待发布续跑"。
// noexcept 是刻意的：走到这里已经不允许失败，任何退出路径都必须把 lane 留在一致状态。
bool ProgramImpl::clear_lane_strict(SequenceState& sequence, RequestControl& request) noexcept {
    try {
        if (!can_clear_lane_strict(sequence)) { return false; }
    } catch (...) { return false; }
    const auto* begin                = continuation_states.data();
    const std::uint32_t continuation = static_cast<std::uint32_t>(&sequence - begin);
    release_active_shared_references_strict(sequence);
    release_active_sequence_kv_strict(sequence);
    release_active_sequence_state_strict(sequence);
    retire_continuation_slot(continuation);
    request.prefill.reset();
    request.lifecycle            = Lifecycle::Empty;
    request.pending              = {};
    request.active_resources     = {};
    request.optional_resources   = {};
    request.publish_continuation = true;
    return true;
}

// 执行失败后的清场入口：把涉及的 lane 逐个 best-effort 放掉并作废（invalidate_lane 会推代次，
// 于是外面还攥着的旧句柄再也对不上号）。但进行中的资源事务可能正掐着或正要读这些活跃 owner，
// 所以有事务在飞时直接返回——引擎级的清理会先中止事务，再来放手；顺序反了就会读到半释放的资源。
void ProgramImpl::clear_execution_failure_lanes(std::span<const std::uint32_t> lanes) noexcept {
    if (has_context_transaction()) { return; }
    for (const std::uint32_t lane : lanes) {
        if (lane >= max_concurrency || active_continuations[lane] >= continuation_capacity) {
            continue;
        }
        clear_lane_best_effort(active_sequence(lane), requests[lane]);
        invalidate_lane(lane);
    }
}

// 兜底清场：不查前提、不假设任何东西成立。只做两件事——控制块复位成空 lane，槽位走 best-effort
// 释放（放不掉也认了）。已经在错误路径上，能恢复的账尽量恢复，恢复不了的留给错误上报。
void ProgramImpl::clear_lane_best_effort(SequenceState& sequence,
                                         RequestControl& request) noexcept {
    request.prefill.reset();
    request.lifecycle            = Lifecycle::Empty;
    request.pending              = {};
    request.active_resources     = {};
    request.optional_resources   = {};
    request.publish_continuation = true;
    const auto* begin            = continuation_states.data();
    const auto* end              = begin + continuation_capacity;
    if (&sequence >= begin && &sequence < end) {
        release_continuation_slot_best_effort(static_cast<std::uint32_t>(&sequence - begin));
    }
}

// ================================================================================================
// 状态镜像的生命周期（program_impl.h 第 8) 段的状态部分）
//
// 一份镜像的生死只看引用数：谁引用了它，谁就要还。序列自己持有的引用分三类——活跃绑定的读写视图、
// 改写检查点、长锚点（外加预留给下一次 fork 的目标槽）。放手分三档：active 变体处理"序列还活着"的
// 场景（可能带着未结算的 fork，先中止再放），检查点场景用普通变体（有未结算的 fork 就直接失败），
// 错误路径用 best-effort 变体。
//
// 别名是要害：同一个镜像可能既被活跃绑定指着、又是改写检查点、还是某个长锚点，所以每个释放点都
// 必须先判重——重复释放同一份镜像是这里最容易犯且最致命的错。
// ================================================================================================

// 活跃绑定对应的设备槽位（供执行期读取）；顺带校验读写视图的角色配对（原地写还是 fork 目标）。
StateImageSelectors ProgramImpl::state_selectors(const SequenceState& sequence) const {
    if (!state_store || !state_store->valid(sequence.state.read) ||
        !state_store->valid(sequence.state.write)) {
        throw std::logic_error("sequence has no active StateImage binding");
    }
    return state_store->selectors(sequence.state.read, sequence.state.write);
}

// 这条序列自己在这份镜像上挂了几份检查点引用（改写检查点 + 每个引用它的长锚点各算一份）。
// 活跃绑定不算检查点引用，它的引用另记。noexcept：纯计数，调用方常在检查与释放路径上用它。
std::uint32_t ProgramImpl::owned_checkpoint_references(const SequenceState& sequence,
                                                       StateImageHandle state) const noexcept {
    std::uint32_t references = 0;
    if (sequence.rewrite_state && *sequence.rewrite_state == state) { ++references; }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        if (anchor.state == state) { ++references; }
    }
    return references;
}

// 这份镜像是不是**只**被这条序列引用（全库检查点引用数 == 它自己的份额）。用于判断释放之后镜像
// 会不会真的消失，从而决定要不要把它算进本次放手能归还的资源里。
bool ProgramImpl::state_exclusive_to_sequence(const SequenceState& sequence,
                                              StateImageHandle state) const noexcept {
    if (!state_store || !state_store->valid(state)) { return false; }
    return state_store->checkpoint_references(state) ==
           owned_checkpoint_references(sequence, state);
}

// 把 tail_hidden / rewrite_checkpoint_hidden 这两个**设备侧隐藏态视图**重新指到对应镜像的物理槽上。
// 隐藏态是执行期直接访问的裸指针，所以镜像一动（换镜子、settle fork、清场）就必须重指，否则会指到
// 已经不属于这条序列的内存。只有落在设备上的镜像才有隐藏态视图，HostOnly 的一律置空。
// 注意 fork 未结算时"已提交"的那一份是 read 侧，不是 write 侧。
void ProgramImpl::refresh_state_views(SequenceState& sequence) {
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
    if (state_store->valid(sequence.state.read) && state_store->valid(sequence.state.write) &&
        state_store->residency(sequence.state.read) != StateReplicaResidency::HostOnly &&
        state_store->residency(sequence.state.write) != StateReplicaResidency::HostOnly) {
        const StateImageHandle committed =
            sequence.state.fork_pending ? sequence.state.read : sequence.state.write;
        sequence.tail_hidden =
            state_images->continuation_hidden_slot(state_store->physical_slot(committed));
    }
    if (sequence.rewrite_state && state_store->valid(*sequence.rewrite_state) &&
        state_store->residency(*sequence.rewrite_state) != StateReplicaResidency::HostOnly) {
        sequence.rewrite_checkpoint_hidden = state_images->continuation_hidden_slot(
            state_store->physical_slot(*sequence.rewrite_state));
    }
}

// 把状态槽位的额度抬到 slots：只允许**恰好加一个**目标槽（fork 的落点），多一个少一个都违约。
// 抬完立刻复算一次额度来确认"预留真的兑现成了额度"——这一步是把 promise 变成账目的核验点。
void ProgramImpl::reserve_state_entitlement(SequenceState& sequence, std::uint32_t slots) {
    const std::uint32_t owned = sequence_exclusive_state_resources(sequence).device.state_slots;
    if (slots == 0 || owned > slots) {
        throw std::logic_error("sequence StateImage entitlement is inconsistent");
    }
    if (owned == slots) { return; }
    if (slots - owned != 1 || sequence.reserved_state) {
        throw std::logic_error("sequence StateImage reservation is not a single destination");
    }
    std::optional<StateImageHandle> reserved = state_store->reserve_destination();
    if (!reserved) { throw std::bad_alloc(); }
    sequence.reserved_state = *reserved;
    if (sequence_exclusive_state_resources(sequence).device.state_slots != slots) {
        throw std::logic_error("sequence StateImage entitlement did not materialize exactly");
    }
}

// 结算一次 fork：写侧的改动被认定成正式的下一步，于是 read = write = 目标槽。源镜像若已经没人引用
// （不是检查点、也不归外部所有）就顺手收回；外部拥有的源不能碰——那是别人借给我们的。
// 事务在飞时不许结算：这时候资源账正被读，换了绑定会让账目自相矛盾。
void ProgramImpl::settle_state_fork(SequenceState& sequence) {
    if (!sequence.state.fork_pending) { return; }
    if (has_context_transaction()) {
        throw std::logic_error("StateImage Fork settlement overlaps a resource transaction");
    }
    const StateImageHandle source      = sequence.state.read;
    const StateImageHandle destination = sequence.state.write;
    const bool external_source         = sequence.state.read_has_external_owner();
    state_store->commit_fork(source, destination);
    sequence.state = ActiveStateBinding{.read = destination, .write = destination};
    if (!external_source && state_store->checkpoint_references(source) == 0 &&
        !state_store->release(source)) {
        throw std::logic_error("unreferenced StateImage fork source could not be released");
    }
    refresh_state_views(sequence);
}

// 全库有没有任何一条活着的 lane 还挂着未结算的 fork。资源事务要不要为 fork 目标留位置、能不能安全
// 地全局回收，都先问这个。
bool ProgramImpl::has_unsettled_state_fork() const noexcept {
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        const std::uint32_t continuation = active_continuations[lane];
        if (continuation < continuation_capacity &&
            continuation_states[continuation].state.fork_pending) {
            return true;
        }
    }
    return false;
}

// 清场用的严格放手（序列**还活着**，但要把这条 lane 收掉）：先把未结算的 fork 中止掉——fork 的
// 目标槽本来就是给"还没决定要不要走的一步"准备的，lane 都不要了，这步自然作废。然后逐个还引用、
// 收回没人再引用的镜像。strict 的含义是：这里任何一步不成立就 terminate——能走到这条路径说明前置
// 检查已经过了，再出问题说明程序有 bug，硬撑着继续只会把账搞得更烂。收尾把序列的状态字段全部归零。
void ProgramImpl::release_active_sequence_state_strict(SequenceState& sequence) noexcept {
    const auto fail = []() noexcept { std::terminate(); };
    if (!state_store) { fail(); }
    try {
        if (sequence.state.fork_pending) {
            state_store->abort_fork(sequence.state.read, sequence.state.write);
        }
        if (sequence.rewrite_state) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            state_store->release_checkpoint_reference(anchor.state);
        }

        const auto release_if_unreferenced = [&](StateImageHandle handle, bool lifetime_owned) {
            if (!lifetime_owned || !state_store->valid(handle) ||
                state_store->checkpoint_references(handle) != 0) {
                return;
            }
            if (!state_store->release(handle)) { fail(); }
        };
        const auto duplicates_binding = [&](StateImageHandle handle) {
            return handle == sequence.state.read || handle == sequence.state.write;
        };

        release_if_unreferenced(sequence.state.write, true);
        if (sequence.state.read != sequence.state.write) {
            release_if_unreferenced(sequence.state.read, !sequence.state.read_has_external_owner());
        }
        if (sequence.rewrite_state) {
            release_if_unreferenced(*sequence.rewrite_state,
                                    !duplicates_binding(*sequence.rewrite_state));
        }
        for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
            const StateImageHandle handle = sequence.long_anchors[index].state;
            bool repeated                 = duplicates_binding(handle) ||
                            (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (std::size_t prior = 0; !repeated && prior < index; ++prior) {
                repeated = sequence.long_anchors[prior].state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
        if (sequence.reserved_state) {
            const StateImageHandle handle = *sequence.reserved_state;
            bool repeated                 = duplicates_binding(handle) ||
                            (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
                repeated = repeated || anchor.state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
    } catch (...) { fail(); }

    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
}

// 检查点场景的严格放手（序列已经不在活跃表里，剩下的只是它留下的检查点）。与 active 变体的关键
// 差别：这里**不允许**出现未结算的 fork——没有活跃执行却挂着 fork，本身就是不可能的状态，直接失败。
// 另一处差别是可回放性：只有 endpoint_valid 时读写视图才算"由这条序列负责"，否则它们只是借来的。
void ProgramImpl::release_sequence_state_strict(SequenceState& sequence) noexcept {
    const auto fail = []() noexcept { std::terminate(); };
    if (!state_store || sequence.state.fork_pending) { fail(); }

    try {
        if (sequence.rewrite_state) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            state_store->release_checkpoint_reference(anchor.state);
        }

        const auto release_if_unreferenced = [&](StateImageHandle handle, bool lifetime_owned) {
            if (!lifetime_owned || !state_store->valid(handle) ||
                state_store->checkpoint_references(handle) != 0) {
                return;
            }
            if (!state_store->release(handle)) { fail(); }
        };
        const auto repeated_before_anchor = [&](std::size_t anchor_index, StateImageHandle handle) {
            if ((sequence.endpoint_valid &&
                 (handle == sequence.state.read || handle == sequence.state.write)) ||
                (sequence.rewrite_state && handle == *sequence.rewrite_state)) {
                return true;
            }
            for (std::size_t prior = 0; prior < anchor_index; ++prior) {
                if (sequence.long_anchors[prior].state == handle) { return true; }
            }
            return false;
        };

        if (sequence.endpoint_valid) {
            release_if_unreferenced(sequence.state.write, true);
            if (sequence.state.read != sequence.state.write) {
                release_if_unreferenced(sequence.state.read,
                                        !sequence.state.read_has_external_owner());
            }
        }
        if (sequence.rewrite_state) {
            const StateImageHandle handle = *sequence.rewrite_state;
            const bool duplicates_endpoint =
                sequence.endpoint_valid &&
                (handle == sequence.state.read || handle == sequence.state.write);
            release_if_unreferenced(handle, !duplicates_endpoint);
        }
        for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
            const StateImageHandle handle = sequence.long_anchors[index].state;
            release_if_unreferenced(handle, !repeated_before_anchor(index, handle));
        }
        if (sequence.reserved_state) {
            const StateImageHandle handle = *sequence.reserved_state;
            bool repeated                 = sequence.endpoint_valid &&
                            (handle == sequence.state.read || handle == sequence.state.write);
            repeated = repeated || (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
                repeated = repeated || anchor.state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
    } catch (...) { fail(); }

    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
}

// 错误路径上的兜底放手：不查前提、不看返回值。每一步都先确认"看起来还在、才去动它"，抛出的异常
// 就地吞掉（已经有一次错误在往上传播了，不能让它被第二次异常盖掉）。能还的还，还不了的就留着——
// 后续的引擎级清理会再扫一遍。
void ProgramImpl::release_sequence_state(SequenceState& sequence) noexcept {
    if (!state_store) { return; }
    if (sequence.state.fork_pending && state_store->valid(sequence.state.read) &&
        state_store->valid(sequence.state.write)) {
        try {
            state_store->abort_fork(sequence.state.read, sequence.state.write);
        } catch (...) {}
    }

    try {
        if (sequence.rewrite_state && state_store->valid(*sequence.rewrite_state) &&
            state_store->checkpoint_references(*sequence.rewrite_state) != 0) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            if (state_store->valid(anchor.state) &&
                state_store->checkpoint_references(anchor.state) != 0) {
                state_store->release_checkpoint_reference(anchor.state);
            }
        }
    } catch (...) {}

    const auto releasable = [&](StateImageHandle handle) { return state_store->valid(handle); };
    if (releasable(sequence.state.write)) { (void)state_store->release(sequence.state.write); }
    if (!sequence.state.read_has_external_owner() && sequence.state.read != sequence.state.write &&
        releasable(sequence.state.read)) {
        (void)state_store->release(sequence.state.read);
    }
    if (sequence.rewrite_state) {
        const StateImageHandle handle = *sequence.rewrite_state;
        const bool duplicates_binding =
            handle == sequence.state.write ||
            (!sequence.state.read_has_external_owner() && handle == sequence.state.read);
        if (!duplicates_binding && releasable(handle)) { (void)state_store->release(handle); }
    }
    for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
        const StateImageHandle handle = sequence.long_anchors[index].state;
        bool duplicate =
            handle == sequence.state.write ||
            (!sequence.state.read_has_external_owner() && handle == sequence.state.read) ||
            (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (std::size_t previous = 0; !duplicate && previous < index; ++previous) {
            duplicate = sequence.long_anchors[previous].state == handle;
        }
        if (!duplicate && releasable(handle)) { (void)state_store->release(handle); }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool duplicate =
            handle == sequence.state.write ||
            (!sequence.state.read_has_external_owner() && handle == sequence.state.read) ||
            (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            duplicate = duplicate || anchor.state == handle;
        }
        if (!duplicate && releasable(handle)) { (void)state_store->release(handle); }
    }
    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
}

// release_active_shared_references_strict 的兜底版：计数为 0、槽位角色不对，都直接跳过——错误路径
// 上"看起来不对"不算异常，只说明账已经乱了，不必再为它 terminate。
void ProgramImpl::release_active_shared_references(SequenceState& sequence) noexcept {
    for (const std::uint32_t index : sequence.shared_prefix_references) {
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
            shared_prefix_states[index].active_references == 0) {
            continue;
        }
        --shared_prefix_states[index].active_references;
    }
    sequence.shared_prefix_references.clear();
}

// ================================================================================================
// KV 地址空间的映射与放手（program_impl.h 第 12) 段）
//
// 三件互相独立的事，别混：**额度**（这个 owner 最多能占多少页，只增不减，多出来的一截转成池内预留）、
// **映射**（页物理上有没有落在设备上，`ensure_mapped_to_tokens` 负责补齐）、**已提交前沿**（对外承诺
// 这些页的内容已经有效，只能靠 `commit_frontier` 前进，靠 `destructive_truncate` 倒退）。所以
// resize / ensure_mapped / commit / trim 才要分成四个入口。
//
// 放手按"地址是否还活着"分两档：还在活跃映射里的走 release_after_deactivate（先停用再释放），
// 已经不活跃的走 release。两套 KV 地址空间（主 KV 与后端 KV）永远一起动——要么都激活要么都不激活。
// ================================================================================================

// 当前后端对应的 KV 缓存：MTP 用解码器的 mtp 缓存，DFlash 用它的 full 缓存，普通解码没有。
qwen3_5::PagedKVCache* ProgramImpl::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (dflash && dflash->full) { return &*dflash->full; }
    return nullptr;
}

const qwen3_5::PagedKVCache* ProgramImpl::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (dflash && dflash->full) { return &*dflash->full; }
    return nullptr;
}

// 后端 KV 已经有效到哪个位置——注意三种后端的语义各不相同：MTP 是独立的 token 计数（mtp_kv_valid），
// DFlash 记的是 context 前沿（dflash_context_frontier），普通解码恒为 0（没有后端 KV）。
std::uint32_t ProgramImpl::backend_kv_valid(const SequenceState& sequence) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return sequence.mtp_kv_valid; }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        return sequence.dflash_context_frontier;
    }
    return 0;
}

// 抬高额度（地址空间的 resize_entitlement 只管增长，多出的那截会被记成池内预留，不是立刻分配）。
// 两套地址空间的"有没有后端"必须和请求一致：只给有后端的序列传 backend_pages 才合法。
void ProgramImpl::resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                                 std::uint32_t backend_pages) {
    if (!sequence.kv || text_pages == 0 ||
        (sequence.kv->backend.has_value() != (backend_pages != 0))) {
        throw std::invalid_argument("KV resize entitlement does not match the sequence bundle");
    }
    text_kv_addresses->resize_entitlement(sequence.kv->text, text_pages);
    if (sequence.kv->backend) {
        backend_kv_addresses->resize_entitlement(*sequence.kv->backend, backend_pages);
    }
}

// 建立执行期映射并写设备表：激活两套地址空间（额度取当前映射页数），然后把"这一切对应表格的哪一行"
// 写成设备上的标量（io.text_kv_table_row / io.backend_kv_table_row），执行期就靠这个行号找 KV。
// 两套地址空间必须**整体一致**——一个激活一个不激活直接违约；中途失败要把已经激活的那一半回滚掉。
void ProgramImpl::bind_sequence_kv(SequenceState& sequence) {
    if (!sequence.kv) { throw std::logic_error("KV allocation bundle is unavailable"); }
    const std::int32_t row = static_cast<std::int32_t>(sequence.lane);
    const bool text_active = text_kv_addresses->active(sequence.kv->text);
    const bool backend_active =
        sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend);
    if (sequence.kv->backend && text_active != backend_active) {
        throw std::logic_error("KV address-space activation is not bundle-atomic");
    }
    try {
        if (!text_active) {
            text_kv_addresses->activate(sequence.kv->text,
                                        text_kv_addresses->mapped_pages(sequence.kv->text), row);
            if (sequence.kv->backend) {
                backend_kv_addresses->activate(
                    *sequence.kv->backend,
                    backend_kv_addresses->mapped_pages(*sequence.kv->backend), row);
            }
        }
        set_device_i32(io.text_kv_table_row, text_kv_addresses->bound_row(sequence.kv->text));
        set_device_i32(io.backend_kv_table_row,
                       sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend)
                                            : 0);
    } catch (...) {
        if (!text_active) {
            if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
                backend_kv_addresses->deactivate(*sequence.kv->backend);
            }
            if (text_kv_addresses->active(sequence.kv->text)) {
                text_kv_addresses->deactivate(sequence.kv->text);
            }
        }
        throw;
    }
}

// 解除执行期映射（顺序与 bind 相反：先后端再主 KV），但不释放地址——映射和归属是两件事，解绑只是
// 把执行期的占用摘掉。解绑本身也不该失败，异常就地吞掉。
void ProgramImpl::unbind_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    try {
        if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
            backend_kv_addresses->deactivate(*sequence.kv->backend);
        }
    } catch (...) {}
    try {
        if (text_kv_addresses->active(sequence.kv->text)) {
            text_kv_addresses->deactivate(sequence.kv->text);
        }
    } catch (...) {}
}

// 把映射补齐到指定 token 数（不够的页从 Host 搬上来，走 device.stream）。这里的 token 数是**下界**：
// 已经映射得更多就不动。注意它抬高的是映射，不是已提交前沿——搬上来的页在 commit 之前不算数。
void ProgramImpl::ensure_sequence_kv_mapped(SequenceState& sequence, std::uint32_t main_tokens,
                                            std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    text_kv_addresses->ensure_mapped_to_tokens(sequence.kv->text, main_tokens, device.stream);
    if (backend_tokens != 0) {
        backend_kv_addresses->ensure_mapped_to_tokens(*sequence.kv->backend, backend_tokens,
                                                      device.stream);
    }
}

// 前进已提交前沿：从这一刻起这些页的内容被当成有效的，可以被别人共享、可以当检查点的底座。
// 后端 KV 跟着一起提交（没提交后端前沿的检查点是不可用的）。
void ProgramImpl::commit_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                     std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity ||
        (backend_tokens != 0 && !sequence.kv->backend)) {
        throw std::logic_error("KV commit request is outside the sequence bundle");
    }
    text_kv_addresses->commit_frontier(sequence.kv->text, main_tokens);
    if (sequence.kv->backend) {
        backend_kv_addresses->commit_frontier(*sequence.kv->backend, backend_tokens);
    }
}

// 破坏性回退已提交前沿（回溯 / 放弃草稿用）。"破坏性"是字面意思：被砍掉的尾页内容和它的旧副本都
// 作废，后缀重建过的镜像也不能再往这里接。
void ProgramImpl::trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                   std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > main_tokens) {
        throw std::logic_error("KV trim request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV trim requested without an allocation");
    }
    text_kv_addresses->destructive_truncate(sequence.kv->text, main_tokens);
    if (sequence.kv->backend) {
        backend_kv_addresses->destructive_truncate(*sequence.kv->backend, backend_tokens);
    }
}

// 交还"增长出来的"那部分额度（只还增量、不动基础额度）。用于一轮扩张结束、确定不再需要预留时，
// 把多占的池内预留还回去。尽力而为：还不动就算了，不影响正确性。
void ProgramImpl::release_sequence_growth_entitlement(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    try {
        text_kv_addresses->release_growth_entitlement(sequence.kv->text);
        if (sequence.kv->backend) {
            backend_kv_addresses->release_growth_entitlement(*sequence.kv->backend);
        }
    } catch (...) {}
}

// 还活跃着的地图怎么放手：先确认两套地址都能"停用后释放"，再先后端后主 KV 地走 release_after_deactivate
// （它会先停用映射，绕过"活跃地址不能直接释放"的限制）。strict——查不过或放不掉都 terminate。
// 最后清掉 bundle 并把没人引用的 Host extent 收回去。
void ProgramImpl::release_active_sequence_kv_strict(SequenceState& sequence) noexcept {
    if (!sequence.kv || !text_kv_addresses ||
        !text_kv_addresses->can_release_after_deactivate(sequence.kv->text) ||
        (sequence.kv->backend &&
         (!backend_kv_addresses ||
          !backend_kv_addresses->can_release_after_deactivate(*sequence.kv->backend)))) {
        std::terminate();
    }
    if (sequence.kv->backend &&
        !backend_kv_addresses->release_after_deactivate(*sequence.kv->backend)) {
        std::terminate();
    }
    if (!text_kv_addresses->release_after_deactivate(sequence.kv->text)) { std::terminate(); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

// 已经不活跃的地址怎么放手：直接用 release（它要求地址不是活跃状态，正是这里的场景——序列已经解绑
// 或从来没绑过）。同样 strict，且同样按后端 → 主 KV 的顺序。
void ProgramImpl::release_sequence_kv_strict(SequenceState& sequence) noexcept {
    if (!sequence.kv || !text_kv_addresses || !text_kv_addresses->can_release(sequence.kv->text)) {
        std::terminate();
    }
    if (sequence.kv->backend &&
        (!backend_kv_addresses || !backend_kv_addresses->can_release(*sequence.kv->backend))) {
        std::terminate();
    }
    if (sequence.kv->backend && !backend_kv_addresses->release(*sequence.kv->backend)) {
        std::terminate();
    }
    if (!text_kv_addresses->release(sequence.kv->text)) { std::terminate(); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

// 兜底放手：先解绑（把地址压回"不活跃"），再逐个 release，最后清 bundle 与 Host extent。
// 不检查任何前提、不看返回值——错误路径上只求尽量还回去。
void ProgramImpl::release_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    unbind_sequence_kv(sequence);
    if (sequence.kv->backend && backend_kv_addresses) {
        (void)backend_kv_addresses->release(*sequence.kv->backend);
    }
    if (text_kv_addresses) { (void)text_kv_addresses->release(sequence.kv->text); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

// 执行期视图：给它一个活跃映射才能拿到（视图里带的是本轮的设备地址，不是长期有效的句柄）。
qwen3_5::PagedKVCacheView ProgramImpl::text_kv_view(const SequenceState& sequence) const {
    if (!sequence.kv || !text_kv_addresses->active(sequence.kv->text)) {
        throw std::logic_error("sequence has no active KV execution mapping");
    }
    return decoder->text_kv.execution_view(text_kv_addresses->execution_row(sequence.kv->text));
}

// MTP 专用视图：非 MTP 后端给空视图（调用方按空判断，不是错误）；是 MTP 却拿不到活跃映射才是违约。
qwen3_5::PagedKVCacheView ProgramImpl::mtp_kv_view(const SequenceState& sequence) const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend ||
        !backend_kv_addresses->active(*sequence.kv->backend)) {
        throw std::logic_error("sequence has no active MTP KV execution mapping");
    }
    return decoder->mtp_cache()->execution_view(
        backend_kv_addresses->execution_row(*sequence.kv->backend));
}

// ================================================================================================
// 两个边角动作（program_impl.h 第 11) 段）
// ================================================================================================

// 把一个标量异步写到设备张量上。异步是刻意的：它走 compute stream，和随后要在同一流上排队的工作天然
// 有序；也因此它只能出现在图外——图里的内容必须是捕获时就定死的，不能靠这种运行时标量。
void ProgramImpl::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

// Root 复用路径上的一次性归零（prefill.cpp 在新序列开始吃 prompt 之前调用）：把执行期位置、三种后端
// 的 KV 有效位置和 work 全部清回起点，让这条 lane 看起来像刚起步。
//
// "ordered" 有两层意思。一是**前提**：必须已经有一份私有的可变写目标（read == write 且角色是
// ActiveMutable），不能带着未结算的 fork；二是**顺序**：先重指视图（refresh_state_views），再清
// work，最后才归零那些设备标量——因为前两步会改这条序列自己的绑定，标量必须在新绑定生效之后再写。
void ProgramImpl::ordered_reset(SequenceState& sequence) {
    if (!state_store->valid(sequence.state.write)) {
        throw std::logic_error("pre-reset StateImage reservation is missing");
    } else {
        if (sequence.state.fork_pending || sequence.state.read != sequence.state.write ||
            state_store->role(sequence.state.write) != StateImageRole::ActiveMutable) {
            throw std::logic_error("StateImage reset requires a private mutable destination");
        }
    }
    refresh_state_views(sequence);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
}


} // namespace ninfer::models::qwen3_5::detail
