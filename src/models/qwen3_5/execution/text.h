#pragma once
#include "models/qwen3_5/program/internal.h"


#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/sparse_moe.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/round_buffers.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

// ============================================================================
// models/qwen3_5/execution/text.h —— Qwen3.5 文本塔的"执行卡片"
// ============================================================================
//
// TextContext 由 Program 层每 Program 一张地构造并持有（见 execution/graphs.cpp、draft.cpp），
// 一次请求里的多个 chunk / 多轮 decode 复用同一个实例。它不拥有持久状态：KV cache、state_、
// io_ 都在别处，这里只是一叠引用 + 一组本轮参数。入口分两类：单序列（prefill_chunk /
// mtp_forward_*，标量从 io_ 取）与批处理（*_batch，整批张量显式传参后装进 active_*）；内部层
// （attn_mix / gdn_mix）不接这些参数，直接读 active_*，为空时退回读 io_。Tap 是特征捕获钩子：
// NullTap 时捕获代码在编译期整段消失。

namespace ninfer::models::qwen3_5::execution {

// Verify 不等于"投机解码专用"：普通 decode 也走 Verify 相（宽度为 1），决定拓扑的是宽度/批大小。
using Phase = qwen3_5::TextPhase;

// GDN 状态这一轮怎么落盘。RecordForReplay 只写每步的原始转移记录、不写最终状态：验证是按 K+1 列
// 一起前向的，被拒绝的草稿列不能污染状态，要等 ReplaySSM 事后按实际接受的列数逐步重建。
// 见 docs/maintainer/replayssm-gdn.md。
enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct NullTap {
    static constexpr bool enabled = false;
};

// processed_tokens 是本次**实际**吃掉的 token 数，可能小于 nominal_length（被 prefill_split_frontier_
// 或视觉 chunk 边界提前收尾），调用方必须按它推进游标；只有 finalized 的那一块才会采样 bonus token。
struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
    runtime::ExecutionTiming timing;
};

// DFlash 特征捕获槽。captured_mask 是 layers 的位掩码（故上限 32 层）；consume_prefill 是必须的
// 回调——张量交回调用方后 workspace 立刻可复用，不能等函数返回再读写。
struct DFlashFeatureSink {
    static constexpr bool enabled = true;
    using PrefillConsumer         = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const std::uint32_t> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    // 抓到的特征必须在回调里用掉或拷走；rewrite_checkpoint 见 rewrite_checkpoint_hidden_output_。
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;

class TextContext {
public:
    // text_kv_base —— 文本 token 在绝对位置空间里的起点：多模态提示里视觉 token 先占了 [0, text_kv_base)，
    // 文本 KV 从 text_kv_base 开始写、RoPE 也从这里偏移。纯文本传 0。
    TextContext(DeviceContext& ctx, const execution::Parameters& weights, WorkspaceArena& work,
                qwen3_5::PagedKVCacheView kv, LinearAttentionStatePool& state,
                qwen3_5::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base,
                qwen3_5::PagedKVCacheView mtp_kv           = qwen3_5::PagedKVCacheView(),
                const qwen3_5::PagedKVCache* batch_text_kv = nullptr,
                const qwen3_5::PagedKVCache* batch_mtp_kv  = nullptr);
    ~TextContext();

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    void set_proposal_head(const LinearParameters* weight, const std::int32_t* ids,
                           int count) noexcept {
        proposal_head_     = weight;
        proposal_head_ids_ = ids;
        proposal_head_n_   = count;
    }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }

    // 单次性的分界点（**绝对位置**）：跑到这里就提前收尾，供上层当作可恢复的 checkpoint 边界；-1 = 不分界，用完自动复位。
    void set_prefill_split_frontier(std::int64_t position) noexcept {
        prefill_split_frontier_ = position;
    }

    void set_rewrite_checkpoint_hidden_output(Tensor* output) noexcept {
        rewrite_checkpoint_hidden_output_ = output;
    }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent_ = extent; }

    // source_slot 是读、destination_slot 是写，**是两个不同的槽位**——就地更新时相同，
    // 投机验证要保留原状态回滚时会不同。越界抛 std::invalid_argument。
    void set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot);
    // 见 GdnStateAction。RecordForReplay 与 replay_records 必须成对出现，否则抛 std::invalid_argument。
    // 这个设置**不像分界点那样自动复位**：本轮结束后由调用方负责改回来。
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);

    [[nodiscard]] const LinearParameters* proposal_head() const noexcept { return proposal_head_; }

    [[nodiscard]] const std::int32_t* proposal_head_ids() const noexcept {
        return proposal_head_ids_;
    }

    [[nodiscard]] int proposal_head_n() const noexcept { return proposal_head_n_; }

    // full_ids 是**完整**的 token 序列（不是这一段）。finalize_at_end 只在请求的最后一块传 true——
    // 此时才采样 bonus token 并写下 logits / pos / rope_pos；含媒体那些重载的位置来自视觉侧。
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const qwen3_5::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   VisionPrefillSession& vision,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);

    // 形状约定：ids 是 [width, batch]，hidden / logits 是 [hidden_size, width, batch]（词表维在前），
    // 每批上限 kMaximumConcurrency（= 8）；KV 表行号与状态槽位都按 batch 给出。

    // **不采样**——token 由调用方自己从 logits 取（传 Phase::Verify 不是笔误：它与 K+1 列的验证同拓扑）。
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_source_slots,
                               const Tensor& linear_state_destination_slots,
                               ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                               Tensor& logits);

    // 投机解码的 target 验证：一次前向 K+1 列（草稿 + 1），上限 kDFlashDecodeMaximumWidth。接受/拒绝
    // 不在这里判——它只按贪心选出 target_tokens，真正的取舍由上层 Program 配合采样配置做。
    // **GDN 必须 RecordForReplay**，否则被拒绝的草稿列会污染状态。
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink);

    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  Tensor& mtp_hidden);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);
    // logits_column >= 0 时直接从该列取 logits 与草稿（省一次全量前向）；input_embeddings 给出时跳过查表。
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::CausalAttentionExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);
private:
    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv_.valid() || batch_mtp_kv_ != nullptr;
    }

    // attn_mix / gdn_mix 是两种 mixer，后者的符号在 GDN 里记反了——它"拼"进残差流而不是替换它；
    // mlp_tail 是后面的 FFN 段。位置 / KV 表行 / 包络一律从 active_* 读，index 是**紧凑层号**。
    void attn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase);
    void gdn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase);
    void mlp_tail(const BlockParameters& weights, Tensor& x, Phase phase,
                  const ops::SparseMoeHints& hints);
    [[nodiscard]] ops::SparseMoeHints next_projection_hints(int layer) const;
    // 同一层的 mixer 与 mlp_tail 各在自己的 work_.scope() 里，段结束即回收，其间张量互不可见。
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);

    template <class Tap>
    void target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                  const Tensor& rope_positions, const Tensor& valid_columns,
                                  const Tensor& kv_table_rows,
                                  const Tensor& linear_state_source_slots,
                                  ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                                  Tensor& logits, Tensor& target_tokens, Tap& tap);

    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                          const Tensor* input_embeddings);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::CausalAttentionExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens);

    // 两种输入形态恰好用一个；视觉那份多出 positions、vision 会话和 rope_delta（RoPE 位置的偏移量）。
    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin = 0;
    };

    // 四个 prefill_chunk 重载最终都汇到这里，**一次调用只处理一块**。
    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                 const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);

    DeviceContext& ctx_;
    const Parameters& parameters_;
    const TextConfig& config_;
    WorkspaceArena& work_;
    qwen3_5::PagedKVCacheView kv_;                         // 单序列文本 KV 视图
    qwen3_5::PagedKVCacheView mtp_kv_;                     // 单序列 MTP KV 视图
    const qwen3_5::PagedKVCache* batch_text_kv_ = nullptr; // 批处理文本 KV（指针 + 每行表行号）
    const qwen3_5::PagedKVCache* batch_mtp_kv_  = nullptr; // 批处理 MTP KV

    LinearAttentionStatePool& state_;  // GDN 状态池（槽位由调用方指定）
    qwen3_5::RoundState& io_;          // 当轮存储包：位置、KV 表行、hidden/logits/token 等 I/O
    Tensor& prefill_hidden_;           // 跨 chunk 暂存的残差流缓冲

    std::uint32_t prefill_chunk_;  // 单 chunk 最多几个 token
    std::uint32_t text_kv_base_;   // 文本 KV / RoPE 的绝对位置起点（纯文本为 0）

    // 只在一次批处理调用期间有效，由 Scoped* RAII 装上、出作用域还原；全为 null 时内部层退回读 io_。
    const Tensor* active_cache_positions_                                          = nullptr;
    const Tensor* active_rope_positions_                                           = nullptr;
    const Tensor* active_kv_table_rows_                                            = nullptr;
    const Tensor* active_linear_state_source_slots_                                = nullptr;
    const Tensor* active_linear_state_destination_slots_                           = nullptr;
    const Tensor* active_valid_columns_                                            = nullptr;
    const Tensor* active_backend_kv_table_rows_                                    = nullptr;
    const ops::CausalAttentionExecutionEnvelope* active_causal_attention_envelope_ = nullptr;
    std::int32_t active_sequence_batch_                                            = 0;
    std::int32_t active_sequence_width_                                            = 0;
    // 视觉占位导致 RoPE 位置相对文本位置的整体偏移（纯文本为 0）。
    std::int32_t rope_delta_                                                       = 0;
    std::int32_t linear_state_source_slot_                                         = 0;
    std::int32_t linear_state_destination_slot_                                    = 0;
    // 默认就地更新；投机路径会改成 RecordForReplay（见 GdnStateAction）。
    GdnStateAction gdn_state_action_          = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* replay_records_   = nullptr;

    std::int64_t prefill_split_frontier_      = -1;      // 提前收尾的绝对位置，-1 = 不分界
    Tensor* rewrite_checkpoint_hidden_output_ = nullptr; // 分界点的 hidden 额外写到哪
    std::uint32_t mtp_proposal_extent_        = 0;       // 最后一块要吐几个草稿 token

    const Weight* embed_                        = nullptr;
    const Tensor* final_norm_                   = nullptr;
    const LinearParameters* lm_head_            = nullptr;
    const LinearParameters* proposal_head_      = nullptr; // MTP 草稿头
    const std::int32_t* proposal_head_ids_      = nullptr; // 草稿槽位 → token id 的映射表（可为空）
    int proposal_head_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr; // 收尾采样配置，nullptr = 贪心
    const MtpParameters* mtp_                   = nullptr; // MTP 塔参数（构造时校验存在性）
};

} // namespace ninfer::models::qwen3_5::execution
