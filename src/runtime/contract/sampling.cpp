#include "runtime/contract/sampling.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::runtime {
namespace {

// 校验的是**解析之后**的值，所以模型自己注册的预设同样会被拦住（预设写错应该尽早炸掉）。
void validate(const ResolvedSamplingParameters& sampling) {
    // 有限性必须单独先查，且在其它区间比较之前：NaN 与任何数比较都是 false，会从每个区间判断里漏进内核。
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    // temperature=0 表示退化成精确 argmax（贪心），不是"没有温度"；上界 2 是采样内核的稳定范围。
    if (sampling.temperature < 0.0F || sampling.temperature > 2.0F) {
        throw std::invalid_argument("temperature must be in [0,2]");
    }
    // 上界 20 不是策略而是**物理约束**（候选域固定，见 src/ops/common/sampling_workspace.h 的
    // kSamplerFastCandidates）；0 在 resolve 里已被改写成 20，所以下界是 1。
    if (sampling.top_k < 1 || sampling.top_k > 20) {
        throw std::invalid_argument("resolved top_k must be in [1,20]");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
    // 两个惩罚项都是对称区间 [-2,2]：可为负（鼓励重复），0 表示不施加惩罚。
    if (sampling.presence_penalty < -2.0F || sampling.presence_penalty > 2.0F) {
        throw std::invalid_argument("presence_penalty must be in [-2,2]");
    }
    if (sampling.frequency_penalty < -2.0F || sampling.frequency_penalty > 2.0F) {
        throw std::invalid_argument("frequency_penalty must be in [-2,2]");
    }
}

} // namespace

ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                            SamplingMode mode, const SamplingOverrides& overrides) {
    const SamplingPreset& preset = defaults.for_mode(mode);
    ResolvedSamplingParameters resolved{
        .temperature       = overrides.temperature.value_or(preset.temperature),
        .top_k             = overrides.top_k.value_or(preset.top_k),
        .top_p             = overrides.top_p.value_or(preset.top_p),
        .min_p             = overrides.min_p.value_or(preset.min_p),
        .presence_penalty  = overrides.presence_penalty.value_or(preset.presence_penalty),
        .frequency_penalty = overrides.frequency_penalty.value_or(preset.frequency_penalty),
        // seed 没有预设可回落（SamplingPreset 里没有 seed 字段），缺省即 0：**不指定种子 = 可复现**，
        // 不是每次随机。
        .seed              = overrides.seed.value_or(0),
    };
    // The registered sampling pipeline has an exact top-20 candidate domain. Preserve the
    // existing public meaning of zero (use the target cap), then expose only concrete values to
    // runtimes.
    // top_k == 0 的公共含义是"用目标上限"，这里具体化成 20；必须发生在 validate 之前（它要求 [1,20]）。
    if (resolved.top_k == 0) { resolved.top_k = 20; }
    validate(resolved);
    return resolved;
}

} // namespace ninfer::runtime
