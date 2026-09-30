#pragma once

#include "ninfer/types.h"

// ============================================================================
// runtime/contract/sampling.h —— 采样参数解析
// ============================================================================
//
// 请求侧 SamplingOverrides（全 optional）与执行侧 ResolvedSamplingParameters（全具体值）的
// 唯一收敛点，"默认值逻辑"在整个执行层只存在这一处；非法组合在这里抛 std::invalid_argument。
// 两个反直觉的解析结果：缺省 seed 解析成 0（**确定**，不是随机）；
// top_k == 0 在预设里表示"用目标上限"，会被改写成 20。

namespace ninfer::runtime {

// Resolves one request at the Engine boundary. The registered preset supplies every omitted
// model-owned field; an omitted seed remains deterministic for direct Engine callers.
[[nodiscard]] ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                                          SamplingMode mode,
                                                          const SamplingOverrides& overrides);

} // namespace ninfer::runtime
