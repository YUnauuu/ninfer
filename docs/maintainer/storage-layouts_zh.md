# NInfer 持久化存储布局

本文档记录当前 `.ninfer` 工件使用的持久化张量布局和资源编码，包括对齐、字节序、填充、编码大小规则以及逻辑解码。数值语义来自 [`tensor-formats.md`](tensor-formats.md)；封装和对象范围来自 [`artifact-container.md`](artifact-container.md)。

## 1. 已注册的标识

存储注册表恰好包含以下标识：

| 标识 | 类型 | 兼容的数值格式 | 逻辑形状 | 对象对齐 |
|---|---|---|---|---:|
| `contiguous_le_v1` | 张量布局 | `bf16`、`fp32`、`int32` | rank `0..16` | 256 字节 |
| `row_split_k128_v1` | 张量布局 | `q4_g64_fp16`、`q5_g64_fp16`、`q6_g64_fp16`、`q8_g32_fp16` | rank 2 `[N,K]` | 256 字节 |
| `block_scale_k16_m128x4_v1` | 张量布局 | `nvfp4` | rank 2 `[N,K]`，`N % 128 == 0`，`K % 64 == 0` | 256 字节 |
| `row_scale_v1` | 张量布局 | `fp8_e4m3fn_row_bf16` | rank 2 `[N,K]` | 256 字节 |
| `raw_bytes_v1` | 资源编码 | 不适用 | 非空字节字符串 | 1 字节 |

这些格式/布局组合定义当前的编解码器支持。原生消费者要求在第 8 节单独说明。

对象对齐适用于 `.ninfer` JSON 中对象相对于负载的 `offset`。内部平面偏移和填充属于所选布局。对象之间的填充属于容器，不计入对象的 `bytes`。

下文使用以下辅助函数：

```text
align_up(x, a) = ceil_div(x, a) * a
```

所有公式输入和中间结果都是非负整数。容器实现必须拒绝无法用其文件偏移和大小类型表示的形状或计算结果。

## 2. `contiguous_le_v1`

### 2.1 逻辑遍历

`contiguous_le_v1` 按 C 顺序直接存储逻辑字：最后一个逻辑维度变化最快。对于形状 `[D0, D1, ..., D(r-1)]`，坐标 `[i0, i1, ..., i(r-1)]` 的线性索引为：

```text
index = (((i0 * D1 + i1) * D2 + i2) ... ) * D(r-1) + i(r-1)
```

rank 为零的形状 `[]` 包含一个标量字。其他合法形状满足：

```text
elements = product(shape)
```

该布局不执行 reshape、转置、类型转换或数值规范化。需要时，这些操作会在特定模型转换配方中、布局编码之前完成。

### 2.2 字节序

字按最低有效字节在前的顺序序列化：

| 格式 | 每个元素的字节数 | 存储的字 |
|---|---:|---|
| `bf16` | 2 | 精确的 16 位 bfloat16 逻辑字，小端序 |
| `fp32` | 4 | 精确的 32 位 IEEE-754 binary32 逻辑字，小端序 |
| `int32` | 4 | 精确的 32 位二进制补码逻辑字，小端序 |

符号零、非规格化数、无穷、NaN 负载以及整数值行为由数值格式契约决定。布局只保留字的位。

### 2.3 编码大小

没有内部前缀、步长表、逐行填充或尾部填充：

```text
payload_bytes = elements * bytes_per_element(format)
```

张量对象 JSON 中的 `bytes` 必须严格等于 `payload_bytes`。

## 3. `row_split_k128_v1`

### 3.1 逻辑和物理几何

逻辑张量是正的 rank-two 矩阵 `[N,K]`。格式提供码宽 `b` 和组大小 `G`：

| 格式 | `b` | `G` | 每组基础字节数 `B` | 每组高位字节数 `H` |
|---|---:|---:|---:|---:|
| `q4_g64_fp16` | 4 | 64 | 32 | 0 |
| `q5_g64_fp16` | 5 | 64 | 32 | 8 |
| `q6_g64_fp16` | 6 | 64 | 32 | 16 |
| `q8_g32_fp16` | 8 | 32 | 32 | 0 |

最后一个轴扩展到 128 的倍数：

```text
K_pad              = align_up(K, 128)
groups_per_row     = K_pad / G
logical_groups     = ceil_div(K, G)
physical_group_cnt = N * groups_per_row
```

`K_pad` 是物理几何，不会加入 JSON 的 `shape`。由于两个已注册的组大小都能整除 128，`groups_per_row` 总是整数。

对于最后一个逻辑上不完整的组，列号大于等于 `K` 的 lane 使用有符号码零。其 scale 仍然是数值格式契约定义的逻辑组 scale。`logical_groups` 之后的每个完整物理组的 scale 字为 `0x0000`，所有码均为零。因此物理填充不会改变解码后的逻辑值。

### 3.2 负载平面

负载按以下顺序包含三个概念平面：

```text
基础码平面
填充零至 256 字节边界
可选的高位平面
填充零至 256 字节边界
binary16 scale 平面
```

Q4 和 Q8 没有高位字节，但仍将 scale 平面放在基础码平面之后的第一个 256 字节边界处。对象内部 scale 平面之后没有填充。

每个平面内的遍历顺序为：

```text
第 0 行第 0 组，第 0 行第 1 组，...，第 1 行第 0 组，...
```

属于同一组的字节彼此相邻。

### 3.3 基础码平面

对于 Q4、Q5 和 Q6，令 `q[i]` 为 lane `i` 的有符号码，并令：

```text
u[i] = q[i] modulo 2^b
```

表示其无符号 `b` 位二进制补码字。每对连续 lane 占用一个基础字节：

```text
base[j] = (u[2*j] & 0x0f) | ((u[2*j + 1] & 0x0f) << 4)
```

因此偶数 lane 位于低半字节，奇数 lane 位于高半字节。每个 G64 组占用 32 个基础字节。

对于 Q8，每个 lane 占用一个包含其精确 8 位二进制补码字的字节。lane `i` 占用字节 `i`，所以每个 G32 组占用 32 个基础字节。排除码 `-128` 的数值格式限制仍然有效。

完整的基础平面是这些按平面遍历顺序排列的逐组字节序列的拼接。

### 3.4 高位平面

只有 Q5 和 Q6 有高位平面。对于每个 lane：

```text
high[i] = (u[i] >> 4) & ((1 << (b - 4)) - 1)
```

高位流按 lane 主序排列。对一个 lane，先输出位 4，Q6 再输出位 5。流中的第 `t` 位存储在字节 `floor(t / 8)` 的第 `(t mod 8)` 位；因此位零是每个字节的第一位。

等价地：

- Q5 的字节 0 在字节位 0 至 7 中包含 lane 0 至 7 的高位；
- Q6 的字节 0 中，lane 0 的位 4 和位 5 位于字节位 0 和 1，lane 1 的位 4 和位 5 位于字节位 2 和 3，依此类推。

一个 Q5 G64 组占用 8 个高位字节。一个 Q6 G64 组占用 16 个高位字节。

### 3.5 Scale 平面

每个物理组拥有一个 16 位 scale 字。Scale 字按与码平面相同的行/组顺序遍历，并以小端序存储：

```text
scale_index(row, group) = row * groups_per_row + group
```

对于逻辑组，该字恰好是数值格式定义的 binary16 乘数。第 3.1 节的填充规则定义了完全物理组的 scale 字。

### 3.6 平面偏移和编码大小

令：

```text
base_bytes  = N * groups_per_row * B
high_bytes  = N * groups_per_row * H
scale_bytes = N * groups_per_row * 2

base_offset  = 0
high_offset  = align_up(base_bytes, 256)
scale_offset = high_offset + align_up(high_bytes, 256)

payload_bytes = scale_offset + scale_bytes
```

字节范围为：

```text
base  = [base_offset,  base_offset  + base_bytes)
high  = [high_offset,  high_offset  + high_bytes)
scale = [scale_offset, scale_offset + scale_bytes)
```

当 `H = 0` 时，`high_bytes = 0` 且 `scale_offset = high_offset`。基础平面末尾到 `high_offset` 之间，以及非空高位平面末尾到 `scale_offset` 之间的字节均为零。

张量对象 JSON 中的 `bytes` 必须等于 `payload_bytes`；它不包括为对齐下一个对象而需要的间隔。

例如，形状为 `[2,130]` 的 Q5 张量具有 `K_pad=256`、每行四组、`base_bytes=256`、`high_bytes=64`、`scale_bytes=16`、`high_offset=256`、`scale_offset=512` 和 `payload_bytes=528`。

### 3.7 行视图和行切片

行寻址是此布局的固有属性。定义：

```text
base_row_bytes  = groups_per_row * B
high_row_bytes  = groups_per_row * H
scale_row_bytes = groups_per_row * 2
```

连续行 `[row_begin, row_begin + row_count)` 的非拥有逻辑视图使用：

```text
base_view  = base_offset  + row_begin * base_row_bytes
high_view  = high_offset  + row_begin * high_row_bytes   # absent when H = 0
scale_view = scale_offset + row_begin * scale_row_bytes
```

它在相应平面中分别覆盖 `row_count * base_row_bytes`、`row_count * high_row_bytes` 和 `row_count * scale_row_bytes`。其逻辑形状为 `[row_count,K]`，并保留父对象的 `K_pad` 和 `groups_per_row`。因此，行视图是三个平面跨度，而不是一个假定连续的负载范围。

如果这些行被物化为独立负载，则三个行跨度按相同的平面顺序拼接，并使用第 3.6 节、以 `N=row_count` 重新计算平面偏移和零填充。这会生成另一个有效的 `row_split_k128_v1` 张量，无需解码或重新打包单个码。

## 4. `block_scale_k16_m128x4_v1`

此布局只存储满足以下条件的 rank-two `nvfp4` 矩阵 `[N,K]`：

```text
N > 0
K > 0
N % 128 == 0
K % 64 == 0
```

它不添加逻辑填充。令：

```text
code_plane_bytes      = N * K / 2
scale_plane_offset    = align_up(code_plane_bytes, 256)
scale_plane_bytes     = N * K / 16
weight_divisor_offset = scale_plane_offset + scale_plane_bytes
payload_bytes         = weight_divisor_offset + 4
```

负载依次为按行主序排列的 E2M1 打包码平面、填充零至 `scale_plane_offset`、经过 swizzle 的 E4M3FN scale 平面，以及小端序 FP32 weight-divisor 字。每个打包码字节中，低半字节是较小的 K 坐标，高半字节是下一个坐标。

对于逻辑行 `n`、scale 组坐标 `g=floor(k/16)` 和 `K_tiles=K/64`，定义：

```text
row_tile   = floor(n / 128)
row_inner  = n % 128
scale_tile = floor(g / 4)
scale_lane = g % 4
```

Scale 字在 scale 平面内的字节偏移为：

```text
(row_tile * K_tiles + scale_tile) * 512
+ (row_inner % 32) * 16
+ floor(row_inner / 32) * 4
+ scale_lane
```

布局解码必须恢复原始的打包 E2M1 字、自然的 `[N,K/16]` E4M3FN scale 字矩阵以及精确的除数值。它不会对任一浮点格式执行解码再重新编码。

## 5. `row_scale_v1`

`row_scale_v1` 只存储正维度的 rank-two `fp8_e4m3fn_row_bf16` 矩阵 `[N,K]`。它不添加逻辑或物理矩阵填充。令：

```text
code_plane_bytes   = N * K
scale_plane_offset = align_up(code_plane_bytes, 256)
scale_plane_bytes  = N * 2
payload_bytes      = scale_plane_offset + scale_plane_bytes
```

负载以每个逻辑权重一个 E4M3FN 字节的行主序码开始。坐标 `[n,k]` 存储在码平面偏移 `n * K + k` 处。`code_plane_bytes` 到 `scale_plane_offset` 的区间由零字节填充。

Scale 平面按递增的 `n` 顺序，为每个逻辑行包含一个小端序 BF16 字。第 `n` 个 scale 字从 `scale_plane_offset + 2 * n` 开始。该布局既不转换 BF16 乘数，也不将其与对应的 E4M3FN 行合并。码和 scale 的有效性以及表示的权重重建由 [`tensor-formats.md`](tensor-formats.md) 中的 `fp8_e4m3fn_row_bf16` 定义。

逻辑行视图由连续的 `K` 个码字节和其一个 BF16 scale 字组成；这两个跨度不是一个假定连续的负载范围。独立的连续切片或行收集通过拼接所选码行、为新的行数重新计算 scale 平面对齐，并按相同行顺序追加所选 scale 字来编码。它不会解码或重新量化任一平面。

## 6. `raw_bytes_v1`

`raw_bytes_v1` 是资源编码，而不是张量布局。其外层对象负载本身就是资源字节字符串：

```text
payload_bytes = resource_length
alignment     = 1
```

它没有嵌入长度、头部、终止符、文件名、字符编码、压缩或尾部填充。资源对象的 JSON `bytes` 是其精确的非零长度，读取器原样返回完整跨度。模型契约为资源分配名称并解释这些字节；通用编码不会根据名称推断含义。

## 7. 解码边界

布局解码只产生持久化的逻辑字：

- `contiguous_le_v1` 按逻辑坐标顺序产生直接的 BF16、FP32 或 I32 字；
- `row_split_k128_v1` 产生逻辑列 `0..K-1` 的分组有符号码和 binary16 scale，丢弃物理列 `K..K_pad-1`；
- `block_scale_k16_m128x4_v1` 产生打包的 E2M1 字、自然的 E4M3FN 组 scale 字以及矩阵级 FP32 weight divisor；
- `row_scale_v1` 产生自然行主序的 E4M3FN 码字和每个逻辑行一个 BF16 乘数；
- `raw_bytes_v1` 产生外层资源字节。

反量化值遵循 [`tensor-formats.md`](tensor-formats.md) 中的重建规则。本文档不选择量化编码器、输出 dtype、累加 dtype、内核、运行时设备布局或模型消费者。

## 8. 逻辑视图和原生操作数

绑定寻址父对象的 C 顺序逻辑元素范围。父对象保留其完整几何和后备分配，因此视图可以使用原始矩阵维度定位码和平面。物化时，每个所需父对象只上传一次，并绑定非拥有视图。

[`weight_view.cpp`](../../src/core/weight_view.cpp) 提供平面寻址以及到原生操作数的桥接。直接张量可以使用连续元素范围。分组整数矩阵可以使用 K 不变的连续完整行，并分别使用码、高位和 scale 指针。

当前原生 `Weight` 桥接要求 FP8 和 NVFP4 使用完整父对象。它们的消费者使用完整矩阵几何进行平面寻址；不能把行切片当作重新打包的较小矩阵负载传入。相邻的逻辑投影仍可共享一个父对象：当所选融合实现消费它们的完整并集时，会将该父对象作为一个原生权重接收。

离线编解码器可以生成具有自身平面偏移的独立切片。加载器不执行这一转换。接受其他视图形式的执行实现必须正确使用原始父对象几何。
