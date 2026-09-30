# NInfer CLI

`build/apps/ninfer` 针对一个 v3 `.ninfer` 产物运行单个请求。在按照本指南操作之前，请先构建 NInfer 并使用[项目 README](../README.md) 下载一个产物。

示例使用 Qwen3.8-27B NVFP4 与 FP8 KV 存储。

## 文本输入

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

`--prompt` 和 `--messages` 中必须且只能提供一个。CLI 通常会省略 `--kv-capacity`，因此共享的 Main Text KV 池会跟随示例中的 32,768 token `--max-context`。

回答内容会流式输出到 stdout。人类可读的启动里程碑和运行时错误会写入 stderr，且不带服务时间戳。推理内容和 CLI 结果报告（计时、吞吐量、GPU 内存、按需输出的 token ID 以及投机解码统计）同样作为无前缀的产品输出使用 stderr，因此 stdout 可以被独立重定向。在终端上，权重物化是一条临时进度行，随后是紧凑的 Engine-ready 摘要。重定向后的 stderr 包含长时间加载的持久可读进度，且没有回车符或 ANSI 转义序列。`--log-level debug` 会暴露每个启动阶段。选项以及本地 prompt/message 输入失败仍保持为直接的命令诊断信息：

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Return one sentence." \
  --max-context 4096 \
  --max-new 64 \
  --kv-dtype fp8 \
  > answer.txt 2> run.log
```

`--chat-template FILE` 会使用本地 Jinja 文件覆盖产物内置的模板。对该文件的更改会在重启 NInfer 后生效：

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --chat-template tools/chat_templates/qwen3_8.jinja --prompt "Hello"
```

省略的 thinking 和 effort 选项会使用所选模板的默认值。`--no-thinking` 或 `--reasoning-effort none` 请求禁用 thinking；其他 effort 值不能与 `--no-thinking` 组合使用。模板会解释所选 effort。`--greedy` 独立选择精确的 argmax 解码。

`--thinking-budget N` 会在新一轮 Qwen thinking 块保持打开期间，为接受的模型来源 token 设置一个正的上限。如果模型在该精确边界处尚未输出 `</think>`，Engine 会将 [Qwen 的规范提前关闭指引](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md) 和 `</think>` 追加到同一常驻序列中，无需采样，通过推理流发布该指引，然后从更新后的上下文恢复普通生成。自然 thinking 关闭、停止条件、取消或边界处的总输出/上下文限制会优先处理，并抑制此插入。该选项不能与 `--no-thinking` 组合使用，但可以与 `--reasoning-effort` 组合使用。

`--max-new` 计入每一个已提交的生成 token，包括内部插入的控制 token。当有效输出容量延伸到 thinking 预算之外时，它必须为完整的 tokenizer 派生控制后缀加上一个关闭后的模型 token 留出空间；容量不足的请求会被拒绝，而不是截断后缀。正常输出会将插入的指引作为推理内容发送到 stderr。`--print-token-ids` 会包含插入的 ID，而 `--raw-output` 会保留原始控制表示。

例如，以下命令允许最多 512 个模型来源 thinking token，同时保留足够的总输出容量用于插入的后缀和答案：

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 \
  --max-new 1024 \
  --thinking-budget 512 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```


## 启动内存配置

GPU 驻留内存在 Engine 启动时冻结：

- 不使用 `--spec` 会省略 MTP/DFlash/DFlash2 权重和状态以及优化后的 proposal head；
- `--spec mtp`、`--spec dflash`（35B-A3B）和 `--spec dflash2`（Qwen3.8-27B）只加载所选的投机后端；
- 使用完整 proposal head 的投机后端会省略优化后的 proposal head；
- Vision 默认禁用，省略其权重以及 Vision 特定的统一工作区范围；
- `--vision` 会加载权重，为 Vision 编码/交接扩展一个 Program 工作区，并启用图像/视频输入。
- 单请求 CLI 使用 root-only 上下文模式，因此不会预留额外的 Device checkpoint StateImage，也不会捕获后续请求无法消费的 continuation。

完整的 `.ninfer` 清单仍会被验证。这些选择不是惰性加载：未启用 Vision 时启动的 Engine 会拒绝媒体输入，且之后无法启用 Vision。DFlash/DFlash2 和 Vision 可以同时启用；这些后端适用于多模态预填充之后的生成文本解码，并不会加速 Vision 编码。默认的投机和 Vision 设置会产生最小的驻留内存配置。

## 结构化消息

`--messages` 接受非空 JSON 消息数组，或包含 `messages` 和可选 `tools` 数组的对象。

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```


当消息文件包含仓库相对媒体路径时，请从仓库根目录运行：

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype fp8 \
  --vision \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```


支持的角色包括 `system`、`developer`、`user`、`assistant` 和 `tool`。所选模板会格式化这些角色。维护中的 Qwen 模板会将 system/developer 消息保留在其输入位置。

消息内容可以是字符串，也可以是包含以下内容的有序数组：

| 内容类型          | 源字段                 | 接受的源                                 |
| :---------------- | :--------------------- | :--------------------------------------- |
| text              | `text`                 | 字符串                                   |
| image / image_url | `image` 或 `image_url` | 本地路径、HTTP(S) URL 或 base64 data URI |
| video / video_url | `video` 或 `video_url` | 本地路径、HTTP(S) URL 或 base64 data URI |

`image_url` 和 `video_url` 可以是字符串，也可以是包含字符串 `url` 的对象。Assistant 历史可以包含 `reasoning_content` 和 `tool_calls`；工具结果使用角色 `tool` 和 `tool_call_id`。

请参阅 [`examples/cli/`](../examples/cli/) 获取已提交的文本、图像、视频、混合媒体、thinking、长解码和长上下文输入。

## 投机解码

投机解码默认禁用。选择 MTP 时可使用一到五个 draft 位置，或选择 35B-A3B DFlash 或 Qwen3.8-27B DFlash2 后端时使用一到十五个。两种 masked-draft 后端都可以与 `--vision` 组合使用。`--lm-head-draft` 选择优化后的 proposal head，并且需要选择一个后端：

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```


对于 DFlash：

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --kv-dtype fp8 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```


对于包含 DFlash2 伴随权重的 Qwen3.8-27B 产物，选择 `--spec dflash2 --draft-tokens 7`，可选地加上 `--lm-head-draft` 和 `--vision`。DFlash2 接受从 1 到 15 的每个 draft 数量；七是检查点推荐值。`groupwise-int` 和 `nvfp4` 产物使用相同的 Engine 路径，包括 CUDA Graph、并发请求、采样惩罚和前缀复用。没有伴随权重的产物在选择时会报告缺少 DFlash2 组件。Vision、MTP 和 DFlash 遵循相同规则：仅当该组件在启动时启用时才需要其权重。

每个 Engine 只能启用一个投机后端。已发布的[性能结果](performance.md/)使用三个 draft token 的 MTP 和七个 draft token（块长度八）的 DFlash，两者都使用优化后的 proposal head。DFlash 接受一到十五个 draft token；七个构成测得的块长度八，而十五使用支持的最大块长度十六。

## 常用选项

该表列出了可执行文件的默认值。上述示例选择 FP8 KV 和 MTP3。

| 选项                                                        | 含义                                                         | 默认值                       |
| :---------------------------------------------------------- | :----------------------------------------------------------- | :--------------------------- |
| `--max-context N`                                           | 每序列逻辑上下文上限                                         | `2048`                       |
| `--kv-capacity N|auto`                                      | 显式共享 Main Text KV 容量，或根据剩余 GPU 内存最大化该容量；省略表示 `--max-context` | `2048`                       |
| `--prefill-chunk N`                                         | 正文本预填充块，以 128 的倍数表示                            | `1024`                       |
| `--max-new N`                                               | 请求的输出 token 限制                                        | `128`                        |
| `--device N`                                                | CUDA 设备索引                                                | `0`                          |
| `--kv-dtype bf16|int8|fp8|nvfp4|k8v4`                       | KV 缓存存储                                                  | `bf16`                       |
| `--spec mtp|dflash|dflash2`                                 | 投机后端                                                     | 关闭                         |
| `--draft-tokens N`                                          | MTP `1..5`；DFlash/DFlash2 `1..15`                           | 未设置                       |
| `--lm-head-draft`                                           | 优化后的 proposal head                                       | 关闭                         |
| `--vision`                                                  | 启用图像/视频输入并加载 Vision GPU 分配                      | 关闭                         |
| `--no-cuda-graph`                                           | 禁用 CUDA Graph 解码                                         | 图开启                       |
| `--chat-template FILE`                                      | 使用本地 Jinja 模板                                          | 产物模板                     |
| `--no-thinking`                                             | 禁用 thinking                                                | 模板默认值                   |
| `--thinking-budget N`                                       | 正模型来源 thinking token 上限；省略表示无限制               | 未设置                       |
| `--reasoning-effort none|minimal|low|medium|high|xhigh|max` | 将 effort 值传递给所选模板                                   | 模板默认值                   |
| `--greedy`                                                  | 精确 argmax 解码                                             | 关闭                         |
| `--temperature F`                                           | 采样温度覆盖                                                 | 已注册模型/模式默认值        |
| `--top-p F`                                                 | nucleus 阈值覆盖                                             | 已注册模型/模式默认值        |
| `--top-k N`                                                 | top-k 阈值覆盖（`0..20`；零选择 top-20 上限）                | 已注册模型/模式默认值        |
| `--min-p F`                                                 | min-p 阈值覆盖                                               | 已注册模型/模式默认值        |
| `--presence-penalty F`                                      | presence penalty 覆盖                                        | 已注册模型/模式默认值        |
| `--frequency-penalty F`                                     | frequency penalty 覆盖                                       | 已注册模型/模式默认值（`0`） |
| `--seed N`                                                  | 采样种子                                                     | `0`                          |

当省略采样标志时，Engine 会为加载的架构和渲染后的 prompt 模式选择通用任务预设。当前官方模型使用：

| 模型            | Prompt 模式  | Temperature | Top-p  | Top-k | Min-p | Presence penalty |
| :-------------- | :----------- | :---------- | :----- | :---- | :---- | :--------------- |
| Qwen3.6-27B     | thinking     | `1.0`       | `0.95` | `20`  | `0`   | `0`              |
| Qwen3.6-27B     | non-thinking | `0.7`       | `0.80` | `20`  | `0`   | `1.5`            |
| Qwen3.8-27B     | thinking     | `1.0`       | `0.95` | `20`  | `0`   | `0`              |
| Qwen3.8-27B     | non-thinking | `0.7`       | `0.80` | `20`  | `0`   | `1.5`            |
| Qwen3.6-35B-A3B | thinking     | `1.0`       | `0.95` | `20`  | `0`   | `1.5`            |
| Qwen3.6-35B-A3B | non-thinking | `0.7`       | `0.80` | `20`  | `0`   | `1.5`            |

每个已注册预设中的 frequency penalty 都是 `0`。特定任务配置文件（例如 Qwen 的精确编码配置文件）使用显式采样覆盖。

重复 `--stop-token-id`、`--stop` 或 `--reasoning-stop` 可添加停止条件。使用 `--raw-output` 暴露前端的原始输出流，使用 `--print-token-ids` 在诊断信息中包含生成的 token ID。

运行 `./build/apps/ninfer --help` 查看确切的选项约定。

## 上下文与内存

官方产物具有 262,144 token 的原生上下文限制。在单张 RTX 5090 上的实际分配取决于所选产物、媒体工作负载、输出预算和 KV 缓存类型。产物描述其模型配置和权重表示；`--kv-dtype` 独立选择运行时 KV 存储。准备好的 prompt 必须适配 `--max-context`；必要时，生成会在剩余上下文容量处停止。`--kv-capacity N` 独立控制共享物理 Main Text KV 池，并向上舍入到 64 token 的页大小。`--kv-capacity auto` 加载所选权重，测量剩余 GPU 内存，并为完整启用的运行时布局直接选择最大的合法页容量。这包括所选的投机后端、固定序列状态、统一工作区和 CUDA Graph 允许量，同时保留默认 1 GiB 自动余量不予分配。它不会探测分配或在请求时调整池大小。单请求 CLI 通常会省略该选项，因此它会跟随 `--max-context`；这一区别主要对并发 Engine 或服务器有意义。

在 Engine 启动时，NInfer 会预留模型权重、持久序列状态、一个阶段复用的 Program 工作区，以及单独的 CUDA Graph 驱动允许量。启用 Vision 时，该单一工作区包含一个通用执行前缀和一个固定项目输出交接区域。Vision 编码可以在产生输出之前复用整个后备空间；当交接区域活跃时，Text/MTP/decode 工作仍位于通用前缀内。因此，容量是最大合法同时占用范围，而不是 Text、Vision 暂存和 Vision 输出分配的总和。文本预填充使用 `min(--prefill-chunk,--max-context)`；Vision 保持现有的 32,768 token 聚合 prompt 预算，但为已注册的 16,384 token 最大单项规划 Device 执行。请求不会执行项目拥有的设备分配或增长。上下文缓存容量控制有意不出现在这个单请求接口中；持久 Engine 和服务器路径负责跨请求复用和可选的 Host 后备。

当 Engine 被销毁时，所有权重、序列、工作区和图分配都会被释放。