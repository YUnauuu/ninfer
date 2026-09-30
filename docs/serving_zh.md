# HTTP 服务

`build/apps/ninfer-serve` 加载一个 v3 `.ninfer` 产物，并在一个常驻 NInfer Engine 之上暴露 OpenAI 和 Anthropic 兼容的 HTTP 端点。

## 启动服务器

```bash
./build/apps/ninfer-serve /data0/cyh/ninfer_models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 \
  --port 8081 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking \
  --device 3
```



该命令使用 Qwen3.8-27B NVFP4。每个请求有 240,000 token 的逻辑上限。一个共享的 240,000 token 主文本 KV 池为已准入的请求提供服务；单独运行时任一请求都可以使用全部容量，当两个请求的完整预留都能容纳时，两个请求并发运行。

在 `C=2` 加两个额外 Device checkpoint 槽位的情况下，进程拥有两个活动 StateImage 保证，外加一个包含两个 Device 常驻 checkpoint 的全局池。八个 pinned Host State 槽位和 8 GiB pinned Host KV 在 Device 压力下保留非活动延续。活动请求容量为二。

其他产物使用相同的命令形式，只是路径不同。对于 35B-A3B DFlash，将 MTP 选择替换为 `--spec dflash --draft-tokens 7 --lm-head-draft`。带有 DFlash2 伴随权重的 Qwen3.8-27B 产物也支持 `--spec dflash2 --draft-tokens 7`，其中 `--lm-head-draft` 可选。DFlash2 接受 1..15 的 draft 数量，并支持相同的采样、并发、前缀复用以及图像/视频请求接口。它也可以与 `--vision` 组合使用。

当省略 `--model-id` 时，服务器公布并接受产物的 `metadata.name`，如果没有存储名称，则回退到其架构名称。显式的 `--model-id` 是公共 HTTP 别名覆盖，不会选择或改变模型执行。

Vision 默认禁用：其权重和 Vision 特定的统一工作区范围不会被分配，媒体请求和 token 计数请求会以 HTTP 400 `vision_disabled` 失败。当服务器必须接受图像或视频输入时，添加 `--vision`。推测性驻留同样由 `--spec mtp|dflash|dflash2` 和 `--draft-tokens` 冻结；省略 `--spec` 则不加载任何推测后端。`--lm-head-draft` 额外加载优化后的提议头。35B-A3B 上的 DFlash 和 Qwen3.8-27B 上的 DFlash2 可以与 `--vision` 组合使用；它们各自在多模态预填充之后加速生成的文本解码，而 Vision 编码和预填充仍在推测加速之外。后续请求无法启用启动时省略的能力。产物只需包含文本骨干以及为此进程选择的可选组件。

## 端点

| 方法和路径                           | 行为                                                 |
| :----------------------------------- | :--------------------------------------------------- |
| `GET /health`                        | Engine 就绪状态                                      |
| `GET /v1/models`                     | 配置的 OpenAI 模型别名和生效的 `max_model_len`       |
| `GET /v1/models/{id}`                | 查找配置的别名和生效的 `max_model_len`               |
| `POST /v1/chat/completions`          | OpenAI 风格聊天生成                                  |
| `POST /v1/responses`                 | OpenAI Responses Core 生成、状态、类型化 Item 和 SSE |
| `POST /v1/responses/input_tokens`    | Responses 提示 token 计数，不进行生成                |
| `GET /v1/responses/{id}`             | 检索本地存储的终端 Response                          |
| `DELETE /v1/responses/{id}`          | 删除本地存储的 Response                              |
| `GET /v1/responses/{id}/input_items` | 列出该 Response 的规范化输入 Item                    |
| `POST /v1/messages`                  | Anthropic 风格消息生成                               |
| `POST /v1/messages/count_tokens`     | checkpoint 原生的扩展输入 token 计数                 |

当 Engine 可以接受工作时，`GET /health` 返回 HTTP 200 和 `{"status":"ok"}`。在 Engine 范围故障后，它返回 HTTP 503 和 `{"status":"unavailable"}`。临时队列饱和不会使 Engine 不可用。该端点保持不认证。

每个 OpenAI 兼容响应都携带唯一的 `x-request-id` 头，包括流式和错误响应。Anthropic 端点使用其单独的 `request-id` 契约。

所有三个生成 SSE 端点在五秒没有协议事件后都会发出标准的 `: keep-alive` 注释。该注释仅用于传输：SSE 客户端会忽略它，它不会改变生成的文本、事件顺序、用量、存储的 Responses 或请求日志。在 Linux 上，已接受的连接还使用 TCP keepalive 和 15 秒的 `TCP_USER_TIMEOUT`；与心跳一起，死掉或不确认的对端通常会在约 20 秒内被取消，包括在请求等待或预填充期间。TCP 栈保持连接并确认数据的对端无法与正在读取的应用程序区分；当下游客户端消失时，代理必须关闭其上游 NInfer 连接。

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```



该端点支持：

- `system`、`developer`、`user`、`assistant` 和 `tool` 历史，以及旧式 `function` 历史；
- 字符串内容和有序的 text/refusal 部分；相邻部分被保留且不插入分隔符，空 wire 内容保持为空轮次；
- 用户 `image_url` 部分、兼容客户端使用的工具结果 `image_url` 部分，以及使用 HTTP(S) 或 data URI 的用户 `video_url` 扩展；图像 detail 省略或为 `auto`；
- 非负的 `max_completion_tokens` 和旧式 `max_tokens` 拼写；零表示仅处理提示而不生成；
- `temperature`、`top_p`、存在/频率惩罚，以及有符号整数 `seed`；
- 兼容的 `top_k`（`0..20`）和 `min_p`（`0..1`）采样器扩展；
- 最多四个非空停止字符串，应用于推理和答案输出；
- `n:1`、纯文本 `modalities` 和 `response_format: {"type":"text"}`；
- 非流式响应和服务器发送事件流；
- `stream_options.include_usage`；
- llama.cpp 兼容的终端 `timings`，以及可选的 `timings_per_token` 和流式 `return_progress` 观察；
- 非严格函数工具，`tool_choice` 为 `auto`、`none` 或 `auto` 模式下的 `allowed_tools`，启用并行调用，助手工具调用历史、工具结果消息和旧式函数调用历史；
- 顶层 `reasoning_effort` 字段；
- `enable_thinking` 和 `preserve_thinking`，可在顶层或 `chat_template_kwargs` 中；
- 助手 `reasoning_content` 和 `reasoning` 历史别名。

Engine 无法提供其可观察行为的选项在请求该行为时会被拒绝。这包括 JSON 约束输出、非零 `logit_bias`、请求的 log 概率、音频/文件输入或音频输出、`strict:true`、必需或命名工具选择、启用工具时的 `parallel_tool_calls:false`、显式低/高图像 detail、网络搜索、审核、低/高 verbosity、存储的 Chat Completions，以及非空旧式 `functions`。每个能力拒绝都会标识受影响的字段以及 NInfer 无法提供的保证。已知的约束解码别名（`grammar`、`structured_outputs`、`guided_json`、`guided_regex`、`guided_choice` 和 `guided_grammar`）会收到相同的显式拒绝，而不是被当作未知提示处理。

语义中性的字段不会使原本可执行的请求失败。全零 `logit_bias`、`logprobs:false`、`top_logprobs:0`、`verbosity:"medium"`、空旧式工具控制、纯文本 `audio` 配置和 `prediction` 都被接受，且不改变 Engine 执行。元数据、用户/安全标识符、服务层级和提示缓存提示同样是建议性的。未知顶层字段会被忽略。

`tool` 消息上的字符串 `name` 被接受为被忽略的、输出中性的兼容扩展，用于将函数名镜像到工具结果的客户端。它不参与工具身份、提示渲染或输出。非字符串值是格式错误的；其他消息角色上的非空名称仍不受支持，因为它们携带已加载聊天模板无法表示的参与者身份。

对于常见生成的 OpenAI 兼容负载，`repetition_penalty` 仅在其中性值 `1` 时被接受，`mm_processor_kwargs` 在为空或仅包含 null 值时被接受。字符串形式的图像/视频 URL 也被接受。

格式错误的协议值返回字段特定的 HTTP 400 错误。无效的媒体源、字节或解码内容使用 `invalid_media`；远程获取和超时失败保留其专用的服务器错误码。规范化提示契约中的失败使用 `invalid_prompt`；类型化容量和可用性失败保留其专用代码。内部不变量失败不会被重新标记为客户端输入错误。

请求 `model` 必须等于公共模型 ID：默认是产物 `identity.model_id`，或显式 `--model-id` 覆盖。推理单独作为 `reasoning_content` 返回；答案文本保留在 `content` 中。

在 Chat Completions、Responses 和 Anthropic Messages 中，直接的顶层工具参数 `type`，或完全由显式原始类型组成的 `anyOf`/`oneOf`，会指导 Qwen 无类型参数文本的转换。它不决定结构完整的标记是否成为工具调用。接受字符串的值保持为字符串，包括空字符串。声明的非字符串参数的空块会被省略。被接受的 JSON 值保留其 JSON 类型；不区分大小写的布尔文本被规范化为 `true` 或 `false`。非空 schema 不匹配仍保持为结构化调用：有效 JSON 保留其表示的类型，其他文本变为 JSON 字符串，以便工具消费者报告验证错误并继续代理循环。没有受支持显式类型的 schema 保留无类型推断。NInfer 不应用默认值、不强制必需属性、不执行递归 JSON Schema 验证，也不使用约束解码。

字符串参数将函数/工具调用标记以及平衡的嵌套 `<parameter=...>...</parameter>` 文本保留为值字节。Qwen wire 格式没有分隔符转义，因此不匹配的嵌套参数开启符或独立的 `</parameter>` 无法无歧义地表示；任一情况都会导致整个工具调用区域回退为普通内容。

消息按输入顺序进入所选模板。维护的 Qwen 模板将 system/developer 消息保留在其原始位置。

携带提示的 JSON 对象在请求解析和提示渲染过程中保留其接收到的成员顺序，包括工具 schema 和历史工具输入。规范模型来源的工具参数在聚合和流式响应中保留该成员顺序，因此未修改的重放会重建相同的有序工具调用。NInfer 不会规范化语义等价的 JSON：如果客户端重新排序成员、插入默认值或以其他方式重写工具对象，则更改后的渲染输入不匹配模型持有的端点，只能复用更早的精确 checkpoint。

`--chat-template FILE` 选择本地 Jinja 模板；默认情况下，服务器使用存储在产物中的模板。示例参见 [CLI 指南](cli.md/#text-input)。

消息内容、工具数据或普通模板 kwargs 中引用的控制 token 拼写会被编码为文本。媒体占位符来自模板，并绑定到实际图像/视频输入。

`chat_template_kwargs` 在 Chat Completions、Responses 和 Anthropic Messages 中向模板传递 JSON 对象。与类型化请求字段重复的值必须一致。Null 标准选项表示未指定；其他 null 值保持为 `none`。消息、工具、生成模式和 tokenizer 特殊 token 不能通过 kwargs 覆盖。

`--default-thinking-budget N` 为以思考模式开始的请求设置正默认思考 token 上限。非思考请求不接收上限。它可以与 `--no-thinking` 共存，因为请求可以显式启用思考。Anthropic 的 `thinking:{"type":"enabled","budget_tokens":N}` 为该请求覆盖此默认值。

在启动命令中添加 `--default-thinking-budget 512`，以将每个启用思考的请求的模型来源思考限制在 512 token。

在上限边界，Engine 首先尊重自然 `</think>`、停止条件、取消或总输出/上下文限制。如果思考仍处于打开状态，它将 Qwen 的规范提前关闭指导及关闭标记提交到同一模型序列而不进行采样，将指导作为推理增量流式传输，并继续正常内容或工具调用生成。插入的 token 计入完成用量和请求的 `max_tokens`/`max_output_tokens` 预算。如果有效输出容量延伸到上限之后，但无法容纳完整的 tokenizer 派生控制后缀加一个关闭后模型 token，则准备会以 HTTP 400 代码 `thinking_budget_capacity_insufficient` 被拒绝，而不是部分插入控制。服务器不承诺模型在标记后会发出非空内容或工具调用。

对于 Chat Completions，`reasoning_effort: "none"` 请求禁用思考。所选模板解释其他标准值（`minimal`、`low`、`medium`、`high`、`xhigh`、`max`）。冲突的显式 `enable_thinking` 和 effort 值返回 `conflicting_template_option`。

`preserve_thinking` 根据所选模板控制推理保留。请求选项覆盖用 `--no-thinking` 和 `--preserve-thinking` 设置的服务器默认值。未指定的思考、effort 和保留选项使用模板的默认值。

流式传输以助手角色 chunk 开始，发送单独的推理和内容增量，然后是 finish-reason chunk 和 `[DONE]`。当 `stream_options.include_usage` 为 true 时，最后一个空 `choices` chunk 包含完成的用量。聚合和流式用量包括缓存的提示 token 和推理 token 详情；未请求 log 概率时，choices 携带 `logprobs: null`，聚合助手消息携带 `refusal: null`，因为不支持 refusal 输出。

### llama.cpp 兼容的请求观察

每个成功的 Chat Completions 响应都包含顶层 `timings` 对象。这是 llama.cpp 兼容的响应扩展，不是 OpenAI 字段。在流中，它附加到 `[DONE]` 之前的最后一个 JSON chunk：当 `stream_options.include_usage` 为 true 时是空 `choices` 用量 chunk，否则是 finish-reason chunk。

```json
{
  "timings": {
    "cache_n": 4096,
    "prompt_n": 4096,
    "prompt_ms": 83.0,
    "prompt_per_token_ms": 0.020263671875,
    "prompt_per_second": 49349.39759036145,
    "predicted_n": 129,
    "predicted_ms": 1140.0,
    "predicted_per_token_ms": 8.90625,
    "predicted_per_second": 112.28070175438596
  }
}
```



`cache_n` 是 Engine 证明的确切复用提示前缀，`prompt_n` 是剩余提示后缀，因此 `cache_n + prompt_n` 等于 `usage.prompt_tokens`。提示时间从准入提交该确切复用选择时开始，到第一个输出 token 被提交时结束。生成时间从该第一个 token 开始，到最后一个提交的输出 token 结束。因此，生成速度使用 `max(predicted_n - 1, 0)` 个 token 间隔；第一个 token 属于提示延迟，不再计为解码间隔。零 token、一 token、零时长和精确缓存命中情况报告有限零速率，而不是 `NaN` 或无穷大。发生 draft 工作时，推测性请求还包含终端 `draft_n` 和 `draft_n_accepted`。

在流式请求上设置顶层 `timings_per_token: true`，以将最新累积计时快照附加到每个可见推理或内容 chunk。这不会启用终端计时，终端计时始终存在。暂时被 UTF-8、停止字符串、推理或工具调用缓冲隐藏的模型提交仍会推进累积 token 计数；下一个可见 chunk 会观察到该已提交前沿。该选项会增加响应序列化和传输量，默认关闭。

设置顶层 `return_progress: true` 与 `stream: true` 一起使用，以接收提示处理 chunk：

```json
{
  "prompt_progress": {
    "total": 8192,
    "cache": 4096,
    "processed": 6144,
    "time_ms": 41
  }
}
```



初始事件具有 `processed == cache`。后续累积事件仅在相应预填充单元提交后发布，当消费者慢于预填充时可能被合并，并且永远不会倒退。最终事件具有 `processed == total`，并先于第一个输出增量。对于精确全前缀命中，初始事件已经具有 `cache == processed == total`，且不报告合成提示工作。`time_ms` 是自提交准入以来的经过墙钟时间；当分母非零时，客户端可以计算实际后缀进度为 `(processed-cache)/(total-cache)`。

### 多模态请求

在发送媒体之前使用 `--vision` 启动服务器：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```



OpenAI 图像和视频源可以是 HTTP(S) URL 或 base64 data URL。

文本和媒体请求使用一个完整提示上下文契约。在聊天模板渲染和媒体 token 扩展之后，结果必须适合 Engine `--max-context`。当前 Vision 运行时还有 32,768 合并 token 包络（131,072 原始 patch）；因此有效 Vision 限制为 `min(--max-context, 32768)`。没有固定的图像/视频项数限制：项数通过聚合源字节、解码像素、原始 patch、Vision token 和实时内存预算来准入。

媒体缓存未命中在有限主机工作池上作为独立的 decode → resize → BF16-pack 任务运行。准备好的负载以获取字节的 SHA-256 加模态为键，因此后续请求中的重复媒体会复用确切不可变 BF16 patch 输入；并发相同未命中使用一次 single-flight 构建。`--media-cache-mib` 限制 LRU 保留的负载，而 `--media-live-mib` 限制每个缓存、请求或运行时引用的负载。缓存驱逐不会使请求引用失效，实时字节仅在最终引用释放时返回。从实时限制派生的请求级准备门控防止并发部分构建使内存账户死锁。

超过 `--max-context` 的扩展提示返回 HTTP 400 `context_length_exceeded`，包括准备好的 token 数和配置的上下文上限。媒体预处理资源拒绝返回 HTTP 400 `media_budget_exceeded`。HTTP 413 `request_too_large` 保留给在 JSON 解析之前超过 `--max-request-mib` 的原始请求体；它不用于模型上下文或媒体资源错误。

## OpenAI 提示缓存

Chat Completions 和 Responses 将 OpenAI 缓存提示转换为可选的共享前缀写入候选：

- 省略 `prompt_cache_options` 会在最新的可表示内容边界创建默认隐式候选；
- `mode:"implicit"` 显式请求相同的自动候选；
- `mode:"explicit"` 禁用该请求的隐式写入；
- 受支持内容上的 `prompt_cache_breakpoint:{"mode":"explicit"}` 创建显式候选。

一个请求最多携带四个不同的写入。隐式目标占用一个槽位，除非它与显式目标重合；其余槽位包含最新的显式边界。更早的 schema 有效历史断点被接受，但不是新的写入候选。已发布前缀的精确读取不需要请求重复标记。

这些字段是优化提示。无法表示为确切渲染 token 前沿的合法边界会被忽略，而不改变提示内容。`prompt_cache_key` 不是 Engine 会话键或前缀身份。有效 TTL/保留值被接受，但 NInfer 不承诺其墙钟驻留；物理保留遵循资源调度器。

## OpenAI Responses Core

NInfer 实现了 OpenAI [Responses API](https://developers.openai.com/api/reference/resources/responses/overview) 的类型化 Item 和语义事件核心。所有受支持的模型实例都使用相同的适配器和 Engine 路由。它有意不被宣传为与 OpenAI 托管工具、持久云存储、后台作业、Conversations 或压缩完全对等。

### 创建 Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```



通过替换 base URL，同一端点可与 OpenAI SDK 一起使用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```



`output_text` 是 SDK 便利属性。它不作为顶层 wire 字段发出；wire 响应包含类型化 `output` Item。

### 创建请求字段

| 字段                                 | NInfer Responses Core 契约                                   |
| :----------------------------------- | :----------------------------------------------------------- |
| `model`                              | 必需非空字符串；必须等于产物派生的公共模型 ID 或显式 `--model-id` 覆盖 |
| `input`                              | 字符串或类型化 Item 数组；仅当 `previous_response_id` 已提供用户查询时，才可以省略或为空 |
| `instructions`                       | 可选字符串，仅为此请求插入到重建对话之前                     |
| `previous_response_id`               | 保留的本地 Response 的可选 ID                                |
| `max_output_tokens`                  | 非负整数；省略时使用 `--default-max-tokens` 执行，但在 Response 对象中保持为 `null` |
| `stream`                             | 布尔值；`true` 选择 Responses SSE 而不是 JSON 正文           |
| `store`                              | 布尔值，默认 `true`；控制本地检索和延续状态                  |
| `temperature`                        | `[0,2]` 中的有限数                                           |
| `top_p`                              | `[0,1]` 中的有限数                                           |
| `metadata`                           | 最多 16 个字符串对；键最多 64 字符，值最多 512               |
| `client_metadata`                    | Codex 客户端扩展；对象或 `null`，作为不透明跟踪元数据接受，无生成效果 |
| `reasoning.effort`                   | `none` 请求禁用思考；其他标准 effort 值传递给所选模板        |
| `chat_template_kwargs`               | 模板参数作为 JSON 对象；标准选项与类型化字段合并             |
| `preserve_thinking`                  | `chat_template_kwargs.preserve_thinking` 的别名；冲突值被拒绝 |
| `text.format`                        | 省略或仅 `{"type":"text"}`                                   |
| `tools`                              | 直接函数定义或包含函数定义的命名空间组；见下文               |
| `tool_choice`                        | `auto`、`none` 或仅函数的 `allowed_tools` 且模式为 `auto`；命名空间选择同时携带 `namespace` 和 `name` |
| `parallel_tool_calls`                | 默认 `true`；仅当没有有效工具可调用时接受 `false`            |
| `max_tool_calls`                     | 非负整数，作为托管工具 no-op 接受；NInfer 不执行托管工具     |
| `truncation`                         | 省略或 `disabled`；过长输入失败，而不是静默丢弃 Item         |
| `top_logprobs`                       | 省略或 `0`                                                   |
| `service_tier`                       | 省略、`auto` 或 `default`；响应报告 `default`                |
| `background`                         | 省略或 `false`                                               |
| `include`                            | 省略或空数组                                                 |
| `stream_options.include_obfuscation` | 可选布尔值；作为传输提示接受，但此本地服务器不发出填充       |
| 缓存和客户端提示                     | `prompt_cache_key`、`prompt_cache_options`、`prompt_cache_retention` 和显式断点遵循 OpenAI 提示缓存；`safety_identifier` 和 `user` 作为客户端提示接受 |

未知顶层字段失败并返回 `unknown_parameter`。已识别但不支持的功能失败并返回字段特定的 400 错误，而不是被静默忽略。

### 输入 Item 契约

字符串 `input` 被规范化为一个带有 `input_text` 部分的用户 `message`。数组输入接受：

| Item                   | 支持形式                                                     |
| :--------------------- | :----------------------------------------------------------- |
| `message`              | 角色 `user`、`assistant`、`system` 和 `developer`；字符串内容或类型化内容数组 |
| `input_text`           | 消息内容部分，包含字符串 `text`                              |
| `output_text`          | 助手消息重放部分，包含字符串 `text`                          |
| `refusal`              | 助手消息重放部分；其文本进入助手历史                         |
| `input_image`          | 用户或助手消息部分，带 HTTP(S) 或 data-URI `image_url`；detail 省略或 `auto`；需要服务器 `--vision` |
| `input_video`          | NInfer 扩展，带 HTTP(S) 或 data-URI `video_url`；需要服务器 `--vision` |
| `reasoning`            | 带 `reasoning_text` 内容的原始重放 Item；摘要/加密元数据可以伴随原始文本，但不能替代它 |
| `function_call`        | 完成的助手调用，带可选 `id` 和命名空间，以及必需 `call_id`、`name` 和 JSON 对象字符串 `arguments` |
| `function_call_output` | 完成的结果，带必需 `call_id` 和可选匹配的 name/namespace 断言；`output` 可以是字符串或非空的 `input_text`/`input_image` 部分数组 |

连续的助手拥有的 Item 按可表示顺序 `reasoning` -> 助手消息内容 -> `function_call` 形成一个助手历史轮次。多个消息 Item 追加其内容部分，多个调用保留声明顺序，仅推理轮次被保留。用户、system、developer 或 `function_call_output` Item 结束该组；需要重排助手内容的顺序失败并返回 `invalid_assistant_history`。结果通过 `call_id` 验证，并在提示渲染前重新排序为调用声明顺序；未知、重复或不可表示的部分结果集失败并返回 `invalid_tool_history`。规范输入 Item 保留客户端顺序。输入 Item ID 在提供时保留，否则生成；重复 ID 失败。

System 和 developer 消息 Item 在输入数组中保留其位置。顶层 `instructions` 表示为当前请求的前导 developer 轮次；目标特定的角色降级仅发生在 Qwen 系列前端中。

`input_text`、`input_image` 或工具结果部分可以携带 `prompt_cache_breakpoint:{"mode":"explicit"}`。写入选择遵循 OpenAI 提示缓存；边界影响复用机会，而不是提示身份或输出语义。字符串消息状态/阶段元数据被接受，但没有 Qwen 提示表示。

`input_file`、`input_audio`、图像 `file_id`、非 `auto` 图像 detail、没有原始推理文本的推理元数据、部分工具 Item 以及其他 Item/内容类型不受支持。存储在响应链中的 HTTP 媒体 URL 在继续该链时会再次获取；当历史媒体字节必须不可变时，使用 data URI。

### 函数工具

Responses 函数定义可以直接声明，而不是在 Chat Completions 的嵌套 `function` 对象内：

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  },
  "strict": false
}
```



它们也可以分组在 Responses 命名空间中：

```json
{
  "type": "namespace",
  "name": "mcp__weather",
  "description": "Weather service",
  "tools": [{"type": "function", "name": "get_current"}]
}
```



NInfer 为每个命名空间/函数对提供不同的内部 Engine 身份，并在聚合输出、SSE 事件和重放 Item 中恢复单独的 `namespace` 和 `name` 字段。因此同一函数名可以出现在不同命名空间中。命名空间成员仍是普通客户端执行的函数；这不会添加远程 MCP 执行器。

NInfer 在 Qwen 提示中渲染这些定义，并将模型输出解析为单独的 `function_call` 输出 Item。每个输出都有一个协议 Item `id`（`fc_...`）和一个不同的 `call_id`（`call_...`）。客户端执行函数并在后续请求中发送 `function_call_output` Item。只有当前有效工具集中的函数才能成为结构化调用；未声明的模型输出保持为普通文本。模式为 `auto` 的 `allowed_tools` 过滤该集合而不改变声明顺序，而 `tool_choice:"none"` 即使历史包含早期调用也会禁用结构化工具输出。

NInfer 不执行函数，也不通过约束解码强制执行 JSON Schema，因此 `strict:true`、必需或命名工具选择、托管工具、远程 MCP 工具和自定义自由形式工具被拒绝。延迟加载、输出 schema 以及排除直接调用的调用方限制也被拒绝，因为其语义无法被遵守。

### Response 对象和用量

终端 wire 响应具有 `object: "response"`，`status` 为 `completed`、`incomplete` 或 `cancelled` 之一，以及类型化 `output` 数组。NInfer 可以发出：

- 包含原始 `reasoning_text` 和空摘要的 `reasoning` Item；
- 包含 `output_text` 部分的助手 `message`；
- 一个或多个 `function_call` Item。

普通模型/字符串停止产生 `completed`。输出 token 或上下文容量耗尽产生 `incomplete`，并带 `incomplete_details.reason: "max_output_tokens"`。SSE 响应开始后接受的错误产生 `response.failed`；验证和准备错误保持为普通 HTTP 错误响应。`completed_at` 仅为已完成的 Responses 填充。仅推理的不完整结果不包含虚构的空助手消息。

用量是 checkpoint 原生的：

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```



`input_tokens` 包括聊天模板和扩展媒体 token。`cached_tokens` 是 Engine 复用的 checkpoint 证明的确切提示前缀。`output_tokens` 是接受的生成 token ID 计数，包括适用的被扣留停止 token。`reasoning_tokens` 在 Qwen 输出解码器中当接受的 token 仍处于推理通道时计数；它不是通过重新 token 化解码文本来估计的。

### Responses 流式传输

设置 `stream:true` 以获取语义 Server-Sent Events。每个帧都使用 SSE 事件名和匹配的 JSON `type`，并且每个 JSON 事件都有单调递增的 `sequence_number`：

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}
```



正常生命周期是：

1. `response.created`，然后 `response.in_progress`；
2. `response.output_item.added` 和 `response.content_part.added`；
3. 零个或多个 `response.reasoning_text.delta` 或 `response.output_text.delta` 事件；
4. 匹配的 `*.done`、`response.content_part.done` 和 `response.output_item.done` 事件；
5. 恰好一个 `response.completed`、`response.incomplete` 或 `response.failed` 终端事件。

函数参数使用 `response.function_call_arguments.delta` 和 `.done`。ID、输出索引和内容索引保持稳定，连接的增量等于终端 Item。Responses SSE 不发出 Chat Completions `[DONE]` 哨兵。启用工具时，普通答案文本仍立即流式传输；只有模糊的 `<tool_call>` 后缀或结构化工具区域被扣留。格式错误的工具标记会作为普通文本刷新回来，不丢失字节。

### 本地响应状态和资源

`store` 默认为 `true`。存储的 Responses 仅存在于该服务器进程中，并受 LRU 存储限制。它们会在重启时丢失，并且不是 OpenAI 的持久云保留服务。

`previous_response_id` 在新输入之前重建完整的已存储输入/输出 Item 历史。当前 `instructions` 值放在最前面，但不会保存到延续上下文中，这与 Responses 规则一致，即先前的顶层 instructions 不会向前传递。函数定义是请求配置而不是对话 Item，必须在工具结果轮次再次发送。重建的提示遵循普通 Engine 路径，因此兼容的 checkpoint 复用自然适用。

存储的 Response 还保留其解析后的 `preserve_thinking` 值。省略该字段的子项继承父值。显式不同的值创建新的语义分支；提示渲染和身份仍决定复用。仅更改布尔值永远不会使模型运行时已证明兼容的精确 checkpoint 失效。

对于 Engine 本地复用，存储的根 Response 接收一个从其响应 ID 派生的有界会话键，每个 `previous_response_id` 子项继承该键。`store:false` 根保持匿名；`store:false` 子项可以读取其继承的会话 checkpoint，但不会替换存储链的最新端点。Response 存储驱逐或删除会移除 HTTP 对象，而不是独立保留的 Engine checkpoint；后者仍受 Engine 自身保留和压力策略限制。HTTP schema 中不添加会话键或缓存标记。

资源行为：

| 端点                                 | 契约                                                         |
| :----------------------------------- | :----------------------------------------------------------- |
| `GET /v1/responses/{id}`             | 返回存储的终端对象，或 404 `response_not_found`；流恢复和非空 `include` 被拒绝而不是忽略 |
| `DELETE /v1/responses/{id}`          | 移除公共检索并返回 `response.deleted`；其他 Responses 已保留的后代上下文仍可用 |
| `GET /v1/responses/{id}/input_items` | 返回提供给该请求的规范化 Item；支持 `after`、`limit` `1..100`（默认 `20`）和 `order` `asc|desc`（默认 `desc`）；除非 `include=message.input_image.image_url`，否则图像 URL 被遮蔽 |
| `POST /v1/responses/{id}/cancel`     | 显式失败，因为不支持后台执行                                 |
| `POST /v1/responses/compact`         | 显式失败并返回 `compaction_not_supported`                    |

`store:false` Responses 无法检索或用作 `previous_response_id`。LRU 驱逐和显式删除也会使 ID 不可用。大于配置存储容量的单个 Response 失败并返回 `response_store_capacity_exceeded`，而不是静默假装已存储。

### Responses 输入 token 计数

`POST /v1/responses/input_tokens` 使用与 Create 相同的提示路径，并且不运行生成。它接受 `model`、`input`、`instructions`、`previous_response_id`、reasoning、函数工具和工具选择、受支持的 text/truncation 值以及 `preserve_thinking` 扩展。因此，父查找、调用 ID 规范化、模板渲染和媒体扩展与对应的 Create 请求完全相同：

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","input":"Count this prompt."}'
```



```json
{"object":"response.input_tokens","input_tokens":11}
```



不支持的 Create 字段包括 Conversations、提示模板、上下文管理、托管审核、Structured Outputs/JSON 模式、非空 `include`、后台执行、压缩、文件/音频以及 OpenAI 托管/MCP/自定义工具。这些是兼容性边界，不是静默接受的占位符。

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```



该端点接受顶层 System 文本、有序 User/Assistant/System 历史、文本和图像块、Thinking 历史、工具使用历史、工具结果、用户定义工具、聚合响应和 Anthropic SSE。连续 User 或 Assistant 消息被连接而不添加分隔符。对话中的 System 消息保留其输入位置。最后一条纯文本 Assistant 消息是 Assistant 预填充：生成继续其现有文本，而不是开启另一轮次。Assistant 预填充不能包含媒体、Thinking 或工具调用，并且不能在启用 Thinking 的情况下开始。

Claude Code 可能将其归属元数据放在顶层 System 数组的第一个块中。如果该块是恰好以 `x-anthropic-billing-header:` 开头的文本块，NInfer 在 token 计数、提示准备和缓存身份构建之前消费整个块。规则是位置性的：字符串形式 System 值、后续数组块或具有相同文本的内联 System 消息保持为普通提示内容。附加到被消费块的 `cache_control` 标记会随之消费，而不会移动到相邻内容。

`max_tokens` 对本地客户端可选，否则使用 `--default-max-tokens`；正值是完整输出预算。`max_tokens:0` 被拒绝，因为 NInfer 不暴露完成的零输出缓存预热生命周期。`temperature`、`top_p`、`top_k` 和 `stop_sequences` 进入 Engine 执行。匹配的自定义停止返回 `stop_reason:"stop_sequence"` 以及实际 `stop_sequence`；上下文耗尽返回 `model_context_window_exceeded`。

Thinking 支持 `disabled`、`adaptive` 和 `enabled`。启用的 Thinking 要求 `budget_tokens >= 1024` 且小于 `max_tokens`，并且该预算传递给 Engine。可见 Thinking 返回时不透明兼容签名；SSE 在关闭块之前发出其 `signature_delta`。请求降级从可见 `thinking` 文本重建本地提示，并将 `signature` 视为非语义传输元数据，因此保留的历史在 serve 重启后仍可用。`display:"omitted"` 被拒绝，因为 NInfer 无法提供 Anthropic 的加密隐藏推理恢复语义。`preserve_thinking` 仍是 NInfer 对已关闭轮次推理历史的扩展。`output_config.effort` 将其协议验证的值传递给所选模板。

用户定义的、非严格工具支持 `name`、`description`、对象 `input_schema` 和 `input_examples`。`tool_choice:auto` 和 `none` 可执行。强制或命名选择、`strict:true`、活动单调用强制、延迟工具、排除直接模型调用的工具、Anthropic 提供/服务器工具、工具集、MCP 和容器被拒绝，因为其所需约束或执行器不存在。`tool_result` 保留文本/图像顺序，并在模型提示中显式标记 `is_error:true`。对于可见的 Assistant 工具使用轮次，下一个 User 轮次必须为每个声明的 ID 提供一个前导结果；有效结果按 ID 匹配并规范化为调用顺序。以结果开始的历史作为截断或导入的对话仍然有效。

工具和受支持 System/User/Assistant/工具历史块上的块级临时 `cache_control` 创建显式共享前缀候选。最多接受四个不同的块级断点。请求级 `cache_control` 以最后一个可缓存块为目标：它与同一目标和 TTL 的显式断点合并，与同一目标但不同 TTL 冲突，并且当四个不同显式目标已存在时需要可用的第五个槽位。TTL 必须为 `5m` 或 `1h`；它是协议提示，不是墙钟驻留保证。

NInfer 将可表示边界映射到精确提示前沿，并忽略合法但不可表示的咨询边界而不改变提示。复用仍需要精确渲染身份，并且可以读取现有所有者而无需另一个 `cache_control`。聚合用量在 `cache_read_input_tokens` 中报告已验证复用 token，并将缓存创建留为未知。流式传输在 Engine 准入提交前缀选择之后、传输/预填充输出之前发出 `message_start`，因此其未缓存/缓存读取划分已经精确；终端累积用量与聚合响应匹配。

文档、搜索结果、文件、Structured Outputs、服务器工具结果、容器上传以及其他依赖执行的块被拒绝，并标识缺失能力。元数据、服务层级、推理地理位置、协议版本/beta 头、缓存 TTL 和未知咨询字段不会阻止原本可执行的请求。请求 `model` 是任意非空本地代理标签，并在响应中回显；它不选择常驻产物。

每个 Messages 响应都携带 `request-id` 头；错误正文也携带 `request_id` 并使用 Anthropic 错误类别。本地准入过载映射到 HTTP 529，队列/媒体超时映射到 HTTP 504。流式传输拥有 Thinking、文本和工具使用的完整 Anthropic 块生命周期。

`POST /v1/messages/count_tokens` 使用产物的 tokenizer、聊天模板和媒体扩展而不进行生成。它与 Messages 共享相同的提示规范化、工具、Thinking 模式、Assistant 预填充、媒体处理和缓存标记解释；仅输出采样和流式字段不影响计数：

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```



## 认证和 CORS

传递 `--api-key VALUE` 以要求与 OpenAI bearer token 或 Anthropic `x-api-key` 头相同的值。`GET /health` 和 CORS 预检请求保持不认证。

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```



`--cors` 添加宽松的浏览器 CORS 头。默认禁用。

## 服务器选项

该表列出了可执行默认值。启动示例选择长上下文 FP8/MTP3 配置。

| 选项                                                      | 含义                                                         | 默认值                    |
| :-------------------------------------------------------- | :----------------------------------------------------------- | :------------------------ |
| `--host H`                                                | 监听地址                                                     | `127.0.0.1`               |
| `--port N`                                                | 监听端口                                                     | `8080`                    |
| `--api-key KEY`                                           | 必需的 bearer 或 `x-api-key` 值                              | 未设置                    |
| `--model-id ID`                                           | 覆盖公共 OpenAI 模型别名                                     | 产物 `identity.model_id`  |
| `--max-context N`                                         | 每个序列的逻辑上下文上限                                     | `8192`                    |
| `--kv-capacity N|auto`                                    | 显式共享主文本 KV 容量，或从剩余 GPU 内存最大化；省略表示 `--max-context` | `8192`                    |
| `--max-concurrency N`                                     | 最大准入请求数；有效范围 `1..8`                              | `1`                       |
| `--max-pending-requests N`                                | 允许等待准入的额外请求                                       | `16`                      |
| `--pending-timeout-ms N`                                  | 最大准备加准入等待                                           | `30000`                   |
| `--prefill-chunk N`                                       | 文本预填充块                                                 | `1024`                    |
| `--log-stats-interval-ms N`                               | 聚合吞吐量报告间隔；`0` 禁用                                 | `5000`                    |
| `--log-level trace|debug|info|warning|error|critical|off` | pretty stderr 详细程度                                       | `info`                    |
| `--device N`                                              | CUDA 设备索引                                                | `0`                       |
| `--context-cost-presets FILE`                             | 可选运行时上下文成本预设注册表                               | 通用 + 编译默认值         |
| `--max-request-mib N`                                     | JSON 解析前的正文大小限制                                    | `384`                     |
| `--media-cache-mib N`                                     | LRU 保留的准备好的 BF16 媒体负载；`0` 禁用保留               | `1024`                    |
| `--media-live-mib N`                                      | 所有实时准备好的 BF16 媒体负载                               | `2048`                    |
| `--media-preprocess-threads N`                            | 有限媒体预处理工作线程；`0` 从主机并发选择最多 16            | `0`                       |
| `--request-log-jsonl FILE`                                | 追加全精度服务器/请求记录                                    | 禁用                      |
| `--response-store-max-records N`                          | 本地保留的 Responses 对象最大数                              | `1024`                    |
| `--response-store-max-mib N`                              | 本地 Response 信封/Item/上下文总预算                         | `256`                     |
| `--kv-dtype bf16|int8|fp8|nvfp4|k8v4`                     | KV 缓存存储                                                  | `bf16`                    |
| `--spec mtp|dflash|dflash2`                               | 推测后端                                                     | 关闭                      |
| `--draft-tokens N`                                        | MTP `1..5`；DFlash/DFlash2 `1..15`                           | 未设置                    |
| `--lm-head-draft`                                         | 优化提议头                                                   | 关闭                      |
| `--default-max-tokens N`                                  | 请求省略时的输出限制                                         | `8192`                    |
| `--default-thinking-budget N`                             | 启用思考的请求继承的正思考上限                               | 未设置                    |
| `--vision`                                                | 启用媒体输入并加载 Vision GPU 分配                           | 关闭                      |
| `--no-cuda-graph`                                         | 禁用 CUDA Graph 解码                                         | 图开启                    |
| `--no-prefix-reuse`                                       | 禁用兼容前缀缓存                                             | 前缀复用开启              |
| `--device-state-slots N`                                  | 活动通道保证之外的额外 Device checkpoint StateImage          | `max-concurrency`         |
| `--host-state-slots N`                                    | pinned Host StateImage 容量                                  | `8`                       |
| `--host-kv-mib N`                                         | 共享 pinned Host 主/后端 KV 字节容量（MiB）                  | `8192`                    |
| `--max-private-continuations N`                           | 私有延续描述符容量                                           | `2 * max-concurrency`     |
| `--max-shared-prefixes N`                                 | Engine 范围共享稳定前缀描述符容量                            | `max(max-concurrency, 4)` |
| `--max-long-anchors-per-continuation N`                   | 每个延续的私有长锚点限制                                     | `2`                       |
| `--no-thinking`                                           | 默认禁用思考                                                 | 思考开启                  |
| `--preserve-thinking`                                     | 默认保留已关闭轮次助手推理                                   | 关闭                      |
| `--cors`                                                  | 宽松浏览器 CORS 头                                           | 关闭                      |
| `--temperature F`                                         | 进程级 temperature 覆盖                                      | 未设置                    |
| `--top-p F`                                               | 进程级 top-p 覆盖                                            | 未设置                    |
| `--top-k N`                                               | 进程级 top-k 覆盖（`0..20`；零选择 top-20 上限）             | 未设置                    |
| `--min-p F`                                               | 进程级 min-p 覆盖                                            | 未设置                    |
| `--presence-penalty F`                                    | 进程级存在惩罚覆盖                                           | 未设置                    |
| `--frequency-penalty F`                                   | 进程级频率惩罚覆盖                                           | 未设置                    |
| `--seed N`                                                | 请求省略时的固定种子                                         | 每请求新随机种子          |
| `--greedy`                                                | 对所有请求强制精确 argmax                                    | 关闭                      |

上下文成本系数在启动时从通用默认值、匹配的编译值以及可选的 `--context-cost-presets FILE` 传输或预填充条目解析一次。预填充条目匹配硬件以及从实际 Text/Vision 配置、绑定和 Uses 派生的签名。没有匹配测量的新表示使用通用预填充系数。格式错误的文件会中止启动；操作上下文成本记录和 JSONL `server_start` 标识所选来源。

Engine 从加载的架构和请求的解析思考模式中选择采样默认值。Qwen3.6-27B 和 Qwen3.8-27B 在思考模式下使用 `1.0/0.95/20/0/0`（temperature/top-p/top-k/min-p/presence penalty），在非思考模式下使用 `0.7/0.80/20/0/1.5`。Qwen3.6-35B-A3B 仅其思考存在惩罚不同，为 `1.5`。所有注册预设的频率惩罚为 `0`。进程标志覆盖注册值，请求字段覆盖进程标志，`--greedy` 最后强制 temperature 为 `0`。

对于 `C=--max-concurrency` 和 `H=--device-state-slots`，总 Device StateImage 容量为 `C+H`：`C` 个槽位保证活动请求，`H` 是全局 checkpoint 池。Host State 和 Host KV 是独立的启动固定 pinned 内存容量；Host KV 由主池和所选后端池共享，并以物理页范围消耗。`--no-prefix-reuse` 选择仅根 Engine 模式，不能与七个显式上下文缓存容量标志中的任何一个组合，包括零值标志。

运行 `./build/apps/ninfer-serve --help` 获取确切选项契约。

Serve 使用 `YYYY-MM-DD HH:MM:SS.mmm LEVEL message` 向 stderr 写入人类可读的操作记录。正常输出涵盖重要启动里程碑、就绪状态、请求生命周期、固定间隔吞吐量和关闭；`--log-level debug` 暴露内部启动和资源规划细节。终端可以在启动期间使用一个瞬态行，但 Serve 吞吐量始终是持久记录。重定向的 stderr 不包含终端控制序列。Pretty 值使用可读单位和四舍五入速率；使用独立请求 JSONL 获取完整字段和全精度。操作记录从不包含提示、生成文本、请求正文、凭据或任意客户端错误消息。如果工具标记因其结构或工具身份无法表示而返回为文本，Serve 会发出一个仅包含失败分类的警告，绝不包含生成的标记。

## 结构化请求日志

`--request-log-jsonl FILE` 启用机器可读测量日志。服务器以追加模式打开 `FILE` 并刷新每个事件，因此连续模型或 MTP 块可以共享一个 campaign 文件。父目录必须已存在。打开文件失败会中止启动；如果日志路径解析到模型产物，也会被拒绝。

每行是一个 `ninfer_serve_request_log` schema-v21 JSON 对象。所有事件都携带 `timestamp_unix_ms` 和进程唯一的 `server_instance_id`；请求 ID 仅在该服务器实例内单调。成功的请求开始记录包括请求范围的获取、媒体预处理墙钟/工作、tokenizer、缓存命中/未命中/single-flight 以及负载大小字段；它们不从进程全局计数器增量推断请求行为。

| 事件               | 内容                                                         |
| :----------------- | :----------------------------------------------------------- |
| `server_start`     | 产物路径、架构、公共名称、实际格式和预填充签名；解析的 Engine 和上下文缓存容量、思考/非思考采样器默认值及进程覆盖、思考历史和思考预算默认值、Device arena、统一工作区内可选的非加性 Vision 布局、Host State/KV 容量和占用、KV 大小分类账、CUDA Graph 允许量、CUDA/GPU 环境以及遮蔽的 argv |
| `request_start`    | 协议、解析的采样器和种子、请求的推理 effort、实际初始思考模式和可选预算、Responses 语义更改标志、输出预算、流/消息/工具形状 |
| `request_rejected` | 解析的请求形状、请求的推理 effort、媒体项数、`phase: "prepare"`，以及同步准备拒绝的确切 HTTP 状态/类型/代码/参数/消息 |
| `request_done`     | 完成原因、提示/完成/缓存/计算预填充 token、前缀复用路径、工具调用解析诊断、请求拥有的物化成本/搜索诊断、思考预算应用计数器、未四舍五入的请求阶段秒数、每请求 Engine Host 暴露以及完整的推测解码计数器 |
| `request_error`    | 解析的请求配置以及生成、取消或结果前传输终端消息             |
| `throughput`       | 间隔 token/解码/上下文缓存压力计数器增量、权威工作线程 Host 工作增量、当前调度器/资源仪表以及解码轮批次统计 |

`requested_reasoning_effort` 和 `preserve_thinking` 记录显式选项，未指定时为 `null`。`enable_thinking` 记录响应是否以思考模式开始。

`request_done.result.tool_call_parse` 记录是否看到完整标记、结构化调用计数、规范化期间省略的空非字符串参数、为消费者验证保留的 schema 不匹配参数以及稳定的文本回退原因。回退原因有 `none`、`malformed_structure`、`duplicate_parameter`、`invalid_tool_name`、`undeclared_tool` 和 `trailing_content`。这些计数器不包含工具参数或生成文本。

`request_done.timings_seconds` 包含 `prepare`、`ttft`、`vision`、`prefill`、`decode` 和 `total` 作为全精度 JSON 数字。其 `speculative` 对象包含 `backend`、`draft_window`、`rounds`、`drafted_tokens`、`accepted_tokens`、`fallback_steps` 和 `accepted_per_position`。速率可以在下游从原始 token 计数和秒数导出，而不是四舍五入的 stderr 字符串。

对于 `server_start.memory`，`workspace.capacity_bytes` 是唯一的物理工作区分配。启用 Vision 时，`vision_workspace` 报告聚合提示和最大项 token 界限，以及同一分配内的编码峰值和交接布局/使用；这些字节不得添加到 `workspace.capacity_bytes`。禁用 Vision 时该字段为 `null`。

`request_done.engine_timing` 分离 FIFO `queue_wait_seconds`、阻塞 `device_wait_exposed_seconds` 以及 `host_exposed_seconds` 下的五个互斥 Host 活动暴露阶段：`engine_boundary`、`program_submit`、`program_post`、`engine_commit_output` 和 `engine_maintenance`。`total` 正好是它们的总和，不包括 Device 等待。嵌套 `decode` 对象报告请求的解码类 Host 暴露、Device 等待和轮数；`units` 报告其预填充/控制单元计数。在紧凑批次中，每个参与请求都被整个轮次延迟，因此这些值解释请求延迟，但**不得跨并发请求求和**。

JSONL 文件不包含生成的响应文本，也从不记录 API 密钥值；`argv` 将该值替换为 `<redacted>`。操作 stderr 摘要是四舍五入的，不是聚合来源。OpenAI Responses、OpenAI Chat 和 Anthropic 生成请求在进入同步准备时接收请求 ID。成功准备产生 `request_start`；准备失败产生 `request_rejected`，没有匹配的开始。每个已开始的生成事务然后恰好有一个机器终端：Engine 返回其结果时 `request_done`，或在结果存在前生成失败时 `request_error`。后续响应渲染、Responses 存储或终端传输失败仅是操作响应事件，不会添加第二个 JSONL 终端。准备前的 schema/模型验证拒绝和仅 token 计数调用不是测量请求，不接收请求 ID。

默认情况下，服务器每五秒持续报告聚合活动。`prefill` 计数间隔期间实际计算的提示后缀 token，不包括前缀缓存命中；`decode` 计数解码轮最终提交的 token，不包括预填充产生的第一个 token。对于 MTP、DFlash 和 DFlash2，这是接受的已提交输出，不是 draft 或被拒绝 token。Pretty `batch` 和 JSONL `average_size` 是同一间隔内解码行轮除以解码轮。`running`、`prefilling`、`decode_ready`、`waiting`、`materializing`、`capture_pending` 和 `terminal_pending` 字段是间隔结束时 Engine 调度器快照。JSONL `context_cache` 对象报告选择、捕获、传输、COW、压力溢出、私有/共享所有者降级和驱逐、checkpoint 丢弃、压力搜索、预算耗尽、最大回退和历史分叉计数器作为间隔增量；`occupancy` 和 `last_selection` 是间隔结束仪表。物化预测是请求拥有的，仅出现在对应的 `request_done` 事件上。`pressure.searches` 计数接受到 Program 资源事务中的计划，包括后来以请求本地中止结束的事务；已提交受害者计数器同样报告产生的稳定缓存更改。

JSONL `throughput.host_work` 对象是聚合权威：Engine 工作线程对每个墙钟时间段计数一次，与批次大小无关。`elapsed_seconds` 包含相同的五个互斥 Host 阶段及其 `total`；`device_wait_seconds` 是分开的。`work_class_seconds` 将 Host 和 Device 等待时间分为解码、预填充和控制类。`detail_subset_seconds` 和 `detail_invocations` 暴露准入、上下文事务、副本和统计发布慢路径；这些详细值已包含在顶层 Host 阶段中，不得添加到 `total`。每轮、每行轮和每次调用规范化值在其分母为零时为 `null`。Pretty 吞吐量包含非零 token 率和计数、当前 running/prefill/decode-ready 组成、非零 waiting/materialization/terminal 状态、平均解码批次以及 Host 活动时间及其占间隔的比例。使用 JSONL 进行完整测量分析。即使没有 token 执行，包含上下文物化或保留活动的间隔也会保留；只有完全空闲的间隔被省略。下游测量应优先使用原始计数器和秒数，而不是四舍五入的 stderr 速率。

## 执行行为

服务器拥有一个常驻 Engine，启动固定容量为 `1..8` 个活动生成请求。在每个解码边界，每个解码就绪请求被压缩到一个批次中，并由一次模型遍历和（启用图时）一次精确批次 CUDA Graph 重放处理。请求仅在其单请求预填充完成后加入该批次；当它完成或被取消时，下一个边界重建批次而不留空行。

`--max-pending-requests` 限制活动集之后等待的请求。总生成请求生命周期容量为 `max_concurrency + max_pending_requests`，包括仍在 CPU/媒体准备中的请求和已完成模型结果但响应尚未释放的请求。容量满返回 HTTP 429 和代码 `server_overloaded`。绝对 `--pending-timeout-ms` 截止时间在准备之前开始，涵盖媒体获取和 Engine FIFO 等待，如果未及时准入则返回 HTTP 503 和代码 `request_queue_timeout`。没有准入 ETA 或无界溢出队列。

输入内存受未完成请求数和每请求 `--max-request-mib` 限制约束。媒体请求额外共享一个准备许可，因此等待媒体请求保留相同的取消和超时截止时间。模型输出受相同有限请求数和每个请求的有效输出 token 限制约束；输出回调和网络序列化在 GPU 执行器之外运行，不会延迟下一批次的形成。

`--max-context` 是每个序列的逻辑上限。`--kv-capacity` 修复活动请求和保留前缀使用的共享主文本 KV 池。`auto` 考虑完整的启用运行时并留下 1 GiB 大小余量；省略该选项时它跟随 `--max-context`。容量在启动时解析一次。

准入在请求完成前预留完整的提示加有效输出页权益。请求保持排队，直到合法资源计划能满足该权益。

每个可复用 checkpoint 包含 KV 和完整延续状态。在准入、捕获和完成边界，资源压力可能将其保留在 Device、将其 StateImage 和/或 KV 副本移动到 pinned Host 内存，或驱逐它。规划器将传入请求工作与强加于保留 checkpoint 的后续恢复成本进行比较。活动请求保留其状态和完成预留，放置选择保留模型语义。完整策略和不变量定义在 [资源调度和上下文缓存](maintainer/resource-scheduling-and-context-cache.md)。

除非服务器以 `--no-prefix-reuse` 启动，否则文本和多模态历史都会复用兼容前缀。多模态命中还要求匹配 token 类型、三轴 MRoPE 位置、编码媒体摘要、网格和消费者跨度。完全位于匹配前缀内的媒体跳过 Vision 执行，而新后缀媒体正常编码。Pretty 完成记录使用可读路径标签显示 `cache N (P%, path)`；JSONL 保留确切的 `prefix_cache_hit_tokens` 和 `prefix_reuse_path` 字段。机器路径有 `root`、`private_endpoint`、`private_turn_closure`、`private_response_replay`、`private_long_anchor` 和 `shared_stable_prefix`。复用验证涵盖 KV、循环状态、隐藏状态、所选后端状态和确切提示前沿。在稳定的 `preserve_thinking=true` 下，辅助 checkpoint 滚动到当前响应确定性生成序言之前不久的消息前沿。因此，规范化响应、紧凑摘要指令或替换用户后缀会重放小型生成序言和仅更改的后缀，同时保留完整稳定对话前缀。稳定的 `false` 将轮次关闭 checkpoint 放在开放轮次中第一个助手开启符之前，因此关闭该轮次可以重新计算其开启符并省略其推理，而不丢弃前面的对话。

`preserve_thinking` 选择新创建 checkpoint 的捕获前沿。现有精确 checkpoint 在模式更改后仍可复用。如果期望边界在所选复用前沿之后且没有快照，Engine 保留有效命中并推迟新 checkpoint。后来在每个保留 checkpoint 之前分叉的请求从根开始。JSONL 完成记录将恢复的 checkpoint 暴露为 `prefix_reuse_path`。推理 effort 更改参与渲染 token 身份和精确前缀选择。

追加的对话中 system 消息是普通提示后缀，因此未更改的先前历史仍符合 `private_endpoint` 条件。如果客户端修改、移除或移动历史 system 消息，token 前缀确实不同，未命中/重置是正确的。

推测后端保留协议输出形状、停止行为和用量计数。如果停止截断多 token MTP、DFlash 或 DFlash2 轮次，Engine 提交确切接受的 target 前缀，以便后续兼容轮次可以复用它。输出限制和上下文容量完成映射到 `length`/`max_tokens`；普通模型或字符串停止映射到 `stop`/`end_turn`。

函数工具渲染到模型提示中，生成的调用解析为协议响应。NInfer 不执行工具，也不通过约束解码强制执行客户端 JSON Schema。

提示 token 用量包括聊天模板和扩展媒体 token。生成 token 用量来自接受的输出 token ID，包括解码文本可能被扣留的停止 token。