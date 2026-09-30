# Stable CLI 示例

本目录包含已提交的、离线的 `--messages` 输入，用于从仓库根目录测试产品 CLI。它涵盖短文本、聊天历史、图像、视频、混合多模态历史、困难思考问题、长解码以及四种长上下文长度。这是面向操作者的示例集，而不是第二套正确性框架。

[`manifest.json`](manifest.json/) 列出了冻结的对比用例、其预期观察结果、推荐的运行时预算以及准备好的提示词 token 数。当独立探索性提示词有意没有冻结的输出判定基准时，它们也可以与这些用例放在一起。

## 快速开始

请从仓库根目录运行，因为 JSON 文件中的媒体路径是相对于仓库的：

```bash
CLI=./build/apps/ninfer
MODEL=models/qwen3_6_27b.ninfer

$CLI "$MODEL" \
  --messages examples/cli/messages/text_smoke_zh.json \
  --no-thinking --greedy --max-new 8
```



预期的 stdout 应恰好为 `42`。`--no-thinking --greedy` 是简单用例的正常对比模式。推理、进度、计时、内存和 MTP 统计信息写入 stderr；答案内容写入 stdout。

## 将同一 fixture 发送到 Serve

`send_to_serve` 将一个 CLI messages 文件提交到已运行的 OpenAI Chat Completions 端点。请将其作为仓库模块运行，以便使用维护中的 Serve 客户端。本地 CLI 图像和视频路径在客户端读取，并编码为数据 URI；服务器永远不会收到本地文件系统路径。

```bash
python3 -m examples.cli.send_to_serve \
  examples/cli/messages/image_chart.json \
  --base-url http://127.0.0.1:8080 \
  --model qwen3.6-27b --no-thinking --max-tokens 64
```



为媒体 fixture 启动 `ninfer-serve` 时请加上 `--vision`。相对媒体路径默认使用当前工作目录；在其他位置调用命令时，请传入 `--media-root`。`--dry-run` 会打印确切转换后的请求，而不联系服务器。

## 文本和多模态用例

```bash
$CLI "$MODEL" --messages examples/cli/messages/text_chat_history.json \
  --no-thinking --greedy --max-new 32

$CLI "$MODEL" --messages examples/cli/messages/text_code_review.json \
  --no-thinking --greedy --max-new 256

for CASE in image_chart image_natural video_temporal multi_image_compare \
            mixed_image_video mixed_multiturn; do
  $CLI "$MODEL" --messages "examples/cli/messages/${CASE}.json" \
    --max-context 8192 --no-thinking --greedy --max-new 128 --vision
done
```



受控观察结果如下：

| 用例                  | 预期观察结果                                                 |
| :-------------------- | :----------------------------------------------------------- |
| `text_chat_history`   | `Cedar|2041-09-17|37`                                        |
| `text_code_review`    | 空输入会除以零；添加显式的空输入策略                         |
| `image_chart`         | `NIFER VISION 731`；三个红色圆圈；左侧蓝色正方形             |
| `image_natural`       | 邮箱 `24`；太阳在右侧                                        |
| `video_temporal`      | 红色圆圈移动；绿色 `3`；当 `3` 出现时正方形仍然可见；结尾 `9` |
| `multi_image_compare` | 两个圆圈、三个圆圈和一个新的黄色星形                         |
| `mixed_image_video`   | `NIFER-9`                                                    |
| `mixed_multiturn`     | `24-9`                                                       |

要通过同一输入路径测试 MTP：

```bash
$CLI "$MODEL" --messages examples/cli/messages/text_smoke_zh.json \
  --no-thinking --greedy --max-new 8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```



## 思考用例

对于这些输入，请不要传入 `--no-thinking`。给推理留出足够空间以完成并过渡到答案内容；模型可以在达到请求的最大值之前自行停止。

```bash
$CLI "$MODEL" --messages examples/cli/messages/thinking_logic_grid.json \
  --greedy --max-context 16384 --max-new 8192

$CLI "$MODEL" --messages examples/cli/messages/thinking_multimodal_checksum.json \
  --greedy --max-context 8192 --max-new 4096 --vision

$CLI "$MODEL" --messages examples/cli/messages/reasoning_jacobian_counterexample_3d.json \
  --greedy --max-context 32768 --max-new 16384
```



逻辑网格只有一个解，并且必须以 `CHECK=4606` 结尾。多模态用例从两张图像和一个视频中读取独立事实，然后必须以 `CHECKSUM=2238` 结尾。Jacobian 提示词是一个独立的精确证明压力用例，没有冻结的模型输出判定基准，因此它有意不属于对比 manifest 的一部分。

## 长解码

三个冻结的 AIME 2026 用例有意获得充裕的预算。它们的目的是运行到模型的停止 token，而不是发现恰好能容纳某个输出的最小 `max-new` 值。

```bash
for CASE in 01 15 30; do
  $CLI "$MODEL" --messages "examples/cli/messages/long_decode_aime26_${CASE}.json" \
    --greedy --max-context 262144 --kv-dtype int8 --max-new 65536
done
```



用例 01、15 和 30 的盒装整数答案分别为 `277`、`83` 和 `393`。

## 长上下文

这些提示词长度包括禁用思考时的聊天模板。每个输入冻结一篇长文档，并将同一个检索针放置在 50% 深度处。

```bash
$CLI "$MODEL" --messages examples/cli/messages/long_niah_8k.json \
  --max-context 262144 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 128

$CLI "$MODEL" --messages examples/cli/messages/long_niah_64k.json \
  --max-context 262144 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 128

$CLI "$MODEL" --messages examples/cli/messages/long_niah_128k.json \
  --max-context 262144 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 128

$CLI "$MODEL" --messages examples/cli/messages/long_niah_256k.json \
  --max-context 262144 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 128
```



四个用例都必须输出：

```text
ORCHID=493817; COLOR=COBALT
```



已提交的提示词 token 数已针对 Qwen3.6-27B 和 Qwen3.6-35B-A3B 前端配置进行验证，这些配置为这些文件生成相同的序列。Qwen3.8-27B 使用相同的 CLI 接口，但带有自己的 tokenizer 和聊天模板资源；使用这些 fixture 时，请检查其准备好的 token 数。

## Fixture 构建

所有 PNG 和 MP4 媒体均为项目作者制作的确定性场景。视频为 5 秒、8 FPS，包含四十个 H.264 帧。运行时测试从不依赖网络 URL 或可变外部内容。

已提交的文件是规范输入；不存在树内重新生成脚本。有意替换时，必须同时更新文件、其 manifest 哈希、提示词 token 数和判定基准。