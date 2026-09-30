# NInfer

> 精选检查点。单 GPU 推理性能最大化。

NInfer 是一个从零编写的 C++/CUDA 推理引擎，面向单块 NVIDIA GeForce RTX 5090 上的 Qwen3.5 Dense 和 MoE 架构。它通过本地 CLI 或兼容 OpenAI/Anthropic 的 HTTP API 运行文本、图像和视频提示。该运行时是刻意专用化的：一块 GPU、一个常驻模型，以及启动时固定的一到八个活动请求容量。

共有五个官方产物可用。快速开始命令使用 Qwen3.8-27B NVFP4。

| 模型            | 权重            | 产物                       | 下载与模型卡                                                 |
| :-------------- | :-------------- | :------------------------- | :----------------------------------------------------------- |
| Qwen3.6-27B     | `groupwise-int` | `qwen3_6_27b.ninfer`       | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B     | `nvfp4`         | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B     | `groupwise-int` | `qwen3_8_27b.ninfer`       | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B     | `nvfp4`         | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer`   | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

每个 v3 `.ninfer` 产物都携带模型配置、编码后的权重、逻辑绑定以及前端资源。运行时执行会将这些事实与已实现的模型和算子能力结合使用。你还可以[转换自己的权重](docs/weight-conversion.md)，复用官方配方，或选择另一种受支持的格式组合。

当前引擎要求 v3 产物。现有的官方 v2 下载可以在本地[升级](docs/weight-conversion.md#upgrade-an-existing-v2-artifact)，无需再次下载权重。

## 快速开始

NInfer 需要 64 位 Linux、一块 NVIDIA GeForce RTX 5090、支持 `sm_120a` 的 CUDA 工具包、CMake 3.28 或更新版本、C++20 主机编译器、Ninja、`pkg-config`、FFmpeg 开发库（`libavformat`、`libavcodec`、`libavutil` 和 `libswscale`），以及 `libcurl >= 7.85`。CUDA 13.1 是经过验证的开发工具包；CMake 不设 CUDA 版本下限。构建会拒绝除 `sm_120a` 以外的 CUDA 架构。

构建产品二进制文件：

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 本机构建差异（nv51）

本机有三处与上面的通用要求不同，直接照抄会失败：

| 差异 | 报错/现象 | 处理 |
| :--- | :--- | :--- |
| CMake 是 3.25.1，低于要求的 3.28 | `CMake 3.28 or higher is required` | 用户目录已装 CMake 4.4.3（`~/.local/bin`，在 PATH 中）；若终端是在安装之前打开的，先执行 `hash -r` |
| PATH 里的 `nvcc` 是 CUDA 11.8 | 不支持 `sm_120a`，构建被拒绝 | 显式指定 `CUDACXX=/usr/local/cuda-13.0/bin/nvcc` |
| 系统 FFmpeg 是 5.1，低于实际需要的 6.1 | `AVCodecParameters` 没有 `coded_side_data` 成员、`av_packet_side_data_get` 未声明 | 用 conda 在用户目录装一份 FFmpeg 6.1，并用 `PKG_CONFIG_PATH` 指向它 |

本机可用的完整命令：

```bash
hash -r   # 终端若在安装 CMake 4.4.3 之前打开，需要清掉 bash 的命令缓存

export PKG_CONFIG_PATH="$HOME/.conda/envs/ninfer-ffmpeg/lib/pkgconfig"
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32
```

- **FFmpeg 的最低版本是 6.1，这一点源码里没有明说。** `cmake/Dependencies.cmake` 只写 `REQUIRED` 不设版本下限，`docs/maintainer/build-system.md` 也明确说「不施加版本下限，实际 API 支持由编译和测试来检验」。实际判据来自 `Dockerfile`：构建阶段用 `ubuntu24.04` 的 `-dev` 包，运行阶段装 `libavcodec60 / libavformat60 / libavutil58 / libswscale7` —— 这正是 FFmpeg 6.1 的 soname。Debian 12 自带的 5.1 会在 `src/media/decode/decode.cpp` 上编译失败。
- 本机的 6.1 装在 conda 环境 `~/.conda/envs/ninfer-ffmpeg`（conda-forge 的 `ffmpeg=6.1` 单包即包含头文件、`.pc` 与运行库）。用 `conda env remove -p ~/.conda/envs/ninfer-ffmpeg` 即可完全卸载。系统未安装对应 `-dev` 包且当前账号无 root，所以走用户目录。
- 用 `-j 32` 而不是 `-j`：本机 128 核，`nvcc` 每个实例占用 2–4 GB 内存，全部并发容易压垮机器。项目本身在 `CMakeLists.txt` 中已把链接限制为串行（`ninfer_link=1`）。
- 若中途换过 CMake 版本或 FFmpeg 路径，重新配置前先 `rm -rf build`（旧的 `CMakeCache.txt` 会导致失败）。
- configure 完成后，`cmake --build` 不再需要上面两个环境变量：路径已缓存进 `build/CMakeCache.txt`。

> 这一节记录的是本机环境特有的绕行方式。如果之后要把本文件作为 README.md 的正式译文提交，应删除本节。

测试和基准测试不包含在默认构建中。`cmake --preset release` 配置相同的产品构建；`cmake --preset dev` 还会启用测试和基准测试，并查找 Python 3 解释器。两个预设都使用 `build/` 并显式重置构建选项。机器特定的编译器和 Python 路径应放在被忽略的 `CMakeUserPresets.json` 中。详情请参见[构建组织与配置](docs/maintainer/build-system.md)。

没有安装目标，也没有打包的二进制分发；请从其源码构建树运行 NInfer。Python 工具独立于 CMake 运行；独立的 HBM 探测工具有自己的[构建命令](tools/README.md#standalone-hbm-probe)。

使用 Hugging Face CLI 下载本示例所用的产物：

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```



启动一个长期运行的文本/代理服务器，具有两个活动请求通道以及显式的 Device/Host 检查点容量：

```bash
./build/apps/ninfer-serve /data0/cyh/ninfer_models/qwen3_8_27b_nvfp4.ninfer \
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
  --device 2 \
  --port 8081
```



每个请求有 240,000 token 的逻辑上限。一个共享的 240,000 token Device KV 池为已接纳的请求服务；当两个请求的预留总量能够容纳时，它们会并发运行。缓存层级提供两个 Device 检查点槽位、八个 pinned Host State 槽位，以及两个活动 StateImage 之外的 8 GiB pinned Host KV。

发送一个 OpenAI 风格的请求：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```



运行一个带 32,768 token 分配的一次性 CLI 请求：

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --device 0
```



回答内容写入 stdout。人类可读的启动/运行时诊断信息以及 CLI 拥有的推理、计时、吞吐量、内存和投机解码报告写入 stderr；推理内容和结果报告保持为无前缀的产品输出。在终端上，权重物化使用一行临时进度行，随后是一个紧凑的 Engine-ready 摘要。重定向的 stderr 会收到持久可读的进度信息，不带终端控制序列。使用 `--log-level debug` 获取完整的启动详情。选项和本地输入错误仍为直接命令诊断信息。使用 `--messages FILE` 和 `--vision` 进行结构化图像/视频输入；请参见 [CLI 指南](docs/cli.md) 和[已提交的示例](examples/cli/)。

## 资源感知的长上下文复用

可复用的前缀检查点包含 KV 以及其精确提示前沿的完整延续状态。驻留在 Device 上的检查点可直接恢复。在压力下，规划器会根据即时恢复工作和后续复用成本，权衡 Device 保留、pinned Host State/KV 以及逐出。活动请求保留其完成预留。

算法请参见[资源调度与上下文缓存](docs/maintainer/resource-scheduling-and-context-cache.md)，公共 HTTP 覆盖（包括热复用、Host 恢复、逐出、共享前缀、调度边界和多模态负载）请参见 [Serve TTFT 基准测试](tools/bench/ttft/)。

## 性能

已发布的测量数据使用 RTX 5090。[性能索引](docs/performance.md)链接到各模型的运行记录和[测量规则](docs/performance/methodology.md)。下表是这些详细结果的摘录。

### 并发 MTP3 解码
MTP（Multi-Token Prediction，多 Token 预测），一次猜3个token。
饱和解码使用 INT8 group-64 KV、CUDA Graphs、MTP3，以及每个活动请求一次 8,192 token 的生成。吞吐量使用完整区间内聚合的已提交解码 token，这些区间的实际解码批次等于配置的并发数。接受率覆盖整个请求波次；这些速率是稳态解码（tok/s）。

| 模型配置                                                     | C=1 tok/s / 接受率 | C=2 tok/s / 接受率 | C=4 tok/s / 接受率 | C=8 tok/s / 接受率 | C8 / C1 |
| :----------------------------------------------------------- | :----------------- | :----------------- | :----------------- | :----------------- | :------ |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2%      | 247.0 / 69.0%      | 309.5 / 68.4%      | 535.0 / 68.3%      | 2.88×   |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3%      | 399.7 / 71.4%      | 699.7 / 69.3%      | 1,146.9 / 68.6%    | 5.67×   |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6%      | 907.2 / 66.3%      | 1,213.5 / 69.6%    | 1,380.7 / 68.0%    | 2.15×   |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 143.8 / 48.9%      | 267.6 / 48.1%      | 461.1 / 45.8%      | 766.6 / 46.0%      | 5.33×   |

### 单请求服务

串行服务语料库使用 INT8 group-64 KV、CUDA Graphs、1,024 token 预填充块，以及预热后的五个固定种子。下表为每个已发布配置保留一个短预填充、一个极端预填充和一个结构化输出 MTP3 数据点；完整的上下文和场景矩阵链接自下方各模型。

| 模型配置                                                     | 7,680 token 预填充 | 260,096 token 预填充 | 结构化 MTP3 解码 |
| :----------------------------------------------------------- | :----------------- | :------------------- | :--------------- |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s     | 5,247.0 tok/s        | 779.6 tok/s      |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s      | 1,614.8 tok/s        | 193.0 tok/s      |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s     | 2,510.6 tok/s        | 252.2 tok/s      |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,274.7 tok/s      | 1,609.7 tok/s        | 224.4 tok/s      |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 8,340.4 tok/s      | 2,203.1 tok/s        | 219.8 tok/s      |

## 评估

能力分数通过 NInfer 的 OpenAI 兼容服务路径测量，启用思考、MTP3 和 EvalScope 1.9.0（0-shot、规则评分、每题一个样本）：

| 模型配置                                                     | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA   | RealWorldQA |
| :----------------------------------------------------------- | :-------- | :-------- | :----------- | :----- | :---------- |
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67%    | 93.33%    | 86.87%       | —      | —           |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33%    | 93.33%    | 84.34%       | —      | —           |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00%    | 90.00%    | 85.35%       | —      | —           |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67%    | 96.67%    | 87.37%       | 66.25% | 82.22%      |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67%    | 96.67%    | 90.40%       | 66.25% | 83.53%      |

Qwen3.6 各行使用温度 0.6 和存在惩罚 1.0；Qwen3.8 各行使用温度 1.0 和存在惩罚 0.0。多模态评估使用 `--vision` 和 81,920 token 的上下文限制。文本评估使用 262,144 token，但 Qwen3.8-27B NVFP4 使用 252,928 token，以在权重之后适配 RTX 5090。每个分数为每题一个样本；模型卡包含正确/总数计数和评估说明。

## 启动说明

GPU 驻留固定在进程启动时。`--spec` 选择投机解码驻留，`--vision` 独立选择 Vision 驻留。Qwen3.6-35B-A3B DFlash 可以与 Vision 结合使用；它加速多模态预填充之后生成的文本解码，而非 Vision 编码本身。

## Docker

在装有 NVIDIA Container Toolkit 的主机上构建运行时镜像：

```bash
docker build --tag ninfer:local .
```



挂载已下载的模型并运行相同的示例服务器配置：

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```



## 能力与限制

官方产物提供以下能力，可选组件在启动时启用：

- 支持思考和非思考提示模式的文本生成；
- 图像、多图像、视频以及混合多模态消息；
- 分块预填充、精确批次 CUDA Graph 解码，以及启动时受限的批处理解码；
- MTP 投机解码，草稿窗口为一到五；
- BF16、INT8、FP8、NVFP4 和 K8V4 KV 存储；
- 离线因果困惑度评分；
- 私有和共享精确前缀复用，具有 Device/Host State 和 KV 保留；
- 模型感知的采样默认值和显式采样器覆盖；
- OpenAI Responses Core、OpenAI Chat Completions 和 Anthropic Messages，包括流式传输、工具、本地响应状态、token 计数和用量统计。

35B-A3B 目标还支持 DFlash，草稿窗口为一到十五，适用于文本和图像/视频 Vision 提示。带有 DFlash2 伴随权重的 Qwen3.8-27B 产物支持 `--spec dflash2 --draft-tokens 7`，用于相同的 Text/Vision Engine 路径，草稿计数为 1..15，并支持完整或优化提案头。

产品边界刻意保持很小：

- 每个 Engine 一块 RTX 5090 和一个常驻模型；
- 启动时固定的一到八个活动请求容量，具有有界 FIFO 入口；
- 无请求抢占、优先级/QoS、活动请求交换、权重卸载、多 GPU 或分布式服务；
- 跨活动请求和保留前缀的单个共享启动时固定 KV 池；
- 模型架构和格式/形状组合使用显式实现的原生路径；
- 解析后的工具调用返回给客户端；NInfer 不执行工具；
- 树内 C++ 头文件不作为已安装 SDK 分发。

`--max-context` 是每个序列的逻辑限制。`--kv-capacity` 设置活动请求和保留前缀使用的共享 Main Text KV 池大小；`auto` 会在启动时根据权重之后剩余的内存解析出最大合法容量，同时保留 1 GiB 的 sizing 余量。显式容量在进程生命周期内保持固定。

## 文档

- [文档索引](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP 服务](docs/serving.md)
- [性能](docs/performance.md)
- [困惑度评估](docs/perplexity.md)
- [权重转换与自定义配方](docs/weight-conversion.md)
- [资源调度与上下文缓存](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT 基准测试](tools/bench/ttft/)
- [CLI 示例](examples/cli/)
- [贡献](CONTRIBUTING.md)

运行相关的 `--help` 以了解确切的当前选项契约。

## 支持

NInfer 是我出于兴趣开发的个人项目。如果你觉得它有用，并希望支持其持续开发，可以在 [Ko-fi 上支持该项目](https://ko-fi.com/neroued)。

支持完全自愿。它不是购买或投资，不附带财务回报、承诺的服务或功能，也不赋予项目决策中的角色。项目的方向、优先级、技术选择和发布计划仍由维护者独立决定。

## 许可证

NInfer 根据 [Apache License 2.0](LICENSE) 许可。

已发布的产物派生自
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)、
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 和
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)。Qwen3.6-27B NVFP4 产物
还使用了来自
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm) 的固定打包权重。
Qwen3.8-27B NVFP4 产物还使用了来自
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) 的固定混合 FP8/NVFP4 权重。这些源仓库根据 Apache-2.0 分发。vendored 依赖保留其各自的许可证文件于 `third_party/` 下。