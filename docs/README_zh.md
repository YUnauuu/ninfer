# NInfer 文档

请先阅读[项目 README](../README.md)，以构建 NInfer、下载已发布的制品，并运行 CLI 或 HTTP 服务器。

## 用户指南

| 文档                                      | 用途                                                         |
| :---------------------------------------- | :----------------------------------------------------------- |
| [CLI](cli.md/)                    | 文本、聊天历史、图像/视频输入、输出流、采样、MTP 以及常见运行时选项 |
| [HTTP 服务](serving.md/)          | OpenAI Responses/Chat Completions、Anthropic Messages、状态、流式传输、token 计数、身份验证和工具调用 |
| [性能](performance.md/)           | RTX 5090 测量覆盖范围、各模型服务结果、方法论和发布规则      |
| [权重转换](weight-conversion.md/) | 官方配方、自定义格式和来源、转换方法、可选组件和制品输出     |
| [困惑度](perplexity.md/)          | 固定语料库和自定义文本的因果困惑度、比较规则、进度和报告     |
| [CLI 示例](../examples/cli/)      | 已提交的文本、多模态、思考、长解码和长上下文输入             |

可执行文件的 `--help` 输出是命令行选项拼写和默认值的准确来源。

## 模型制品

| 模型            | 权重            | 下载                                                         | 带版本的模型卡源文件                                         |
| :-------------- | :-------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Qwen3.6-27B     | `groupwise-int` | [Hugging Face](huggingface.co/neroued/Qwen3.6-27B-NInfer) | [模型卡](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B     | `nvfp4`         | [Hugging Face](huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [模型卡](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B     | `groupwise-int` | [Hugging Face](huggingface.co/neroued/Qwen3.8-27B-NInfer) | [模型卡](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B     | `nvfp4`         | [Hugging Face](huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [模型卡](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [模型卡](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

## 仓库本地指南

- [基准测试](../bench/README.md)
- [测试](../tests/README.md)
- [工具](../tools/README.md)
- [能力评估](../eval/README.md)

## 维护者参考

[`maintainer/`](maintainer/) 下的现行参考文档记录了当前的架构、模型、制品和维护契约。这些文件不是额外的用户工作流程或已安装的 API 文档。

[引擎架构](maintainer/engine-architecture.md)是唯一的顶层参考。其他参考文档各自负责更窄的契约：

| 文档                                                         | 职责                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [代码结构](code-structure.md/)                       | 仓库导览图：目录归属、模块走读、构建目标图和阅读顺序         |
| [引擎架构](maintainer/engine-architecture.md)        | 模型/配置/权重归属、从加载到执行的流程、请求、调度、事务和图 |
| [构建系统](maintainer/build-system.md)               | CMake 目标、显式源文件归属、CUDA 编译边界、预设和开发者配置  |
| [制品容器](maintainer/artifact-container.md)         | v3 目录、对象、逻辑绑定、Uses、资源以及文件分帧/分片         |
| [数值格式](maintainer/tensor-formats.md)             | 所表示的值、编码/缩放、转换算术和数值解释                    |
| [存储布局](maintainer/storage-layouts.md)            | 打包、平面偏移、填充、编码大小和视图寻址                     |
| [Qwen3.5 模型](maintainer/qwen3_5-model.md)          | 稠密/MoE 数学、实例配置、逻辑参数、MTP、视觉和状态语义       |
| [DFlash 和 DFlash2](maintainer/dflash.md)            | 条件化、掩码草稿计算、提议分布和后端状态                     |
| [资源调度和上下文缓存](maintainer/resource-scheduling-and-context-cache.md) | 候选选择、保留、物化以及设备/主机检查点策略                  |
| [分页 KV 上下文存储](maintainer/paged-kv-cache.md)   | 类型化池、页、副本、地址空间、预留和使用者视图               |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md)         | 原始转移记录和已验证状态前缀的忠实提交                       |
| [算子开发](maintainer/op-development.md)             | 语义边界、源文件归属、数值资格验证和性能证据                 |
| [运行日志](maintainer/logging.md)                    | 日志归属、呈现、严重级别和数据策略                           |
| [线性基准测试](maintainer/linear-benchmark.md)       | 纯 Linear 测量、指标和测试套件                               |
| [线性调优和报告](maintainer/linear-tuning.md)        | 调优范围、优先点、分派权衡和最终性能报告格式                 |

模型卡包含官方制品事实和来源出处。[转换指南](weight-conversion.md/)是制作制品的入口。确切的配置字段、参数展开和原生支持域由这些参考文档所链接的代码维护。