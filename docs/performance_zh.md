# 单 GPU 服务性能

已发布的测量结果使用一块 NVIDIA GeForce RTX 5090，通过 NInfer 的公开 HTTP 服务路由进行。请在下方选择一个模型，查看其详细结果、运行条件、输出限制和复现命令。这些是记录下来的历史测量结果；一个模型/后端受到支持，并不意味着每个工作负载或并发水平都有已发布的测量结果。

请阅读[测量与发布规则](performance/methodology.md)，了解工作负载定义、指标公式、统计方法、比较要求以及标准结果页面格式。

## 已发布覆盖范围

每个单元格链接到相关结果部分。“未发布”描述的是测量覆盖范围，而非产品支持情况。C 表示配置的请求并发数；K 表示草稿 token 的数量。

| 模型 / 权重                       | MTP0 上下文配置                                              | 单请求投机解码                                               | 语料库完工时间                                               | MTP3 解码饱和                                                |
| :-------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Qwen3.6-27B / `groupwise-int`     | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | 未发布                                                       | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-27B / `nvfp4`             | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | 未发布                                                       | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-35B-A3B / `groupwise-int` | [8K–256K](performance/qwen3.6-35b-a3b.md#no-speculation-context-profile) | [MTP3; DFlash K=7 随机/贪婪](performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash C=1](performance/qwen3.6-35b-a3b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.6-35b-a3b.md#decode-saturation) |
| Qwen3.8-27B / `groupwise-int`     | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | 未发布                                                       |
| Qwen3.8-27B / `nvfp4`             | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |

Qwen3.8 和 Qwen3.6-35B-A3B 的 C=1 语料库数据点也提供了各自的单请求阶段表。Qwen3.6-27B NVFP4 MTP3 阶段表来自一个语料库 C=1 数据点，其完整完工时间未在此处发布。Qwen3.8 NVFP4 饱和报告保留了配置和数值，但没有测试过的 Git 修订版本；模型页面记录了该来源限制。

## 如何解读结果

| 问题                                       | 应使用的指标                          |
| :----------------------------------------- | :------------------------------------ |
| 提示处理或单个解码阶段有多快？             | 预填充阶段、服务端 TTFT、解码阶段     |
| 完整的固定请求集需要多长时间？             | 语料库完工时间、语料库解码、请求数/秒 |
| 在完整批次下持续达到的聚合解码速率是多少？ | 稳态解码                              |

这些速率使用不同的时间边界。服务端 TTFT 是内部阶段之和；外部流式 TTFT 有其[自己的基准测试契约](../tools/bench/ttft/README.md)。即使提示和种子相同，随机运行也可能生成不同的 token 总数。输出限制和重复样本在测量语料库中仍带有标记；仅凭吞吐量并不能证明任务成功完成。参见 [35B 终止与异常](performance/qwen3.6-35b-a3b.md#termination-and-anomalies)以及 [Qwen3.8 DFlash2 结果](performance/qwen3.8-27b.md#dflash2-completion-outcomes)。

## 相关参考

- [服务基准测试运行器](../tools/bench/README.md#serving-corpus-benchmark)：用法和本地报告文件。
- [引擎与算子基准测试](../bench/README.md)：它们各自的测量范围和命令。
- [能力评估](../eval/README.md)：评估工作流；已发布分数位于[模型卡](readme.md/#model-artifacts)，并附有 [README 摘要](../README.md#evaluation)。
- [困惑度](perplexity.md/)：离线因果评分测量与比较规则。

模型页面是详细结果的权威来源。README 和模型卡性能表是指向这些页面的摘录；在替换适用的测量结果时，应一并更新它们。