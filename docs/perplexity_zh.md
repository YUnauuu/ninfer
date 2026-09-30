# Perplexity 评估

`ninfer-perplexity` 用于测量由 v3 `.ninfer` 产物生成因果困惑度（causal perplexity）。
它使用该产物的分词器、Text 模型、选定的 Main KV 表示、最终归一化以及主输出头。
它是一个离线评估器，不是服务端点，也不是 logits 导出 API。
仅加载 Text 权重和资源；不需要 Vision 和推测组件。

## 运行固定语料库

仓库包含 `ninfer-ppl-1m-v1`，这是一组固定的 16 个独立 UTF-8 流，涵盖
英文参考文本、英文长文本、中文参考文本以及 NInfer C++/CUDA 代码。
`full` 选择所有流；`--quick` 从每个领域选择一个流。



```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype fp8
```



默认评估使用 4,096 token 的上下文和 2,048 token 的步幅。使用 `--context` 和
`--stride` 更改该协议，或使用 `--text FILE` 对单个 UTF-8 文件评分。可用的 Main
KV 表示为 `bf16`、`int8`、`fp8`、`nvfp4` 和 `k8v4`。



```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```



运行 `./build/apps/ninfer-perplexity --help` 查看完整命令界面。评估器会加载
模型一次，在评分前读取并分词每个选定的流，并将可读的启动、语料库、评分和
每个流的摘要写入 stderr。交互式权重加载和评分使用一条临时进度行；重定向的
评分每十秒发出一次持久进度。`--log-level debug` 会暴露内部启动和流开始细节。
最终的领域/总体表格仍作为产品输出写入 stdout；独立的完整精度机器报告为
`profiles/perplexity/` 下的 `report.json`，除非 `--output` 提供了一个空目录。

对于 KV 格式比较，推荐的长上下文配置是使用完整语料库，并设置
`--context 65536 --stride 32768`，且不使用 `--quick`。

## 指标

对于流 `x[0..N)`，`x[0]` 之后的每个 token 都会被恰好评分一次。一个窗口 `[b,e)`，其目标
后缀为 `[s,e)`，贡献如下：



```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```



每个窗口都从空 State 和 Main KV 开始，因此 `b` 之前的历史被有意排除。
因此，所报告的指标是固定窗口、截断上下文的因果困惑度：



```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```



第一个窗口对 `[1,min(context,N))` 评分。此后的每个窗口按 `stride` 个目标前进，同时
保留最多 `context-stride` 个前序 token 作为局部上下文。流之间从不共享历史。

## 比较运行结果

进行数值比较时，除被测量的变量外，保持语料库、上下文、步幅和执行设置固定不变。
在比较 KV 格式时使用相同的产物；在比较权重格式时使用相同的 KV 格式。

语料库名称是工作负载规模，而不是精确的 token 数。精确的输入和已评分 token 数
是来自当前产物分词器的运行时结果，并记录在每个报告中。报告包含每个窗口、流、
领域以及按 token 加权的总体聚合的未舍入 NLL/PPL 值。

schema-v2 报告会标明产物的架构、公开名称、实际权重格式和预填充签名，以及
工作负载和数值结果。