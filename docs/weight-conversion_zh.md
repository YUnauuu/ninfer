# 权重转换

NInfer 的转换器根据本地权重和 Python 配方（recipe）创建 `.ninfer` 产物。配方可以复用官方转换、更改选定的层或投影、组合多个来源，或调用你自己的转换方法。产物包含生成的配置、编码后的权重、逻辑绑定以及前端资源。

请在仓库根目录下运行以下命令。

## 升级现有的 v2 产物

离线升级工具支持官方的 Qwen3.6/3.8-27B groupwise-int 和 NVFP4 产物，以及 Qwen3.6-35B-A3B groupwise-int。将你的检出版本更新到当前的 `master` 并[重新构建 NInfer](../README.md#quick-start)，然后使用 Python 3.11 运行：

```bash
python3 tools/upgrade_ninfer_v2_to_v3.py \
  models/qwen3_8_27b_nvfp4.ninfer \
  models/qwen3_8_27b_nvfp4.v3.ninfer
```



输出必须使用新路径。升级后，可以直接使用它，或将其重命名以替换原文件。已存储的权重值和格式会被保留。升级还会安装来自 `tools/chat_templates/` 的匹配模板。发布的 SHA-256 校验和仅适用于下载的文件。

## 从官方配方开始

源权重转换需要带有 PyTorch 和 NumPy 的 Python 3.11 环境。默认使用 CUDA；`--device cpu` 选择 CPU 转换。下面的输入路径是你的本地检查点目录的占位符。

对于 Qwen3.6-27B 浮点源权重：

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --proposal \
  --name qwen3.6-27b \
  --out models/qwen3_6_27b.ninfer
```



`--components` 默认为 `text`。只包含你想要分发的可选组件。`--proposal` 添加由投机解码使用的索引提案头（indexed proposal head）；它使用仓库的 token 排序，默认为 131,072 行。普通的全词表输出头会被保留。

内置配方是 [`official_recipes.py`](../tools/convert/official_recipes.py) 中的普通 Python 函数：

| 配方                | 主要表示选择                                | 额外来源    |
| :------------------ | :------------------------------------------ | :---------- |
| `qwen3_6_27b`       | Q4/Q5 投影，Q6 词表权重                     | 无          |
| `qwen3_8_27b`       | Q4/Q5 投影，Q8 词表权重                     | 无          |
| `qwen3_6_35b_a3b`   | Q4 专家，Q5/Q6 专家下投影，Q8 共享/投影权重 | 无          |
| `qwen3_6_27b_nvfp4` | 导入的 NVFP4，选定的 BF16 投影，Q8 词表权重 | `quantized` |
| `qwen3_8_27b_nvfp4` | 导入的 NVFP4/FP8，从 BF16 生成的 FP8 嵌入   | `quantized` |

这些名称选择转换选项。运行时执行根据架构、配置和产物中存储的实际绑定来选择。`--name` 设置公开的模型名称；它不选择内核。

对于带 DFlash2 的 Qwen3.8-27B NVFP4/FP8 产物：

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_nvfp4 \
  --source quantized=/path/to/Qwen3.8-27B-NVFP4 \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b \
  --out models/qwen3_8_27b_nvfp4.ninfer
```



MTP 和 Vision 使用主来源。DFlash 和 DFlash2 使用对应的命名来源，以 `--source dflash=PATH` 或 `--source dflash2=PATH` 提供。一个产物可以包含多个可选组件；Engine 在启动时只加载选定的那些，最多包含一个投机后端。组件的可用性和启动选择是相互独立的。

## 更改配方的一部分

将以下内容保存为 `my_recipe.py`：

```python
from tools.convert.official_recipes import qwen3_6_27b


def configure(model, recipe, sources):
    qwen3_6_27b(model, recipe, sources)
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method="grouped_absmax",
    )
```



使用相同的来源和组件选择运行它：

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe my_recipe.py \
  --components text,vision,mtp \
  --proposal \
  --out models/my_qwen.ninfer
```



默认入口函数是 `configure`；`--recipe my_recipe.py:customize` 选择另一个函数。或者，使用官方 `--recipe`，并只将更改放入 `--override` 文件。覆盖在基础配方和可选的提案头设置之后运行。

`model.parameters` 将逻辑名称映射到它们的形状、来源和数学输入。要查看所选组件的可用名称，配方可以打印它们：

```python
for name, parameter in model.parameters.items():
    print(name, parameter.shape, parameter.inputs)
```



`recipe.assign` 接受一个名称、一个名称列表，或 shell 风格的模式，如 `text/layers/*/mlp/down`。匹配不到任何内容的 selector 会失败。赋值按 Python 顺序运行；后面的赋值只替换显式提供的选择。更改 `format` 不会自动更改 `method`。使用 `layout="auto"` 在覆盖先前的显式布局选择时，为格式选择已注册的布局。

主要选择如下：

| 参数                | 含义                                   |
| :------------------ | :------------------------------------- |
| `format`            | 持久化的数值格式                       |
| `layout`            | 物理编码；除非显式设置，否则从格式推断 |
| `method`            | 内置方法名称或 Python 可调用对象       |
| `source`            | 来自所选来源的逻辑值或编码行           |
| `parameters`        | 传递给方法的 JSON 可序列化数值参数     |
| `rows=(begin, end)` | 在半开区间内覆盖完整的前导轴行         |
| `activation_policy` | 对参数数学输入的激活精度的许可         |

例如，`rows=(0, 128)` 可以给前 128 行一个不同的格式。必要时这会创建多个物理部分。容器可以表示该结果；预期的 Op 还必须支持消费这些部分。当前的原生投影通常要求连续的父区域，因此一个投影的任意拆分不能自动执行。

## 格式、方法和激活精度

转换器当前写入以下格式：

| 格式                                                       | 浮点输入的内置方法 | 导入已编码输入              |
| :--------------------------------------------------------- | :----------------- | :-------------------------- |
| `bf16`、`fp32`、`int32`                                    | `cast_direct`      | 通过来源读取器直接读取字    |
| `q4_g64_fp16`、`q5_g64_fp16`、`q6_g64_fp16`、`q8_g32_fp16` | `grouped_absmax`   | 如需要，提供自定义方法/来源 |
| `fp8_e4m3fn_row_bf16`                                      | `fp8_row_maxabs`   | `import_encoded`            |
| `nvfp4`                                                    | 提供自定义量化器   | `import_encoded`            |

`grouped_absmax` 每组存储一个 FP16 缩放因子和有符号整数编码。`fp8_row_maxabs` 首先将输入值舍入到 BF16，然后生成 E4M3FN 编码和每行一个 BF16 乘数。`import_encoded` 保留兼容的编码和缩放字，包括 NVFP4 的矩阵权重除数。它不会对其进行反量化和重新量化。

确切的数值和打包规则见[数值格式](maintainer/tensor-formats.md)和[存储布局](maintainer/storage-layouts.md)。仅来源格式名称不能确立兼容性：缩放方向、粒度、编码含义和轴顺序也必须匹配。

激活许可独立于存储的权重格式：

| 策略      | 允许的激活路径 |
| :-------- | :------------- |
| `A16Only` | A16            |
| `AllowA8` | A16、A8        |
| `AllowA4` | A16、A8、A4    |

它们允许选择；它们不强制内核使用最低精度。在多个投影之间共享一个激活的融合操作必须遵守它们各自许可的交集。`recipe.use(parameter, input_name, ...)` 可以独立设置一个数学输入；名称可在 `parameter.inputs` 中获得。

NVFP4 A4 输入需要正有限激活除数。`import_encoded` 从所选来源获取它，或配方通过 `recipe.use(..., auxiliaries={"activation_input_divisor": value})` 提供它。共享权重保留单独的 Use 记录；共享权重不会隐式共享校准。

## 融合父项和逻辑投影

Qwen 适配器单独暴露 Q、K、gate 和 V，即使它们来自融合的源张量。它还为 attention、GDN、MLP 和 MoE 提供有限的打包组。兼容的选择由内置方法自动打包到共享父项中。

对于 Dense groupwise 配方，attention Q/K 组成一个 Q4 父项，gate/V 组成一个 Q5 父项。原生融合 Op 接收两个权重。要使用受支持的单父项 FP8 形式，覆盖可以一起分配所有四个投影：

```python
def configure(model, recipe, sources):
    for layer, kind in enumerate(model.config["layer_types"]):
        if kind != "full_attention":
            continue
        prefix = f"text/layers/{layer}/attention/"
        recipe.assign(
            [prefix + role for role in ("query", "key", "gate", "value")],
            format="fp8_e4m3fn_row_bf16",
            layout="auto",
            method="fp8_row_maxabs",
            activation_policy="AllowA8",
        )
```



在 `qwen3_6_27b` 或 `qwen3_8_27b` 之后，将此文件用作 `--override`。它将该配方下的其他参数保持不变。适配器处理源 Q/gate 行顺序；覆盖适用于逻辑投影。更改它们的表示可能会改变数值结果和物理内核。

为了显式组织，`recipe.group([names...])` 按给定顺序连接兼容的选择，`recipe.separate(names)` 为那些参数禁用自动分组，`recipe.share(parameter, target)` 将等形状参数绑定到相同的物理数据。显式组必须不相交，并使用未拆分的、格式、布局、方法和方法参数匹配的选择。NVFP4 父项还需要一个共同的权重除数。自动分组仅限于内置方法；自定义方法可以请求显式组。

分组选择存储。模型执行代码选择受支持的融合实现。加载器上传存储的表示，而不会重新打包不方便的排列。

## 读取另一个来源

`--model` 提供主配置、默认资源和名为 `base` 的来源。使用重复的 `--source NAME=PATH` 添加其他 Safetensors 来源；它们在使用时打开。支持单文件 Safetensors 和索引分片。仅含张量的额外来源可以省略模型配置；携带配置的来源会根据相关模型几何进行检查。

要在配方中从另一个兼容检查点替换逻辑参数：

```python
name = "text/layers/0/mlp/down"
recipe.assign(
    name,
    source=model.source(name, sources["alternate"]),
    format="q6_g64_fp16",
    method="grouped_absmax",
)
```



提供 `--source alternate=/path/to/alternate-checkpoint`。`model.source` 应用架构的来源名称和轴映射，包括 Q/gate 提取。内置的 compressed-tensors 读取器理解已实现的逐行 FP8 和 NVFP4 编码/缩放约定。它可以为另一个量化器暴露解码值，或为精确导入暴露编码行。

对于另一种文件格式或量化约定，提供 `LogicalSource`。它的值读取器接受扁平 C 顺序元素边界，并准确返回该范围。例如，配方可以从存储在配方旁的 NumPy 文件读取逻辑矩阵：

```python
from pathlib import Path
import numpy as np
import torch
from tools.convert.sources.logical import LogicalSource


def configure(model, recipe, sources):
    name = "text/layers/0/mlp/down"
    path = Path(__file__).with_name("mlp-down.npy")
    data = np.load(path, mmap_mode="r")
    if tuple(data.shape) != model.parameters[name].shape or not data.flags.c_contiguous:
        raise ValueError("mlp-down.npy must have the logical shape and C-order storage")

    def read_values(begin, end):
        values = data.reshape(-1)[begin:end].astype(np.float32, copy=True)
        return torch.from_numpy(values)

    source = LogicalSource(tuple(data.shape), str(path), read_values)
    recipe.assign(name, source=source, format="q6_g64_fp16", method="grouped_absmax")
```



将此用作覆盖。NumPy 文件必须已经遵循逻辑行/列顺序。对于不熟悉的量化来源，其读取器在返回值之前执行相应的解码。要保留现有的兼容编码字，还要提供返回 `EncodedRows` 的 `read_encoded`，以及格式所需的除数访问器。它们的定义在 [`sources/logical.py`](../tools/convert/sources/logical.py) 中。

## 编写转换方法

方法接收一个 `PrepareRequest` 并返回 `request.job(produce=...)`。准备阶段验证目标并确定辅助值。`produce` 函数读取有界的源区域，并通过 `TensorOutput` 写入值或编码/缩放；写入器负责放置和文件 I/O。

此示例在现有的分组量化器之前添加显式裁剪。它演示了方法接口；裁剪阈值是配方作者做出的数值选择。


```python
import math
import torch
from tools.artifact.formats import QuantFormat, get_format
from tools.convert.quantization.groupwise import quantize_matrix


def clipped_grouped(request):
    if len(request.target.shape) != 2 or not isinstance(
        get_format(request.target.format), QuantFormat
    ):
        raise ValueError("clipped_grouped requires a grouped-integer matrix")
    limit = float(request.parameters["clip"])
    if not math.isfinite(limit) or limit <= 0 or request.rows_per_chunk <= 0:
        raise ValueError("clip and rows_per_chunk must be positive")
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("source contains non-finite values")
            encoded = quantize_matrix(
                values.clamp(-limit, limit),
                request.target.format,
                device=request.device,
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def configure(model, recipe, sources):
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method=clipped_grouped,
        parameters={"clip": 1.0},
    )
```



将此用作覆盖。`request.values` 按父项顺序遍历准备好的逻辑输入，包括显式组。`output.write_codes` 执行已注册的打包并验证编码/缩放；方法不应重复该字节布局逻辑。直接输出使用 `output.write_values`。将源块和临时设备张量限制在方法的工作集内。`--rows-per-chunk` 默认为 512；自定义方法自行决定如何使用它。

## 资源、文件和检查

Text 包含 `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` 和 `generation_config.json`。Vision 添加其图像和视频处理器配置。资源来自 `--model`；`--resource ROLE=PATH` 替换选定的资源：

```text
--resource chat_template.jinja=/path/to/chat_template.jinja
```



官方转换示例选择这些维护的模板：

| 模型              | 模板                                                         | 默认值                                            |
| :---------------- | :----------------------------------------------------------- | :------------------------------------------------ |
| Qwen3.6 Dense/MoE | [qwen3_6.jinja](../tools/chat_templates/qwen3_6.jinja) | 思考开启；省略已结束轮次的推理                    |
| Qwen3.8           | [qwen3_8.jinja](../tools/chat_templates/qwen3_8.jinja) | 思考开启；effort 为 `xhigh`；保留已结束轮次的推理 |

使用你自己的 Jinja 文件来更改产物的默认模板。启动时的 [`--chat-template FILE`](cli.md/#text-input) 覆盖存储的模板。`generation_config.json` 被保留；采样预设仍由架构和显式应用/请求设置决定。

默认最大文件大小为 32,000,000,000 字节，包括帧结构。较小的产物保持为单个文件。较大的产物使用诸如 `models/my_qwen.ninfer` 的条目，外加同一目录中的 `my_qwen.ninfer.part-0001`、`my_qwen.ninfer.part-0002` 等。只将条目路径传递给 NInfer，并保持其所有记录的部分在一起。`--max-file-bytes` 更改限制。

转换会在产物旁边写入 `models/my_qwen.ninfer.conversion.json`，记录来源、方法、格式、组件配置、文件和计时。现有输出文件不会被覆盖。该报告对于复现配方很有用；Engine 读取产物本身。

```bash
python3 -m tools.artifact.inspect models/my_qwen.ninfer --objects --bindings
python3 -m tools.artifact.inspect models/my_qwen.ninfer --json
```



检查读取目录事实而不运行推理。转换拒绝缺失的逻辑覆盖、无效的源几何、不支持的编码和无效的方法输出。实际的 Op 支持由消费者在准备、资源查询、预热或执行期间检查。一个有效的文件在所选组合能够运行之前可能还需要额外的 Op 支持。通过正常的 [CLI](cli.md/) 或[服务](serving.md/)途径演练你打算使用的阶段和可选组件。