# CoVT native depth pilot

第一篇 proposal 的原生深度解码恢复与 E0/E1 实验仓库。核心目标：确认官方深度读出能否恢复，然后测量 thought 状态与专家特征各自对深度关系的影响。

**范围**：资源审计、原生解码分支、确定性重放、深度标定、条件解码四组合实验，以及探索性的中间层状态干预。尚未完成答案路径定位或修复训练。E1 只能说明可视化分支的依赖，不能证明答案使用了该信息。

## 当前验证状态

- 已确认：公开 checkpoint 索引包含恢复所需的四个 `depth_token_generator` 参数。
- 已确认：本仓库重建算子与固定版本官方 `DepthReconstructor` 在随机张量、非方形网格和不同 batch 下数值一致；此次最大绝对误差为 **0.0**。
- 已完成：CPU 几何数据/标定 smoke test；软件测试覆盖错误权重拒绝、Range 下载约束、token 位置、无缓存重放等。
- **已完成首轮 Colab A100 实测**：12 张图像均得到原生深度状态，10 张通过全部 E0 检查；合格 pilot 样本深度关系 7/7、语言答案 2/7。两个样本出现缓存模式导致的答案差异，原因未查明。E1 仅一个有效场景族。详见 [首轮结果与限制](colab_20260918_results.md)。
- **已完成 FP32 后续验证**：40/40 通过全部 E0 检查；32 张 pilot 图像深度关系 32/32、语言答案 8/32。固定/跨图像状态保持 100% 深度关系正确，位置乱序降至 58.125%，置零全部弃权；E1 扩大至 8 个场景族。详见 [后续结果与限制](colab_20260918_followup.md)。
- **已完成答案诊断**：CoVT 与 Qwen 基座各 128 次生成。简单标签与大小识别均正确；直白近距离问法均为 11/16，但标签交换一致性仅 3/8 个族。见 [答案诊断](answer_diagnostic_20260918.md)。
- **已完成 200 图冻结读出扩展**：固定旧状态为 199/200 正确、1 次弃权；40 图原生对照中 39 张状态合格，语言答案 22/40 正确。见 [扩展结果](shift_validation_20260918.md)。
- **已完成大小平衡对照**：32 张图中语言答案 25/32 正确；大小冲突条件为 13/16。中间层相反深度替换在可配对的 20 张图上未改变答案。见 [平衡对照与干预结果](balanced_intervention_20260918.md)。
- **已完成七层扫描**：相反深度替换在各层均为 0/20 答案翻转，最大条件概率变化 0.00710 个百分点；置零在若干层破坏输出结构。140/140 原样写回和 20/20 答案读出正控制通过。见 [扫描结果与格式审计](layer_scan_20260918.md)。
- **已完成连续层、全层与位置扩展**：自然供体替换扩大至 28 层、15 个思考前缀位置后仍无答案翻转，最大条件概率变化为 0.0472 个百分点；同深度控制为 0.0422 个百分点。200 次干预均保持原生成 token 序列，100/100 原样写回通过。见 [多层与位置扩展结果](block_intervention_20260918.md)。

实际审计结果及限制见 [docs/restoration_audit.md](restoration_audit.md)。`runs/` 保存下载的远程运行记录，默认不提交。几何 oracle smoke 输出不能当成模型准确率。

## 远程环境验证（无需下载模型）

以下运行命令仅用于 Colab 或其他远程机器。本机不运行实验。建议建立独立 Python 3.11 环境，不修改现有科研环境：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -e '.[test]'
python -m pytest -q
covt-pilot doctor --out runs/doctor.json
covt-pilot smoke --out runs/smoke
covt-pilot make-scenes --families 20 --seed 0 --out data/pilot
```

未安装为包时，设置 `PYTHONPATH=src`，使用 `python -m covt_pilot`。所有实验输出目录必须为空，避免覆盖已有结果。

场景是 CPU 解析光线求交生成的球体与地面，提供 camera-axis depth、可见实例、标记位置和场景参数。每个 family 包含 base、depth swap、仅颜色变化、A/B 标签交换四个变体；所有变体处于同一 split。前 20% family 用于 calibration，其余为 pilot。该生成器用于验证仪器和发现信号，不能替代后续丰富场景或真实图像验证。

## 恢复官方解码器

```bash
python -m pip install -e '.[model,test]'
covt-pilot audit --out runs/resources \
  --revision 48f5a921c01db42b33ca0949b303b17a3e085bf4 \
  --code-revision c3497a325b0fdda4d91ade456a01b0e325ee3150
covt-pilot verify-reference \
  --source runs/resources/covt_qwen2_5_vl.py --out runs/reference.json
covt-pilot recover-decoder --audit runs/resources/report.json --out runs/decoder
```

以上多行命令采用 Bash 的 `\` 续行；PowerShell 请合并为单行。`audit` 只下载元数据和源码。`recover-decoder` 使用 HTTP byte ranges 提取四个张量，不下载完整 VLM 分片；服务器不支持 Range 时明确停止。不会随机初始化缺失权重，也不会训练替代探针。

若已有完整 checkpoint，可在 `recover-decoder` 后增加 `--local-checkpoint /path/to/snapshots/48f5a921c01db42b33ca0949b303b17a3e085bf4`，直接读取本地 safetensors 中的四个参数。此入口无需网络，也不把完整模型载入内存。

网络中断后，可以保留已下载的元数据，用 `audit-cache --help` 查看离线完成审计的入口。必须保留固定 revision 的文件及 GitHub Contents API 响应，不应混合不同版本。

## 准备专家与模型环境

实验仅在远程环境运行，本机只编辑代码、准备上传文件和整理已有结果。7B 模型权重、专家、缓存和工作区都需要内存；`doctor` 只检查环境，不保证模型可装入。首轮已在 Colab A100 40GB 上运行；`scripts/colab_e0.py` 提供 Colab 专用的小规模执行入口，需先在远程独立 Python 3.11 环境中安装本仓库的 `[model,test]` 依赖。

保留官方 DepthAnything 实现，检出审计中的固定代码提交：

```bash
git clone https://github.com/Wakals/CoVT.git third_party/CoVT
git -C third_party/CoVT checkout c3497a325b0fdda4d91ade456a01b0e325ee3150
```

从官方 [Depth Anything V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) 下载 `depth_anything_v2_vitl.pth`，放到 `checkpoints/depth/`。可先解析不可变版本再下载：

```python
from huggingface_hub import HfApi, hf_hub_download
repo = "depth-anything/Depth-Anything-V2-Large"
revision = HfApi().model_info(repo).sha
print("Expert revision:", revision)
hf_hub_download(repo, "depth_anything_v2_vitl.pth", revision=revision,
                local_dir="checkpoints/depth")
```

专家采用 Large/vitl、层 `[4,11,17,23]`。图像预处理保持 CoVT 原始调用：PIL RGB 图像先缩放到 256×256，再直接传入其 `image2tensor`；不擅自修正潜在的颜色通道约定。专家权重 SHA256 和代码版本记录在每次 E0 的 `run.json` 中。

## E0：原生生成与恢复检查

```bash
covt-pilot e0 \
  --manifest data/pilot/manifest.jsonl \
  --audit runs/resources/report.json \
  --decoder-dir runs/decoder \
  --covt-repo third_party/CoVT \
  --expert-checkpoint checkpoints/depth/depth_anything_v2_vitl.pth \
  --device cuda --expert-device cpu --dtype float32 \
  --out runs/e0
```

FP32 是当前 Colab A100 pilot 的明确配置；CLI 的历史默认值仍为 BF16，因此复现时不要省略 `--dtype float32`。四图定向诊断中，FP32 消除了缓存模式的 token 差异，单独提升输出头精度未消除 BF16 的答案翻转；详见 [精度诊断](cache_precision_diagnostic.md)。FP32 增加显存需求，不能据此保证任意输入均无差异。

第一次 E0 会通过 Hugging Face 加载固定版本完整 VLM，可能下载十几 GB 权重。它执行：

1. checkpoint 原生 chat template 和 greedy generation；保留完整 token IDs、prompt、image grid、文本及无效答案。
2. cached/uncached 两次生成比较；未正常终止的输出不能通过检查。
3. 在生成的四个 `<|depth_pad|>` **输入位置**提取末层归一化状态，不做 `-1` 位移。零个、多个或不完整的 depth span 明确标记不合格。
4. 无缓存完整序列重放与 identity replacement，核对 hidden states 与 logits。此 identity 位点属于可视化末端，不能据此声称验证了共享因果祖先。
5. 专家重算、解码器 identity、一致的原始深度图输出；保存 `.npz` 状态和特征以供 E1 使用。
6. 仅用 calibration family 确定深度方向和尺度；在 pilot 上报告专家/解码器/答案联合错误和按 family 重采样的区间。

`--atol`、`--rtol` 为数值重放容差，默认值是待验证的工程起点。深度弃权阈值固定为校准尺度的 0.05，是 pilot 设定，不能称为确认性研究中已论证的等价边界。没有任何阶段自动将“未显著变化”判为等价。

主要输出：`run.json`（环境和资源）、`results.jsonl`（全部尝试与失败）、`calibration.json`、`scored.jsonl`、`summary.json`、每个合格样本的原始图/状态/专家特征。统计报告注明对 E0 合格样本的条件化，失败数保留在分母说明中。

## E1：条件解码的四组合实验

```bash
covt-pilot e1 --e0-dir runs/e0 --decoder-dir runs/decoder --out runs/e1
```

对同一 pilot family 的 base/depth-swap 与 base/nuisance 配对，分别计算 `D(Ha,Fa)`、`D(Hb,Fa)`、`D(Ha,Fb)`、`D(Hb,Fb)`。固定原标定，保存四幅原始图、状态效应、专家效应和交互项。深度交换效应按 donor 的深度方向定向，避免平衡 A/B 标签后正负效应相互抵消；nuisance 报告绝对变化。

此实验替换的是末端解码输入。语言答案不在 E1 中重新计算。共享上游干预及所有受影响后继的重新执行属于后续 E2，实现前不得据 E1 得出“thought 未被答案使用”的结论。

## 继续条件

新增 Colab 入口：`scripts/colab_followup.py` 固定 seed=18、40 张图像，执行 FP32 E0 和末端状态对照。`scripts/state_controls.py` 比较原生、校准均值/示例、零 hidden、零投影 token、token 乱序及跨族状态；保持 E0 校准不变，分别使用原生与 FP32 解码器。零 hidden 仍经过 MLP 偏置；零投影 token 才是平坦图负对照。五次随机对照先在场景族内平均，不增加独立样本数。

- 解码链恢复失败：检查源码、四个权重、专家预处理和 token 位置，不训练新探针冒充原生读出。
- 解码关系不可靠：先处理恢复/分布问题，暂不做大规模 reader 搜索。
- 图像专家固定后，状态替换具有可重复的语义影响：再实现 E2/E3。
- 只有可视化解耦，尚未联系自然错误：保留有限结论，不启动修复训练。

官方资料：[CoVT](https://github.com/Wakals/CoVT)、[训练说明](https://github.com/Wakals/CoVT/blob/main/docs/Train.md)、[论文](https://arxiv.org/abs/2511.19418v3)。第三方源码与权重沿用各自许可，仓库不重新分发完整权重。
