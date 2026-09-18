# 2026-09-18 实验归档索引

所有相对路径均以 `covt-pilot/` 为根。`runs/` 中的大文件由 `.gitignore` 排除，保存在当前工作区；`docs/` 报告与汇总元数据可纳入版本控制。这里只保存下载与整理结果，模型实验在 Colab 上执行。

## 首轮 BF16 E0/E1

- 目录：`runs/colab-20260918-backup/`。
- `INDEX.json`：校验值与 E0/E1 汇总。
- `covt-e0-e1-results-20260918-032737.zip`：完整原始归档，293227867 字节，SHA256 已核验。
- `runs/colab-e0-20260918-032255/`：完整解压内容，含原生解码器、特征缓存、地图、场景、结果与日志。
- 历史已执行笔记本：`runs/CoVT_Native_Depth_E0_A100.ipynb`。
- 报告：[首轮结果](colab_20260918_results.md)。

## 精度诊断、FP32 E0 和状态对照

- 目录：`runs/colab-20260918-followup/`。
- `INDEX.json`：精简和完整归档的预期 SHA256、实际核验状态及汇总；以 `full_verified` 判断完整归档是否已落盘核验。
- `covt-followup-reports-20260918.zip`：精简归档，1237582 字节，已核验并解压。
- `covt-followup-full-20260918.zip`：完整缓存归档，预期 1026766241 字节。
- 解压的 `runs/cache-diagnostic-*/`：四种精度、四张诊断图像的逐步 token/logits 记录。
- 解压的 `runs/followup-fp32-20260918-043002/`：固定实验计划、40 张图像、E0、16 个 E1 配对、七类状态对照。
- 解压的 `scripts/`、`src/`、`tests/`：该次运行源码快照；当前维护入口位于 repo 对应目录。
- `CoVT_Native_Depth_Followup_A100.executed.ipynb`：本轮已执行笔记本，包含原始输出和归档校验值。
- 报告：[缓存精度诊断](cache_precision_diagnostic.md)、[后续实验](colab_20260918_followup.md)。

原始 `run.json` 中的绝对路径反映 Colab 执行环境，保留不改写。重用缓存时应在远程机器提供实际的 `--e0-dir` 和首轮恢复的 `--decoder-dir`。复现完整生成还需下载报告中固定版本的模型与专家 checkpoint；本归档不重复分发完整模型权重。
