# 原生深度恢复审计

审计日期：2026-09-17。状态区分：**权重名称可用**、**组件数值一致**、**完整原生运行已验证**。本次只确认前两项。

## 固定资源

| 资源 | 版本 |
|---|---|
| `Wakals/CoVT-7B-depth` | `48f5a921c01db42b33ca0949b303b17a3e085bf4` |
| `Wakals/CoVT` 代码 | `c3497a325b0fdda4d91ade456a01b0e325ee3150` |
| `train/src/training/covt_qwen2_5_vl.py` Git blob | `a0b774fdecdea00fd79f026d20450585cd46d825` |
| 上述文件 SHA256 | `8764990efc31c283c99121586e724fe0dcab24ae643c37297f6fd0dc64ffbc44` |

源码通过 GitHub Contents API 取得，并校验 Git blob SHA1。模型元数据和 safetensors index 从固定 revision 获取。初次联网审计部分请求超时，随后从相同固定版本补全缓存；原始错误保留在 `runs/resources-01/report.json`。

## 实际恢复链

```text
生成的 <|depth_pad|> 输入位置
  -> 最终语言 norm 的输出 [4,3584]
  -> depth_token_generator:
       Linear(3584,3584) -> GELU -> Linear(3584,1024)
  -> 四个 token 分别与四层专家 patch features 作内积
  -> 每层 bilinear 上采样（align_corners=False）
  -> 四层平均，得到原始图
```

四个必要参数均位于 `model-00004-of-00004.safetensors`：

```text
depth_token_generator.0.weight
depth_token_generator.0.bias
depth_token_generator.2.weight
depth_token_generator.2.bias
```

`depth_projection`、`depth_query_vectors`、`depth_cross_attention` 也在索引中，但属于代码中的早期 feature-alignment 分支；后期重建分支调用 `depth_token_generator`。恢复时不能混用这两条路径。

论文公式有 softmax，固定代码的 `DepthReconstructor` 无 softmax；本实现遵循代码，并提供直接从上游 AST 提取该类的数值对照。当前测试的最大绝对误差为 **0.0**，仅覆盖重建算子，不覆盖真实 checkpoint 全链路。

深度专家为 Depth Anything V2 Large/vitl，特征层 `[4,11,17,23]`。CoVT 将 PIL RGB 图像缩放为 256×256，然后直接传入专家的 `image2tensor`。本实现保留该调用约定，后续应单独检查通道约定的影响。

## 尚待 GPU 验证

1. 四个实际张量的恢复/加载及模型版本匹配。
2. 完整专家权重与特征预处理。
3. 原生模型是否稳定生成四个 depth pad，是否正常终止。
4. 缓存与无缓存推理、完整序列 identity replay 的数值容差。
5. 合成及真实图像上，解码器的深度关系是否可靠。
6. **共享因果祖先尚未验证**；末端 norm identity 不提供答案路径证据。

本机有 RTX 2000 Ada Laptop 8GB，安装的 Transformers 为 4.57.6，且缺少 accelerate。项目为完整模型运行指定独立环境中的 Transformers 4.50.1；没有修改全局环境或下载完整 7B 模型。

此次在线选择性权重恢复在网络请求阶段长时间未返回，已中止，没有生成可用的 `decoder.safetensors`。这不改变已确认的索引结果，也不代表参数已恢复。已提供从现有 Hugging Face snapshot 本地提取四个参数的替代入口。

可复核链接：[固定源码](https://github.com/Wakals/CoVT/blob/c3497a325b0fdda4d91ade456a01b0e325ee3150/train/src/training/covt_qwen2_5_vl.py)、[固定权重索引](https://huggingface.co/Wakals/CoVT-7B-depth/blob/48f5a921c01db42b33ca0949b303b17a3e085bf4/model.safetensors.index.json)。
