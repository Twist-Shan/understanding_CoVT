"""Build an offline HTML report from reviewed report content and saved figures.

Standard library only. Does not import or execute the PDF builder or run models.
"""
import ast
import base64
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/build_research_report.py'
OUT = ROOT / 'output/html'
OUT.mkdir(parents=True, exist_ok=True)
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
figures = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'savefig':
        figures[node.targets[0].id] = ROOT / 'output/pdf/figures' / (ast.literal_eval(node.value.args[0]) + '.png')

sections = []
for node in tree.body:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name):
        continue
    call = node.value
    name = call.func.id
    if name not in {'sec', 'p', 'para', 'formula', 'table', 'figure', 'conclude'}:
        continue
    if name == 'table' and len(sections) == 18:
        audit = json.loads((ROOT/'runs/colab-20260918-layer-scan/extracted/runs/layer-scan-20260918-093053/final_audit.json').read_text(encoding='utf-8'))
        rows = []
        for layer in [0,4,8,14,20,24,27]:
            natural = next(r for r in audit['details'] if r['layer']==layer and r['condition']=='opposite_depth')
            zero = next(r for r in audit['details'] if r['layer']==layer and r['condition']=='zero')
            rows.append([layer,'0/20',f'{natural["max_delta_p_A"]*100:.5f}',zero['invalid']])
        args = [ast.literal_eval(call.args[0]), rows]
    elif name == 'table' and len(sections) == 21:
        records = [json.loads(s) for s in (ROOT/'runs/colab-20260918-block-intervention/extracted/runs/block-intervention-20260918-100708/results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        best = max((r for r in records if r['condition']=='opposite_depth'),key=lambda r:abs(r['delta_p_A']))
        args = [ast.literal_eval(call.args[0]), [['目标答案','A','A'],['P(A | A 或 B)',f'{best["conditional_p_A"]-best["delta_p_A"]:.8f}',f'{best["conditional_p_A"]:.8f}'],['概率变化','基线','−0.047183 个百分点'],['全词表 A−B logit 差变化','基线','−0.00356865']]]
    else:
        args = [figures[a.id] if isinstance(a, ast.Name) and a.id in figures else ast.literal_eval(a) for a in call.args]
    if name == 'sec':
        sections.append({'title': args[0], 'source': args[1] if len(args) > 1 else '', 'blocks': []})
    elif sections:
        sections[-1]['blocks'].append((name, args))
assert len(sections) == 24

# Plain-language explanations. Each tuple is purpose, design, example, result, conclusion.
EXPERIMENTS = {
3: (
 '先弄清原来的深度图是怎样算出来的，并检查我们恢复的代码是否与原实现一致。',
 ['模型生成四个专门用于深度的标记。我们取这四个位置的内部数值，记为 H。每个位置有 3584 个数。', '另一套深度模型单独处理同一张图，得到四层图像特征，记为 F。它提供每个图像位置的空间信息。', '把 H 通过原模型的小型投影网络变成四个 1024 维向量，分别与 F 的对应层做点积；放大四张结果图，再取平均。全部使用公开权重，不训练新参数。', '先用人工构造的数值、不同网格形状和批量大小核对计算，再在 E0 中核对真实模型流程。'],
 '说明性例子：向量固定为 (1,0)，图像位置的特征分别是 (2,5) 和 (1,9)，点积仍会得到不同的分数 2 和 1。因此，即使 H 固定，图像特征 F 也能让结果图随图像改变。',
 '已测计算对照中的最大误差为 0。恢复的深度图数值没有换算成米。',
 '计算实现通过了对照。仅凭生成了合理的深度图，还不能判断语言模型是否用这些状态回答问题。'),
4: (
 '明确什么叫“深度判断正确”，避免只凭热图颜色判断。',
 ['每张合成图的 A、B 标记都对应可见物体表面的相机深度 z；z 较小的一点更近。这个标签来自场景生成记录。', '在每个标记周围取半径 5 到 9 像素的环带，分别计算深度图分数的中位数，减少字母本身对测量的影响。', '用 B 的中位数减去 A 的中位数，再乘校准得到的方向、除以固定尺度，得到关系分数 r。方向和尺度只用校准图确定。', 'r 大于 0.05 判 A 更近，小于 -0.05 判 B 更近；中间区域记为弃权。后续测试不重新调阈值。'],
 '说明性例子：校准方向为 -1、尺度为 100，B 减 A 得到 -80，则 r=0.8，判 A。若差只有 -3，r=0.03，记为弃权。',
 '报告中的深度正确率都依据这一规则计算。热图颜色条表示原始读出分数，关系分数 r 是经过校准的另一个量。',
 '深度标签、原始图分数和关系分数分别有明确含义；弃权需要与选错方向分开报告。'),
5: (
 '统一正确率、概率变化和答案翻转的计算方式，让不同实验可以被准确理解。',
 ['一个场景族包含同一基础场景的几种变体；同族图片有关联。报告同时注明图像数和场景族数。', '逐图正确率把所有图片放在一起算；族平均先算每族结果，再让每族占相同权重。置信区间通过重复抽取场景族计算，重复 2000 次。', '概率指标只比较答案 A 和 B 的相对分数。完整生成的答案另外检查，要求格式可解析、答案没有歧义。', '有效翻转要求干预后得到有效答案，而且与原答案不同。乱码、提前结束或同时写 A/B 都需要单独计数。'],
 '说明性例子：第一族 3 张全对，第二族 1 张全错，逐图正确率是 75%，族平均是 50%。A 的条件概率从 0.8400 变为 0.8405，增加 0.05 个百分点。',
 '后续图表分别展示答案是否改变、概率改变多少，以及输出格式是否有效。三者不混成一个指标。',
 '小样本中的零翻转只能说明这批干预没有改变答案，不能证明在所有情形下效应都为零。'),
6: (
 'E0 检查整个恢复流程能否可靠运行：生成的标记、提取的状态、专家特征和最终深度图是否能对应起来。',
 ['生成 3 个场景族，共 12 张图，随机种子为 0。1 个族的 4 张图用于校准，另外 2 个族的 8 张图用于初步评估。', '语言模型使用 BF16，解码器使用 FP16，深度专家使用 FP32。模型冻结，按最高分 token 逐步生成。', '检查是否自然生成恰好四个深度标记；再检查保存状态后重放、专家重算、解码重算能否复现。', '对同一输入比较使用生成缓存和不使用缓存的输出。通过检查后，才进入相应深度与答案分析。'],
 '真实案例 scene_00002_base：B 的表面深度约 3.4078，A 约 5.3795，所以 B 更近。原生深度图和专家图都判 B，语言答案却为 A。',
 '12 张图中有 2 张的缓存与无缓存答案不一致。最终合格的评估图为 7 张：深度关系 7/7 正确，语言答案 2/7 正确。',
 '恢复流程可以运行，但首轮存在数值一致性问题。深度正确与语言答对也已经出现分离，需要分别追查。'),
7: (
 '检查首轮答案不一致是否与计算精度有关，以免后面的干预效果被数值误差混淆。',
 ['选首轮 2 张失败图和 2 张通过图，固定输入、模型来源和生成方式。', '依次比较 BF16、BF16 主体加 FP32 输出头、FP16、完整 FP32。每种都测缓存生成、重复缓存生成和无缓存生成。', '既比较整段 token，也单独比较 A/B 答案，并记录答案位置上 A 与 B 的分数差。'],
 '真实失败例子：BF16 下，同一答案前缀的 A-B 分数差在缓存模式为 -0.375，无缓存模式为 0。完整 FP32 下两者约为 0.03962 和 0.03967，都偏向 A。',
 'BF16 及仅提高输出头精度时，仍有 2/4 张的答案不一致。FP16 的 4/4 张有 token 差异，但只是标点变化。完整 FP32 的 4/4 张全部一致。',
 '后续改用完整 FP32。在这 4 张图上，问题得到规避；具体是哪个层或算子造成差异，原因未查明。'),
8: (
 '用更多独立场景检查 FP32 流程，并重新比较深度结果与语言答案。',
 ['使用新种子 18，生成 10 个场景族、40 张图。2 个族共 8 张图用于校准，8 个族共 32 张图用于评估。', '语言模型改为完整 FP32，使用 eager 注意力实现；专家在 Colab CPU 上运行。没有训练，也没有放宽 E0 检查。', '先逐图完成一致性检查，再在同一批 32 张评估图上分别记录原生深度、专家深度和语言答案。'],
 '说明性例子：一张图的真值为 B，深度图判 B，文字答 A，就同时计入“深度正确”和“语言错误”，两个结果都保留。',
 '40/40 张通过检查。评估集原生深度和专家深度均为 32/32 正确，语言答案为 8/32；有 24 张深度正确但语言答案错误。',
 '后续比较有了可用样本。这个结果说明两种输出的表现不同，尚不能解释模型为何答错；新旧批次也不能直接用来估计改精度的准确率收益。'),
9: (
 'E1 要区分：深度图的变化主要来自语言模型状态 H，还是来自专家图像特征 F。',
 ['取同一场景的原图和深度交换图，分别保存各自的 H 和 F。', '计算四种组合：原 H+原 F、交换 H+原 F、原 H+交换 F、交换 H+交换 F。每次只改指定来源。', '交换 F 时，使用 F 所属图片的 A/B 采样位置。校准规则保持不变。这里只重新计算深度图，不重新生成语言答案。'],
 '真实首轮例子：原组合 r 约为 -1.1709，判 B；只换 H 后仍约为 -1.1737；只换 F 后变为 +1.0730，判 A。',
 '首轮只有一个合格场景族可用于该展示。换 F 改变了深度方向，换 H 没有改变方向。',
 '单个案例提示 F 的影响较大，需要在更多场景族上检查。该实验只涉及深度读出分支。'),
10: (
 '检查 E1 的现象能否在多个场景中重复，并区分远近变化与颜色变化。',
 ['使用 FP32 E0 的 8 个评估场景族。每族取原图与深度交换图、原图与颜色变化图，共 16 对。', '每对都计算四种 H/F 组合，再用关系分数差表示换 H 或换 F 的影响。', '按场景族展示绝对变化量；图的纵轴用对数刻度，因为两种变化量相差较大。'],
 '真实 scene_00002 深度交换例子：只换 H 的关系分数变化约为 -0.000620，只换 F 的变化约为 +2.09677。',
 '深度交换时，换 H 的平均绝对变化约 0.000311，换 F 约 1.802808；关系翻转分别为 0/8 和 8/8。颜色变化时，两者均为 0/8 翻转。',
 '在这 8 个场景族的末端深度读出中，换 F 的影响明显更大。还没有证据说明自然 H 与固定 H 在所有输入上都等效。'),
11: (
 '进一步检查每张图片是否必须使用自己的 H，以及四个状态的位置顺序是否重要。',
 ['固定每张测试图片自己的 F，分别使用原 H、校准图 H 的平均值、一个校准示例的 H，以及另一张图的 H。固定 H 全部来自校准部分。', '再打乱四个状态与四层专家特征的对应顺序，并分别测试清零 H、清零投影后的向量。', '评估 32 张图、8 个族。随机条件使用 5 个种子；这些重复不当作新的独立场景。另用 FP32 解码器复核。'],
 '说明性例子：固定均值条件下，所有测试图都用完全相同的四个向量，但每张图仍有自己的专家特征。位置打乱则把本来对应第 1 层的向量放到另一层。',
 '原状态、固定均值、固定示例和跨图状态均为 100% 正确。打乱顺序后正确率为 58.125%，弃权率 1.25%；两种清零条件均全部弃权。',
 '位置对应关系有作用。在目前数据中，尚未证明每张图独有的末端 H 对正确读出是必要的；清零结果不能直接解释为语言答案受深度控制。'),
12: (
 '检查低答案正确率是否与问法、字母识别或基础识别能力有关，并与原始基座模型比较。',
 ['复用 8 个族中的原图和标签交换图，共 16 张。CoVT 与 Qwen 基座分别完成 8 种任务，各生成 128 次。', '任务包括原技术问法、直白问谁更近/更远、认左边字母、比较面积、回答左右，以及使用较大字母的两个任务。', '每种任务的分母都是同一组 16 张图；使用各模型自己的输入处理器和模板，FP32、greedy、最多 192 个新 token。'],
 '真实 scene_00002_base：CoVT 在原技术问法下答 A，改问 closer to the camera 后答 B，问 farther 后答 A；这张图真值为 B。',
 '两模型的字母识别和面积比较均为 16/16；直白问近均为 11/16。CoVT 原技术问法为 6/16，基座为 8/16。',
 '问法会影响结果，字母识别错误无法单独解释低准确率。本批未显示 CoVT 在直白问近任务上优于基座。'),
13: (
 '检查答案是否随题目要求作出一致变化，补充单题正确率看不到的问题。',
 ['同一张图分别问谁更近、谁更远，检查答案是否互为 A/B。', '保持物体和位置不变，只交换 A/B 字母。问谁更近时字母答案应变化；问左还是右时位置答案应保持不变。', '只在成对记录上计算一致性，并将一致性与正确率分开。'],
 '说明性例子：原图左边的 B 更近。交换字母后，左边变成 A；正确的字母答案应从 B 改为 A，左右答案仍为左。',
 '近/远答案互补：CoVT 9/16，基座 10/16。交换标签后近距离答案翻转：两模型均 3/8 族。左右答案保持一致：分别为 7/8 和 5/8 族。',
 '简单识别任务通过后，关系回答的一致性仍然不足。两个答案互补也不保证两题都答对。'),
14: (
 '检查此前得到的固定 H 换到新的物体形状、遮挡和视角后，是否仍能读出远近关系。',
 ['用种子 1901 生成 50 个族、200 张图，包含尺寸变化、椭球、盒体、部分遮挡和相机变化，每类 40 张。', '保留旧校准的方向、尺度、阈值和固定 H。只为每张新图提取新的 F，不用这些测试图重新校准。', '比较固定均值 H、固定示例 H 和专家自身的深度结果。这一轮无需为全部图片生成新的 CoVT 状态。'],
 '真实失败例子 shift_00001_depth_swap：真值 A，固定均值 r=0.03239，落在弃权区间，因而没有选 A 或 B。',
 '两种固定状态都为 199/200 正确、1 张弃权；专家为 198/200 正确、2 张弃权。固定状态正确率的族区间为 98.5%–100%，与专家区间重叠。',
 '固定状态的效果延续到这批新几何场景。当前结果不足以主张固定状态优于专家，也尚未排除图像中的简单线索。'),
15: (
 '在新场景中直接比较固定 H 和各图自己的 H，并检查“看起来大的物体更近”是否已经足够答对。',
 ['在预先选定的 40 张图上生成自然 CoVT 状态，再比较原状态、固定状态、跨图状态和打乱顺序。', '不生成四个深度标记的图片记为状态提取失败，同时保留语言答案结果。', '另外按可见物体像素面积建立简单规则：总选面积较大的物体更近。在全部 200 张和 40 张子集上分别计分。'],
 '真实 shift_00001_label_swap 直接输出 B，没有生成四个深度标记。这张图有语言答案，但无法进入需要四个状态的对照。',
 '39/40 张有合格状态，语言答案正确 22/40。合格图片中，原生、固定和跨图深度预测相同。面积规则在 200 张上正确率 84%，在 40 张子集上为 92.5%。',
 '固定状态现象在子集中仍存在；较大的物体通常更近，构成明显的混杂因素，因此下一轮要平衡大小线索。'),
16: (
 '让物体远近与图中大小不再总是指向同一答案，检查模型是否还能判断远近。',
 ['种子 1902，8 个族，每族同时组合两种远近顺序与两种大小顺序，共 32 张图。A/B 真值各 16 张，大小线索一致/冲突各 16 张。', '用物体半径=相机深度×角半径来控制投影大小；交换远近时保留指定的角半径和水平投影位置。', '这个构造同时会改变物体的真实半径和地面接触位置。它平衡了大小线索，但没有隔离所有物理因素。', '分别计算语言答案、固定 H 的深度读出和专家深度，并单独统计冲突组。'],
 '真实 balanced_000_d0_s1：A 更近但看起来更小。语言答 A，固定 H 的分数约 +0.07139，判 A；专家分数约 -0.03600，弃权。',
 '全部 32 张：语言 25/32，固定 H 深度 30/32，专家 26/32。冲突的 16 张：分别为 13/16、14/16、10/16。总选大物体的规则在冲突组为 0/16。',
 '大小线索无法单独解释固定 H 的全部成功。模型在冲突组仍能判断不少图片，但数据仍限于小规模合成场景。'),
17: (
 '开始直接测试语言答案：把中间层的四个深度位置换成另一张图的状态，看看后面生成的答案会不会变。',
 ['32 张平衡图中有 26 张自然生成四个深度标记；可配成相反深度对照的目标图为 20 张，来自 6 个族。', '在第 14 层的输入位置替换四个深度状态，然后从最后一个深度标记之后重新生成后续文本。层编号从 0 开始。', '设置原样写回、相反深度供体、同标签且来自其他族的供体，以及位置循环打乱。原样写回用来确认替换代码本身不会改变输出。', '检查状态数值是否真的改变，再记录答案和格式。各条件样本数不同，分别注明。'],
 '真实 balanced_000_d0_s0 的真值和答案均为 A；使用相反深度图 d1_s0 的状态后，目标仍答 A。',
 '原样写回 26/26 完全一致。相反深度条件 20 次、同深度条件 26 次、位置打乱 26 次都没有改变答案，输出均有效。',
 '第 14 层、四个位置的这些替换没有控制答案。替换范围还较小，需要继续扫描其他层并扩大覆盖。'),
18: (
 '检查第 14 层是否选得不合适，并比较自然状态替换与强烈的清零操作。',
 ['沿用 20 张匹配图片，在第 0、4、8、14、20、24、27 层分别替换四个深度位置。', '每层测相反深度供体和置零，共 280 次干预；另做 140 次原样写回。完整重新生成后续答案。', '同时在原答案前的固定上下文上测 A/B 条件概率变化。这个概率测量与自由生成分别报告。', '另做 20 次最终答案位置的正对照，确认测量过程能够检测到人为替换造成的 A/B 变化。'],
 '说明性例子：若替换后 A 的概率只从 84.000% 到 84.007%，虽然数值变了，生成答案仍可能为 A。',
 '自然供体替换在所有层均为 0/20 答案翻转；最大的概率绝对变化约 0.0071 个百分点。置零变化较大，但许多输出格式异常，下一节单独核查。',
 '单层自然供体替换的影响在已扫描位置很小。置零后出现文本变化，还需要确认是否得到有效答案。'),
19: (
 '重新核查置零结果，避免把停止生成、重复文本或含糊字母误算为答案改变。',
 ['逐条读取置零后的原始文本，检查完整答案标签、唯一 A/B 字母及是否同时出现 A 和 B。', '保留原解析器结果作为审计记录，同时单独报告严格格式是否有效。', '把固定上下文下的概率变化与重新生成文本区分：前者已经给定正常答案前缀，后者可能根本没有生成到正常答案位置。'],
 '真实第 4 层输出含重复结束标签和多个 B；第 14 层有输出写成 A or B。这样的文本不适合直接作为干净的远近答案翻转。',
 '第 4、8、14、24 层置零后的完整无歧义答案均为 0/20；第 20 层只有 3/20，且答案未变。第 0、27 层均 20/20 有效，答案也未变。',
 '置零说明计算过程会受到干扰。当前输出不足以证明某个深度语义被稳定地改变成另一个深度语义。'),
20: (
 '直接检验“只换一层或四个位置，范围太小”的解释，逐步扩大替换层数和位置数。',
 ['仍使用同一组 20 张匹配图、6 个族，依次替换前部 0–8 层、中部 9–17 层、后部 18–27 层，以及全部 0–27 层。前四个阶段都替换四个深度位置。', '全层四位置没有有效翻转后，按事先写好的扩展规则，替换全部 28 层的 15 个生成前缀位置：前面 11 个位置加 4 个深度标记。', '每阶段比较相反深度供体和同深度供体。同深度供体来自其他族，并匹配标签、远近顺序、大小顺序和线索类别。', '供体状态来自其自然运行。在目标生成的每一步重新施加替换，再生成完整后续文本。供体答案 token 不参与替换。', '5 个阶段各做 20 次原样写回和两类各 20 次干预，共 100 次检查、200 次实际干预。'],
 '说明性例子：全层条件下，目标经过每一层时，都把指定位置换成供体的对应状态。15 位置条件进一步包括深度标记之前的生成位置。',
 '100/100 次原样写回完全一致。200 次干预均正常结束，答案格式有效，整段后续 token 均未改变。相反深度的最大概率变化从前层阶段约 0.0108 个百分点增加到 15 位置阶段约 0.0472 个百分点。',
 '扩大层数和位置后仍没有改变答案，因此单层覆盖少已不足以单独解释阴性结果。实验仍未覆盖图像位置、问题位置等其他信息路径。'),
21: (
 '用概率变化最大的真实案例说明全层实验到底改变了多少，并明确哪些位置还没有测试。',
 ['从已完成的全层、15 位置、相反深度干预中，选绝对概率变化最大的案例作解释；该案例不是随机代表样本。', '比较目标和供体的图像、真值与自然答案，再核对干预前后概率、生成文本和状态差异。', '单独统计供体自然答案确实与目标不同的目标图，检查零翻转是否仅因供体也给出同一答案。'],
 '真实目标 balanced_003_d1_s0 的真值和答案为 A；供体 balanced_003_d0_s0 的真值和答案为 B。替换后目标仍答 A。',
 'A 的条件概率从 0.84342670 变为 0.84295487，减少约 0.04718 个百分点。20 张目标中有 6 张的相反深度供体自然答案不同，这 6 张仍全部没有翻转。',
 '供体答案不同也没有带来有效控制。下一步应检查信息位置和供体状态是否包含足够的远近差异；仅继续扩大相同替换范围，依据有限。'),
}

def esc(s):
    return html.escape(str(s))

def markup(s):
    # Source is authored report text; preserve only its simple mathematical markup.
    s = re.sub(r'第 (\d+) 页', lambda m: f'第 {m[1]} 节', s)
    return re.sub(r'<font[^>]*>(.*?)</font>', r'\1', s)

def render_block(name, args):
    if name == 'table':
        heads, rows = args[:2]
        return '<div class="table-scroll"><table><thead><tr>' + ''.join('<th scope="col">'+esc(x)+'</th>' for x in heads) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    if name == 'figure':
        path, caption = args[:2]
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        return f'<figure><img loading="lazy" src="data:image/png;base64,{data}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'
    if name == 'formula':
        return '<div class="formula">'+markup(args[0])+'</div>'
    if name == 'para':
        return '<p><strong>'+esc(args[0])+'。</strong> '+markup(args[1])+'</p>' + ('<p class="minor-conclusion">本段结论：'+markup(args[2])+'</p>' if len(args)>2 and args[2] else '')
    if name == 'conclude':
        return '<p class="conclusion"><strong>当前结论：</strong>'+markup(args[0])+'</p>'
    return '<p>'+markup(args[0])+'</p>'

parts = []
for n, section in enumerate(sections, 1):
    if n == 1:
        continue
    title = section['title']
    if n == 2:
        title = '先了解实验之间的关系'
    parts.append(f'<section id="s{n}"><div class="section-number">{n:02d}</div><h2>{esc(title)}</h2>')
    if section['source']:
        parts.append('<p class="source">证据记录：'+esc(section['source'])+'</p>')
    if n in EXPERIMENTS:
        purpose, design, example, result, conclusion = EXPERIMENTS[n]
        parts.append('<h3>实验目的</h3><p>'+esc(purpose)+'</p><h3>实验内容与设计</h3><ol class="steps">'+''.join('<li>'+esc(step)+'</li>' for step in design)+'</ol>')
        parts.append('<aside class="example"><strong>简单例子</strong><p>'+esc(example)+'</p></aside>')
        parts.append('<h3>结果与分析</h3><p>'+esc(result)+'</p>')
        if any(name == 'figure' for name, _ in section['blocks']):
            parts.append('<p class="figure-intro">下面的图展示本节的实际输入或测量结果；具体样本、坐标轴与颜色含义见图注。</p>')
        for name, args in section['blocks']:
            if name in {'table', 'figure'}:
                parts.append(render_block(name, args))
        parts.append('<p class="conclusion"><strong>当前结论：</strong>'+esc(conclusion)+'</p>')
        parts.append('<details><summary>计算方法、参数与原始报告说明</summary>')
        for name, args in section['blocks']:
            if name not in {'table', 'figure'}:
                parts.append(render_block(name, args))
        parts.append('</details>')
    elif n == 2:
        parts.append('<p>E0 检查恢复流程是否可靠。E1 分别替换语言模型状态和专家特征，检查深度图由哪些输入决定。之后的实验检查问法、图像线索，最后才通过中间层替换测试语言答案。</p>')
        parts.append('<p>所有模型保持冻结，没有做修复训练。实际模型实验在 Colab A100 完成；这个网页只整理已有记录。</p>')
        for name, args in section['blocks']:
            if name != 'table':
                parts.append(render_block(name, args))
        parts.append('<aside class="example"><strong>几个常用词</strong><p>状态：模型在某层某个 token 位置的一组数值。供体：提供替换状态的另一张图。目标：接受替换、随后重新生成答案的图。原样写回：把目标自己的状态放回去，用于检查干预代码。场景族：共享基础场景的一组变体。</p></aside>')
    else:
        parts.extend(render_block(name, args) for name, args in section['blocks'])
    parts.append('</section>')

toc = ''.join(f'<a href="#s{i}"><span>{i:02d}</span>{esc(s["title"])}</a>' for i,s in enumerate(sections,1) if i>1)
css = '''
:root{--ink:#233747;--muted:#617080;--accent:#176f69;--line:#dce4e9;--paper:#fff;--bg:#f3f6f8}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:22px}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.9 "Microsoft YaHei","PingFang SC",system-ui,sans-serif}a{color:#176b9a}a:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #d78335;outline-offset:4px}.layout{max-width:1450px;margin:auto;display:grid;grid-template-columns:280px minmax(0,1fr);gap:40px;padding:30px}nav{position:sticky;top:25px;align-self:start;max-height:calc(100vh - 50px);overflow:auto;padding-right:10px;font-size:13px}nav h2{font-size:18px;margin:0 0 16px}nav a{display:flex;gap:10px;text-decoration:none;color:var(--muted);padding:8px 4px;line-height:1.6;border-bottom:1px solid var(--line)}nav a:hover{color:var(--accent);background:#e8f0f1}nav span{font-variant-numeric:tabular-nums;color:var(--accent)}main{min-width:0}header,section{background:var(--paper);padding:44px 48px;margin-bottom:24px;border:1px solid var(--line);border-radius:8px}header{border-top:5px solid var(--accent)}.eyebrow,.section-number{font-size:13px;letter-spacing:.1em;color:var(--accent);font-weight:700}h1{font-size:34px;line-height:1.5;margin:16px 0}h2{font-size:25px;line-height:1.55;margin:8px 0 18px}h3{font-size:18px;margin:25px 0 8px;color:#245770}p{margin:10px 0 17px}.source,.meta{font-size:13px;color:var(--muted);overflow-wrap:anywhere}.steps{padding-left:24px}.steps li{padding-left:4px;margin:10px 0}.example{border-left:3px solid #c59048;padding:15px 22px;background:#fcf8f1;margin:25px 0}.example p{margin:4px 0}.conclusion{border-left:3px solid var(--accent);background:#edf6f3;padding:16px 20px;margin:24px 0 8px}.minor-conclusion{font-size:14px;color:var(--accent)}figure{margin:25px 0}figure img{display:block;width:100%;height:auto;border:1px solid #edf0f2;background:white}figcaption,.figure-intro{font-size:13px;color:var(--muted);line-height:1.8;margin-top:12px}figcaption{overflow-wrap:anywhere}.table-scroll{max-width:100%;overflow-x:auto;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.7}th{background:#24475e;color:white;text-align:left}td,th{padding:12px 14px;vertical-align:top;overflow-wrap:anywhere;border-bottom:1px solid var(--line)}tbody tr:nth-child(even){background:#f3f7f9}details{margin-top:24px;border:1px solid var(--line);border-radius:5px;padding:14px 20px;font-size:15px}summary{cursor:pointer;color:#245770;font-weight:600}details[open] summary{margin-bottom:20px}.formula{background:#eef3f7;padding:18px 20px;margin:15px 0;overflow-x:auto;font-family:"Microsoft YaHei",sans-serif;line-height:2}button{border:1px solid #a8b9c5;border-radius:5px;padding:9px 14px;background:white;color:var(--ink);cursor:pointer;margin:6px 8px 6px 0;font:inherit;font-size:13px}.overview{padding-left:23px}.overview li{margin:10px 0}footer{font-size:13px;color:var(--muted);padding:15px 5px 40px}.skip{position:absolute;left:-9999px}.skip:focus{left:20px;top:10px;z-index:2;background:white}.top{display:inline-block;margin-top:12px}
@media(max-width:1000px){.layout{grid-template-columns:220px minmax(0,1fr);gap:20px;padding:20px}header,section{padding:30px}}
@media(max-width:760px){body{font-size:16px}.layout{display:block;padding:12px}nav{position:static;max-height:260px;background:white;padding:16px;border:1px solid var(--line);margin-bottom:16px}header,section{padding:24px 20px}h1{font-size:28px}h2{font-size:23px}td,th{padding:9px;font-size:13px}.table-scroll table{min-width:460px}}
@media print{body{background:white;font-size:11pt}.layout{display:block;padding:0}nav,.actions,.top{display:none}header,section{border:0;border-radius:0;padding:15px 0;break-before:page}header{break-before:auto}h2,h3{break-after:avoid}figure,tr,.conclusion{break-inside:avoid}details{border:0}summary{display:none}.table-scroll{overflow:visible}a{color:inherit;text-decoration:none}}
'''
document = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="CoVT 已完成实验的目的、设计、真实案例、结果和结论；独立离线版。"><title>CoVT 实验报告 · HTML 详解版</title><style>'''+css+'''</style></head><body id="top"><a class="skip" href="#main">跳到正文</a><div class="layout"><nav aria-label="报告目录"><h2>实验报告目录</h2>'''+toc+'''</nav><main id="main"><header><div class="eyebrow">CoVT PILOT · 已完成实验</div><h1>CoVT 实验报告<br>目的、做法与目前的结论</h1><p class="meta">实验记录截至 2026 年 9 月 18 日 · HTML 详解版 · 全部图片内嵌，可离线阅读</p><p>我们想回答两个问题：能不能恢复模型原来的深度图？这些深度状态是否会影响模型最后的文字答案？为此，先核对计算流程，再做图像和状态对照，最后逐步扩大语言模型内部的替换范围。</p><ul class="overview"><li><strong>原生深度分支已经恢复。</strong>完整 FP32 的 40 张图通过流程检查；其中 32 张评估图的深度关系全部正确，语言答案答对 8 张。</li><li><strong>固定状态也能读出很多正确的远近关系。</strong>新几何集合为 199/200；大小线索与远近冲突时为 14/16。专家图像特征提供的信息需要单独考虑。</li><li><strong>目前还没有实现稳定的答案控制。</strong>在全 28 层、15 个生成位置的替换下，20 张目标图的答案仍未改变。这个结果有明确的样本和位置范围。</li></ul><p class="conclusion"><strong>总的结论：</strong>深度图能读对，与语言答案会利用相应状态，是两个需要分别验证的问题。目前已确认前者在这些合成场景中可行，后者尚未获得有效干预证据。</p><div class="actions"><button type="button" id="expand">展开全部计算细节</button><button type="button" id="collapse">收起全部计算细节</button><button type="button" id="print">打印 / 保存为 PDF</button></div></header>'''+''.join(parts)+'''<footer>本报告基于项目内已归档记录与已核对的 PDF 内容整理。网页制作没有新增模型推理或训练。报告中的“下一步”均为建议，未作为已完成结果。<br><a class="top" href="#top">回到顶部 ↑</a></footer></main></div><script>
const panels = [...document.querySelectorAll('details')];
document.getElementById('expand').addEventListener('click',()=>panels.forEach(p=>p.open=true));
document.getElementById('collapse').addEventListener('click',()=>panels.forEach(p=>p.open=false));
document.getElementById('print').addEventListener('click',()=>window.print());
let savedOpen=[];
window.addEventListener('beforeprint',()=>{savedOpen=panels.map(p=>p.open);panels.forEach(p=>p.open=true)});
window.addEventListener('afterprint',()=>panels.forEach((p,i)=>p.open=savedOpen[i]));
</script></body></html>'''
target = OUT / 'CoVT_实验综合报告_详解版.html'
target.write_text(document, encoding='utf-8')
assert document.count('<figure>') == 13
assert document.count('<section ') == 23
assert document.count('<details>') == 19
assert all(f'id="s{i}"' in document for i in range(2,25))
assert 'src="http' not in document and '<script src=' not in document
(OUT / 'report_manifest.json').write_text(json.dumps({'html':target.name,'figures':13,'offline':True,'new_model_experiments':False,'plain_language_sections':list(EXPERIMENTS),'source':'scripts/build_research_report.py'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'path':str(target),'bytes':target.stat().st_size,'figures':13},ensure_ascii=False))
