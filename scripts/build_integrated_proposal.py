"""Merge proposal one, accessible explanation and archived results into HTML/PDF.

Reads previously reviewed artifacts. No model, rendering of new scenes, or training.
"""
from pathlib import Path
import base64,copy,hashlib,html,io,json,re
from lxml import html as LH, etree
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]; PROPOSALS=ROOT.parent/'vlm_proposals'
OUT=PROPOSALS/'integrated';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'output/html/CoVT_实验综合报告_详解版.html'
tree=LH.fromstring(SOURCE.read_text(encoding='utf8'))
def element(id):return tree.get_element_by_id(id)
def inner(e):return (html.escape(e.text) if e.text else '')+''.join(etree.tostring(c,encoding='unicode',method='html') for c in e)
def content(id):
    e=copy.deepcopy(element(id))
    for c in list(e):
        if c.tag=='h2' or c.get('class') in ['section-number','eyebrow']:e.remove(c)
    return inner(e)
def figures(id):return ''.join(etree.tostring(e,encoding='unicode',method='html') for e in element(id).xpath('.//figure'))
def dataset_overview():
    blocks=[]
    for c in element('dataset-show'):
        if c.tag=='h3':break
        if c.tag!='h2':blocks.append(etree.tostring(c,encoding='unicode',method='html'))
    return ''.join(blocks)
def table(head,rows):return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def conclude(s):return '<p class="conclusion"><strong>本节结论：</strong>'+s+'</p>'
chapters=[]
def add(title,body):chapters.append((title,body))

add('CoVT 在研究什么，为什么值得研究？','''
<p><strong>CoVT 全称为 Chain-of-Visual-Thought，中文可叫“视觉思维链”。</strong>本项目研究这一公开方法的深度版本，文件夹与文稿统一使用 CoVT。</p>
<p>我们关注一个容易被忽略的问题：<strong>模型给人看的中间图像，是否真的反映它作出答案时使用的信息？</strong>例如模型画出的深度图正确地显示 B 更近，但文字却回答 A。这时需要知道哪里出了问题：是图主要靠另一个模型画对的，还是信息已经存在、但回答过程没有使用，或者答案走了别的信息路径？</p>
<h3>为什么这件事有意义？</h3><p>当模型能展示深度图、物体区域或其他中间结果时，人们可能据此判断模型的回答是否可靠。如果这些展示与答案的计算关系不清楚，一张漂亮或正确的图就不足以解释某次回答。弄清这种关系，有助于正确使用可视化，也能帮助决定应该修复视觉表示、回答过程，还是两者之间的连接。</p>
<p>本项目希望得到的成果有三个：一个可以复现的原生可视化检查流程；对错误来源的受控区分；在定位到明确问题后，检验是否能做有限的修复。前两项已有部分进展，修复尚未进行。我们不预设 CoVT 一定存在某一种缺陷。</p>
<h3>这份整合版与原 proposal 的关系</h3><p>主线来自第一篇 <em>Dual-Readout Misalignment in Visual Thoughts</em>，可直译为“视觉思维的两种输出之间是否不一致”。这里的两种输出，就是深度图读出的远近关系和模型的文字回答。第二篇关于可复用查询状态的 proposal 保留为独立方向，没有把深度实验作为其验证结果。</p>
<p>原 proposal 写于 2026 年 9 月 17 日，包含研究计划。这份中文整合版纳入截至 9 月 18 日的实验记录，并用“已完成”“部分完成”“尚未执行”区分状态。所有训练、扩展模型和自然图片验证计划，都不作为已完成结果。</p>
'''+table(['读者第一次遇到的词','这里的含义'],[['视觉语言模型（VLM）','能接收图片与文字问题，并生成文字回答的模型。'],['状态','模型在某一层、某一个位置的一组内部数值。'],['解码器','把内部数值与相关输入转换为可查看结果的计算模块。'],['对照实验','保持一部分条件相同，只改变指定因素，比较结果。'],['干预','直接修改模型内部某处数值，再运行后续计算。']])+conclude('我们研究的是可视化与回答之间的实际计算关系；研究意义来自弄清机制和结论边界，不取决于一定找到模型失效。'))

add('从普通看图问答到 CoVT：读原论文总览图','''<p>普通看图问答模型接收图片和文字问题，经过计算，生成答案。文字思维链让模型先写出中间步骤；CoVT 希望这些中间步骤还包含与视觉信息有关的数值表示。它不是人的主观思维，“思维链”是方法名称。</p>'''+content('start-figure1'))
add('视觉表示、可视化图和文字答案分别是什么？',content('start-figure2'))
add('深度专家怎样从图片得到远近信息？',content('lesson3'))
add('四个深度状态怎样参与生成深度图？',content('lesson4'))
add('CoVT 怎样训练？训练与实际回答有什么区别？',content('start-figure3')+content('lesson5'))

add('把原 proposal 的研究问题变成可检验的解释','''
<p>我们先区分三件事：<strong>某个状态中可以读出信息；它参与的计算能得到正确图；模型确实依赖它作答。</strong>这三件事相关，但没有哪一件自动证明另外两件。原 proposal 的重点是通过实际修改计算来区分解释。</p>
<p>如果模型原来答 A，替换状态后改答 B，首先说明这次操作影响了回答。它是否代表“正确传递了深度信息”，还要看替换来源、目标真值、格式是否正常以及其他对照。特别是目标图片仍显示 A 更近时，注入相反状态让它答 B 也可能使答案变错；这类测试测的是依赖关系，不直接代表性能改善。</p>
'''+table(['可能解释','通俗含义','当前证据状态'],[
['专家特征支持了正确可视化','另一套深度模型已经提供了大量空间信息，因此固定状态也可能画对。','固定状态与输入交换实验支持这一解释在当前数据中起重要作用。'],
['视觉状态也帮助答案','同一视觉信息能沿后续计算影响图和文字。','当前自然供体替换未找到稳定答案效应，不能宣布已验证。'],
['两种输出读取了不同内容','图能读出任务信息，但回答没有沿同一方式利用它。','尚未定位具体读取路径，不能仅凭图对、文字错确认。'],
['答案依靠其他图像信息','模型还可以通过图像位置或其他文字位置获得信息。','与目前现象相容，但没有完成路径隔离，仍是假设。'],
['训练或生成格式产生影响','模型表现可能受训练后的权重和输出格式影响，而不只是四个状态内容。','清零会损坏输出；基座对照不能单独区分具体训练成分。']])+'''
<h3>最重要的设计要求</h3><p>如果要比较同一内部改动对深度图和文字的影响，必须在两者都能受到影响的计算位置做改动，并重新计算所有受影响的后续结果。只在最后画图模块改数值，不会自动构成对文字生成的干预。这是原 proposal 中“共同上游位置”的意思。</p>
'''+conclude('几种解释可以同时成立。当前证据最清楚的是专家特征对可视化的重要作用；答案信息经过哪里、是否存在特定读取错误，还没有确定。'))

add('数据从哪里来？为什么示例中 B 更近？',content('dataset-case')+dataset_overview()+'<h3>实际图片：几何变化与大小平衡</h3><p>第一组图展示尺寸、形状、遮挡和相机变化。第二组图展示同一基础场景的远近顺序与表观大小如何分别安排。图片均来自实际实验归档。</p>'+figures('s14')+figures('s16')+'''
<p>这份文稿展示关键样例。全部 284 张基础场景及其标签，可在项目的“CoVT_数据集展示.html”中查看；它是同批数据的浏览方式，不是新评测集。独立场景族数为 71，衍生变体和重复运行不能作为独立样本数量相加。</p>
'''+conclude('当前数据用于控制变量和检查机制，仍以简单几何图为主。没有完成自然图片上的验证，也没有执行原 proposal 的大规模确认性数据计划。'))

add('怎样计分？先把“正确”“弃权”“改变”讲清楚','''
<h3>远近标签来自哪里？</h3><p>每个标记处的真实表面点都有场景深度 z；z 较小者更近。这与模型预测无关。专家也会预测错误，因此不能把专家预测当成标准答案。</p>
<h3>怎样从深度图得到 A/B？</h3><p>在 A、B 标记中心周围取半径 5 到 9 像素的环带，分别求图上分数的中位数。再用 B 减 A，按校准图确定的方向和尺度转换为关系分数 r。r&gt;0.05 判 A，r&lt;-0.05 判 B，中间范围记为弃权。</p><p>说明性例子：方向为 -1、尺度为 100，B-A=-80 时，r=0.8，判 A；B-A=-3 时，r=0.03，弃权。阈值是当前小实验的固定设定，未被证明适用于所有数据。专家图与重建图均依据记录中的评分规则计分。</p>
'''+figures('s4')+'''
<h3>怎样看文字答案是否改变？</h3><p>同时保留完整自由生成的文本与答案位置的概率。有效翻转要求输出有明确、正常的 A/B 答案，并与原答案不同。提前结束、重复标签和“A or B”分别审计。概率轻微变化不等于答案变化。</p>
<h3>怎样处理同一场景的多张图？</h3><p>逐图正确率是正确图片数除以相应评估图片数；族平均先算每个场景族的结果，再让各族占相同权重。区间用场景族为单位重复抽样 2000 次，种子为 0，取 2.5% 和 97.5% 分位数。随机打乱重复 5 次仍然只有原来的场景族数。</p><p>说明性例子：一族 3 张全对，另一族 1 张全错，逐图正确率为 75%，族平均为 50%。少数场景族全成功时，抽样区间可能退化为一个值，这不代表总体不确定性消失。</p>
'''+conclude('深度正确率、文字正确率、格式有效率、答案翻转率和概率变化分别回答不同问题；全文不把它们混成一个分数。'))

# Seven completed experiment narratives, with the important archived figures.
extra={6:[7,6],7:[9,10],8:[11],9:[12,13],10:[],11:[18],12:[20,21]}
titles={6:'实验一：恢复原生深度图并排查计算差异',7:'实验二：分别替换状态和专家特征',8:'实验三：固定状态能否用于不同图片',9:'实验四：排查问法、字母识别与回答一致性',10:'实验五：平衡物体大小与远近关系',11:'实验六：修改中间状态，重新生成文字答案',12:'实验七：扩大到多层、全部层和更多位置'}
for lesson in range(6,13):
    body=content(f'lesson{lesson}')
    # Balanced images already appear in the dataset section; avoid duplicating them.
    if lesson==10:
        e=LH.fragment_fromstring(body,create_parent='div')
        for f in e.xpath('.//figure'):f.getparent().remove(f)
        body=inner(e)
    if lesson==8:
        body+='''<p><strong>扩展子集的结果：</strong>新 200 张中预选 40 张做自然状态对照，39 张生成合格的四个状态。语言正确 22/40；合格图的原生、固定、跨图深度预测一致。位置打乱的族平均正确率为 56.5%。无法生成四个状态与前面的数值精度问题是不同失败类型。</p>'''
    if lesson==7:
        body+='''<p><strong>下面两张图怎样读：</strong>H 表示四个 CoVT 状态，F 表示专家图像特征；0 代表原图，1 代表交换后的图。r 是上一节定义的关系分数，正值判 A，负值判 B，接近零时弃权。四组合图用于直接比较“只换 H”和“只换 F”；跨场景曲线再检查这种差别是否重复。纵轴对数刻度每相邻数量级相差十倍。</p>'''
    if lesson==10:
        body=body.replace('下图展示一个真实场景的四种组合。重点看：同一远近顺序下，哪一个物体看起来更大可以改变。','第 8 节已展示一个真实场景的四种组合。同一远近顺序下，哪一个物体看起来更大可以改变。')
        body+=table(['范围','文字答案正确','固定状态深度正确','专家深度正确'],[['全部 32 张','25/32','30/32','26/32'],['大小一致 16 张','12/16','16/16','16/16'],['大小冲突 16 张','13/16','14/16','10/16']])
    if lesson==11:
        body+='''<p><strong>选择范围：</strong>32 张平衡图中 26 张生成四个深度标记；20 张满足相反深度配对要求，来自 6 个族。原样写回与最终答案读出正对照通过，但后者只说明检测工具能观察到答案变化，不能证明深度语义已被正确控制。</p>'''
        body+=table(['置零位置（层号）','完整无歧义答案','解释'],[['0、27','各 20/20','答案均未改变。'],['4、8、14、24','各 0/20','出现标签损坏、歧义或提前结束。'],['20','3/20','有效的 3 个答案未改变。']])
    if lesson==12:
        body+='''<p><strong>细微概率变化的比较：</strong>最宽范围内，同深度供体的最大变化为 0.0422 个百分点，相反深度为 0.0472。相反减同深度的族平均定向效应约 0.00377 个百分点，区间 [-0.00573, 0.0136] 包含零。当前没有证明两种效果等价，也没有得到稳定方向控制的证据。</p>'''
    for original in extra[lesson]:body+=figures(f's{original}')
    body+=conclude(['当前可靠恢复了读出流程；图对而文字错是待解释的观察。','状态与专家输入的影响已被分开；这一步只检验画图分支。','固定状态现象有重复证据，但只适用于已测试的输入与评分。','问法和基本识别不足以解释所有错误；关系一致性仍有限。','简单大小规则不能解释全部成功，其他线索仍可能存在。','已测试的单层操作没有控制正常答案，异常输出不能作为语义转移证据。','覆盖更多层和位置仍未改变答案；具体信息路径尚未定位。'][lesson-6])
    add(titles[lesson],body)

add('把结果放回原 proposal：哪些目标已经达到？','''
<p>原 proposal 用 E0 至 E5 给研究阶段编号。前文为了容易阅读，按七组实际实验说明；两套编号含义不同。下表把当前结果对应回原研究计划。</p>
'''+table(['原计划','原本希望得到的证据','目前进展'],[
['E0：恢复与校准','原生输出可复现，评分规则可靠。','主要实现已完成；FP32 40/40 检查通过。仍不能保证任意输入都可靠。'],
['E1：解码器输入分解','区分状态与专家特征的作用。','已完成小规模对照，固定状态和新几何扩展也已完成。'],
['E2：共同上游语义干预','同一次改动重算图与文字，得到可解释的联合变化。','部分完成：中间层及范围扩展已做，未获得稳定答案控制；尚未形成充分的双输出语义分离证据。'],
['E3：定位答案与视觉信息路径','确定具体路径，区分状态使用与其他信息来源。','尚未完成。扩大层数不能替代路径定位。'],
['E4：比较两种输出敏感的方向','用梯度等方式筛选候选，再用有限干预验证。','尚未执行；不应把梯度差别直接当成机制结论。'],
['E5：有限修复','在已定位问题上修复，并在独立数据验证。','尚未执行；目前不满足直接宣称读取错误并开展定向修复的证据条件。']])+'''
<p>原 proposal 计划了较大的发现、验证和确认数据划分，也计划了自然图片及其他模型复现。目前只有 284 条基础图像记录、71 个基础场景族；答案与干预实验反复使用其中子集。不能把原计划的样本预算写成已经完成的数据量。</p>
'''+conclude('当前成果主要是恢复与诊断证据；原 proposal 最有研究价值的路径定位和受控修复仍然是后续工作。'))

add('接下来如何做：每一步先规定什么结果才值得继续','''
<p><strong>以下全部是计划，尚未执行。</strong>依据当前结果调整顺序，优先确认有没有可控的任务信息，再决定是否扩大样本或训练。不能因为一次替换没效果，就无限扩大同类扫描。</p>
<h3>第一步：检查目前没有覆盖的计算位置</h3><p>先检查深度标记之后、正式答案之前的过渡文字位置，再检查问题位置和图像位置。所有条件都保留目标输入，记录准确位置和所替换数值，并与同位置原样写回、同深度供体及无关变化比较。若使用固定后续文本测概率，应同时报告不固定后续文本的完整生成。</p>
<p><strong>继续条件：</strong>至少找到一种正常输出下的可重复任务相关变化，且相反深度供体的影响超过匹配对照。若仍只有格式损坏或极小、无方向的变化，应收窄为诊断边界，不直接进入修复训练。</p>
<h3>第二步：从同一次干预同时重算两种输出</h3><p>选定双方都能受到影响的位置，修改后重新计算四个状态、重建图和文字答案，不能只刷新其中一个分支。专家特征等保持项明确记录。观察是否出现图与文字一起随预期改变，或其中一方改变而另一方保持。</p>
<p>说明性例子：目标图 A 更近，供体 B 更近；若替换后重建图判 B、文字仍答 A，说明两个输出对该操作反应不同。目标原图仍然支持 A，所以还需要路径证据，不能马上称为模型犯错。</p>
<h3>第三步：定位具体信息传递过程</h3><p>把“图像到状态再到答案”和“图像经其他位置到答案”分开检查。可先在发现数据上筛选少量候选组件，再在独立数据上做精确替换。只禁止答案直接读取图像位置，也可能遗漏已经复制到文字位置的信息，因此不能轻易声称切断了所有旁路。</p>
<h3>第四步：有明确定位后，再做有限修复</h3><p>先用诊断性的正确状态替换检查错误是否有机会被修复；这种操作使用了额外正确信息，只是诊断上限。之后才考虑在少量已定位的回答组件上训练小型更新。与相同训练数据、相同参数量的普通微调以及随机位置更新比较，才能区分定向修复和一般训练收益。</p>
<p>现阶段删除原 proposal 中提前指定的训练组件数、更新秩和训练场景数量。只有确定需要修复的位置与错误类型后，才根据验证结果确定训练规模、停止规则和对照。测试数据与供体正确答案不得进入实际部署。</p>
<h3>第五步：检验结论能否扩展</h3><p>在新种子、更多物体和相机设置上冻结方案复测；随后考虑自然图片远近任务。分割可作独立扩展，但需要相应模型和解码器，也要重新定义区域与答案指标。自然图像缺少精确受控几何，不能沿用合成图一样强的反事实解释。</p>
'''+conclude('下一步的优先目标是得到可重复、格式正常且任务相关的干预效应，并定位路径；训练修复应在这些证据之后。'))

add('怎样保证结论可信：统计、失败和停止规则','''
<p>发现阶段用于选择层、位置、供体策略和指标；验证阶段用于检查候选是否重复；最终确认阶段使用此前没有用于选择方案的独立场景族。同一族的所有变体、问题和干预必须保持在同一划分中。当前大部分扫描属于探索，不构成完成了这种确认性流程。</p>
<p>供体若被多个目标复用，统计需要尊重这层相关性；未来可采用不重叠配对或同时考虑供体与目标依赖的重采样方式。根据已有试验估计变异，再确定样本量与停止规则，不能仅根据运行次数声称统计能力充足。</p>
<p>“没有观察到改变”和“已经证明等价”要区分。若要主张两种条件近似等价，应提前规定多大的变化才有实际意义，并要求区间落在该范围内。当前的 0/20 翻转及很小概率差，只描述已测数据；没有预先给定且验证过的普遍等价界限。</p>
<p>所有不生成深度状态、格式损坏、缓存不一致和干预失败都要保留。既报告全部尝试的结果，也报告符合操作条件的子集，避免只展示成功提取或只展示明显案例。案例图片用于解释，主结论依据整批结果。</p>
'''+table(['阶段','进入下一阶段的条件','条件未满足时'],[['计算恢复','原样写回和必要的数值检查通过','先修实现，不解释机制。'],['语义干预','正常输出中出现可重复、超过对照的方向效应','报告已测范围与限制，不盲目扩大扫描。'],['机制定位','具体信息路径的解释在新场景与新供体上成立','保留竞争解释。'],['修复与推广','独立图片上改善一致性，且不牺牲正常视觉证据','不宣称已找到通用修复。']])+conclude('失败与阴性结果本身也需要有清楚的分母、作用范围和可复核记录。'))

add('预期贡献、当前限制与项目判断','''
<p>原 proposal 的贡献目标不止于展示“深度图对、答案错”，也不止于报告“删掉某些状态答案没变”。较强的成果需要说明一个明确的计算分支、一次可解释的共同上游改动、两种输出的反应，以及对正常错误的解释能力；有限修复或跨模型复现可进一步支持其意义。</p>
<p>当前最稳妥的阶段性贡献是：恢复可复现的原生深度检查流程；展示专家特征与状态的作用差异；发现固定状态在多批合成场景中仍能支持正确读出；通过格式审计限制置零结果的解释；确定一组自然状态替换在指定层和位置上没有控制答案。</p>
<p>局限包括：只测试一个公开深度版本，基础图片数量有限且变体相关，部分样本没有自然生成四个标记，供体有复用，许多位置尚未检查，没有自然图片验证，也没有修复训练。模型还可能通过多条路径获得相同信息，或合理地优先依据目标图片而不是注入的相反状态。</p>
<p>原 proposal 的文献定位已经提醒：相关研究讨论过视觉状态利用、替换测试和图像捷径。本整合版沿用其截至 2026 年 9 月 17 日的文献审计范围，不把“状态可能未被使用”宣称为新的发现。正式投稿前需要重新核对文献与差异点。</p>
'''+conclude('项目值得继续的关键是能否从现有诊断推进到可验证的计算机制。当前结果支持继续做有针对性的路径检查，不支持宣布已证明普遍失效或已完成修复。'))

add('附录：精确配置、指标与证据索引',content('s23')+content('s5')+content('s24')+'''
<h3>本整合版的来源与复现</h3><p>研究计划来源：vlm_proposals/dual_readout_proposal.tex；原参考文献：dual_readout_references.bib；文献审计：literature_audit.md。第二篇 reusable_query_states_proposal.tex 是独立研究方向。本版没有修改原始英文文件。</p><p>图文与结果来源：covt-pilot/output/html/CoVT_实验综合报告_详解版.html，以及其中标明的 runs 归档。原论文图来自官方 PDF，裁切坐标和源文件校验值保存在 output/html/paper_figures/provenance.json。</p><p>本版生成脚本：covt-pilot/scripts/build_integrated_proposal.py。脚本只读取已存在的图文和记录，输出中文 HTML、PDF 和来源清单，没有模型推理或训练。</p><p>进一步阅读：<a href="https://github.com/Wakals/CoVT">CoVT 官方仓库</a>；<a href="https://wakalsprojectpage.github.io/covt-website/static/pdf/paper.pdf">CoVT 官方论文</a>；<a href="https://arxiv.org/html/2406.09414v2">Depth Anything V2 论文</a>。原论文示意图的版权归原作者。</p>''')

# Normalize and assemble a single readable document, rather than appending reports.
normalized=[];figure_number=0
for index,(title,body) in enumerate(chapters,1):
    for a,b in [('ᵢ','<sub>i</sub>'),('꜀','<sub>c</sub>'),('₁','<sub>1</sub>'),('₂','<sub>2</sub>'),('⁻¹²','<sup>-12</sup>'),('²','<sup>2</sup>')]:body=body.replace(a,b)
    e=LH.fragment_fromstring(body,create_parent='div')
    for h in e.xpath('.//h2'):h.tag='h3'
    for f in e.xpath('.//figure'):
        figure_number+=1
        caps=f.xpath('./figcaption')
        if caps:
            cap=caps[0]
            raw=cap.text_content()
            raw=re.sub(r'^图\s*\d+\s*[｜|]','',raw).strip()
            if raw.startswith('原论文图'):raw='来源：'+raw
            for c in list(cap):cap.remove(c)
            cap.text=f'图 {figure_number}｜'+raw
    for p in e.xpath('.//p'):
        if p.text:p.text=re.sub(r'第 23 节','本版附录',p.text)
    normalized.append((title,inner(e)))
chapters=normalized
css=tree.xpath('//style')[0].text
css+=''' .layout{grid-template-columns:270px minmax(0,1fr)}.walk{break-inside:avoid}#main .sample-grid{max-width:340px}.status{font-weight:bold;color:#176f69}.title-lead{font-size:20px;color:#245770}@media(max-width:760px){.layout{display:block}}@media print{details{display:block}}'''
toc=''.join(f'<a href="#p{i}"><span>{i:02d}</span>{html.escape(t)}</a>' for i,(t,_) in enumerate(chapters,1))
header='''<header><div class="eyebrow">CoVT 项目 · 第一篇 proposal 中文整合版</div><h1>模型画出的深度图，<br>真的参与了它的回答吗？</h1><p class="title-lead">从 CoVT 原理到已完成实验，再到下一步研究设计</p><p class="meta">整合基线：2026 年 9 月 17 日 proposal；实验记录截至 2026 年 9 月 18 日</p><p>面向不了解视觉语言模型的读者。先讲问题与意义，再解释模型和数据，之后逐步阅读实际实验与后续计划。图片全部内嵌，可离线阅读。</p><p class="conclusion"><strong>当前结论：</strong>原生深度图已恢复；固定状态仍能读出很多正确关系；指定内部状态替换尚未控制文字答案。信息路径定位与修复训练还未完成。</p><p>原 proposal 的假设、已观察结果与未来计划在各节分别标明。本文不把正确可视化当作模型已经使用相应信息的证明。</p></header>'''
doc_html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CoVT 中文整合 Proposal：原理、实验与计划</title><style>'+css+'</style></head><body><div class="layout"><nav aria-label="目录"><h2>阅读目录</h2>'+toc+'</nav><main id="main">'+header+''.join(f'<section id="p{i}"><div class="section-number">{i:02d}</div><h2>{html.escape(t)}</h2>{body}</section>' for i,(t,body) in enumerate(chapters,1))+'</main></div></body></html>'
html_path=OUT/'CoVT_研究计划与实验_中文整合版.html';html_path.write_text(doc_html,encoding='utf8')

# PDF from the exact same chapter content.
pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('CNB','C:/Windows/Fonts/msyhbd.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('CN',normal='CN',bold='CNB',italic='CN',boldItalic='CNB')
W,H=A4;M=43;CW=W-2*M
styles={
 'body':ParagraphStyle('Body',fontName='CN',fontSize=10.2,leading=16.6,spaceAfter=8,wordWrap='CJK',textColor=colors.HexColor('#233747')),
 'h1':ParagraphStyle('Chapter',fontName='CNB',fontSize=19,leading=27,spaceAfter=15,wordWrap='CJK',textColor=colors.HexColor('#17324d')),
 'h3':ParagraphStyle('Subheading',fontName='CNB',fontSize=11.5,leading=18,spaceBefore=9,spaceAfter=7,keepWithNext=True,textColor=colors.HexColor('#245770')),
 'caption':ParagraphStyle('Caption',fontName='CN',fontSize=8.3,leading=12.5,spaceAfter=11,wordWrap='CJK',textColor=colors.HexColor('#566879')),
 'conclusion':ParagraphStyle('Conclusion',fontName='CNB',fontSize=10.1,leading=16.5,spaceBefore=6,spaceAfter=11,wordWrap='CJK',textColor=colors.HexColor('#176f69')),
 'cell':ParagraphStyle('Cell',fontName='CN',fontSize=8.7,leading=13,wordWrap='CJK'),
 'th':ParagraphStyle('TableHead',fontName='CNB',fontSize=8.7,leading=13,wordWrap='CJK',textColor=colors.white),
 'toc':ParagraphStyle('TOCEntry',fontName='CN',fontSize=9.4,leading=14,spaceBefore=2,wordWrap='CJK'),
}
def inline(e):
    s=html.escape(e.text or '')
    for c in e:
        t=c.tag
        if t=='br':v='<br/>'
        elif t in ['strong','b']:v='<b>'+inline(c)+'</b>'
        elif t in ['em','i']:v='<i>'+inline(c)+'</i>'
        elif t in ['sub','sup']:v=f'<{t}>'+inline(c)+f'</{t}>'
        elif t=='a' and c.get('href','').startswith('https://'):v='<a href="'+html.escape(c.get('href'),quote=True)+'" color="#176b9a">'+inline(c)+'</a>'
        else:v=inline(c)
        s+=v+html.escape(c.tail or '')
    return s
def paragraph(e,style='body',prefix=''):return Paragraph(prefix+inline(e),styles[style])
def imageflow(e,maxheight=285):
    src=e.get('src','');assert src.startswith('data:image/png;base64,')
    raw=base64.b64decode(src.split(',',1)[1]);im=PILImage.open(io.BytesIO(raw));w,h=im.size
    width=min(CW,maxheight*w/h);height=width*h/w
    return Image(io.BytesIO(raw),width=width,height=height,hAlign='CENTER')
def flows(e):
    out=[]
    if e.tag in ['p','h3','h4','summary']:
        style='h3' if e.tag!='p' else ('conclusion' if 'conclusion' in e.get('class','') else 'body')
        if e.text_content().strip():out.append(paragraph(e,style))
    elif e.tag=='div' and 'formula' in e.get('class',''):
        out.append(paragraph(e));out.append(Spacer(1,5))
    elif e.tag=='div' and e.get('class')=='flow':
        cells=[]
        for c in e:
            cells.append(Paragraph('<br/>'.join(inline(x) for x in c),styles['cell']))
        t=Table([cells],colWidths=[CW/len(cells)]*len(cells))
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#eef5f7')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)]))
        out.extend([t,Spacer(1,10)])
    elif e.tag=='figure':
        children=[]
        for im in e.xpath('./img'):children.append(imageflow(im))
        children.append(Spacer(1,5))
        for cap in e.xpath('./figcaption'):children.append(paragraph(cap,'caption'))
        out.append(KeepTogether(children))
    elif e.tag=='table':
        rows=e.xpath('./thead/tr|./tbody/tr|./tr');data=[]
        for row in rows:
            data.append([paragraph(c,'th' if c.tag=='th' else 'cell') for c in row if c.tag in ['td','th']])
        n=len(data[0]); widths=[CW/n]*n
        if n==3:widths=[CW*.24,CW*.38,CW*.38]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#24475e')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0f5f8')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        out.extend([t,Spacer(1,11)])
    elif e.tag in ['ol','ul']:
        for i,c in enumerate(e,1):
            if c.tag=='li':out.append(paragraph(c,prefix=f'{i}. ' if e.tag=='ol' else '• '))
    elif e.tag=='img':out.append(imageflow(e,200))
    elif e.tag in ['script','style']:pass
    else:
        if e.text and e.text.strip():out.append(Paragraph(html.escape(e.text),styles['body']))
        for c in e:out.extend(flows(c))
    return out
class ReportDoc(SimpleDocTemplate):
    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and f.style.name=='Chapter':
            key=getattr(f,'bookmark',None)
            if key:
                self.canv.bookmarkPage(key);self.canv.addOutlineEntry(f.getPlainText(),key,0,False)
                self.notify('TOCEntry',(0,f.getPlainText(),self.page,key))
def footer(c,d):
    c.saveState();c.setStrokeColor(colors.HexColor('#dce4e9'));c.line(M,38,W-M,38)
    c.setFont('CN',8);c.setFillColor(colors.HexColor('#617080'));c.drawString(M,25,'CoVT 项目 · 原理、研究计划与已完成实验 · 中文整合版')
    c.drawRightString(W-M,25,str(d.page));c.restoreState()
story=[Paragraph('模型画出的深度图，<br/>真的参与了它的回答吗？',styles['h1']),Spacer(1,8)]
cover=LH.fragment_fromstring(header,create_parent='div')
for p in cover.xpath('.//p'):story.append(paragraph(p,'conclusion' if 'conclusion' in p.get('class','') else 'body'))
story.append(Spacer(1,20));story.append(Paragraph('本版结构：问题与意义 → CoVT 和专家原理 → 数据与测量 → 七组已完成实验 → 原计划进展 → 下一步与证据边界。',styles['body']))
story.append(PageBreak());story.append(Paragraph('阅读目录',styles['h1']))
toc_pdf=TableOfContents();toc_pdf.levelStyles=[styles['toc']];story.append(toc_pdf)
for i,(title,body) in enumerate(chapters,1):
    story.append(PageBreak());h=Paragraph(f'{i:02d} / {html.escape(title)}',styles['h1']);h.bookmark=f'p{i}';story.append(h)
    story.extend(flows(LH.fragment_fromstring(body,create_parent='div')))
pdf_path=OUT/'CoVT_研究计划与实验_中文整合版.pdf'
doc=ReportDoc(str(pdf_path),pagesize=A4,rightMargin=M,leftMargin=M,topMargin=42,bottomMargin=53,title='CoVT：视觉思维的可视化与答案关系——中文整合研究计划',author='CoVT Research')
doc.multiBuild(story,onFirstPage=footer,onLaterPages=footer)
sources=[SOURCE,PROPOSALS/'dual_readout_proposal.tex',PROPOSALS/'dual_readout_references.bib',ROOT/'output/html/paper_figures/provenance.json']
(OUT/'integration_sources.json').write_text(json.dumps({'scope':'First proposal; independent second proposal remains separate','chapters':[t for t,_ in chapters],'figures':figure_number,'new_experiments':False,'sources':[{'path':str(p.relative_to(ROOT.parent)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],'pdf_sha256':hashlib.sha256(pdf_path.read_bytes()).hexdigest()},ensure_ascii=False,indent=2),encoding='utf8')
(OUT/'README.md').write_text('# 中文整合研究计划\n\n以第一篇 dual_readout_proposal 为主线，整合入门原理、原论文插图、已完成实验与更新后的计划。原英文 proposal 保持原样。\n\n- CoVT_研究计划与实验_中文整合版.html：离线阅读版，所有图像内嵌。\n- CoVT_研究计划与实验_中文整合版.pdf：打印与分享版，正文与 HTML 使用同一来源。\n- integration_sources.json：原稿、图文来源及校验值。\n\n重建：用含 reportlab、lxml、Pillow 的 Python 执行 covt-pilot/scripts/build_integrated_proposal.py；不运行模型实验。\n',encoding='utf8')
print(json.dumps({'chapters':len(chapters),'figures':figure_number,'html':str(html_path),'pdf':str(pdf_path)},ensure_ascii=False))
