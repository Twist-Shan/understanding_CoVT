"""Add a reader introduction using attributed figures from the original paper."""
from pathlib import Path
import base64,hashlib,json,re,runpy
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[1]
b=runpy.run_path(str(ROOT/'scripts/build_dataset_report.py'))
OUT=ROOT/'output/html'
FIG=OUT/'paper_figures';FIG.mkdir(exist_ok=True)
pdf=ROOT/'docs/references/covt-paper.pdf'
doc=pdfium.PdfDocument(str(pdf))
url='https://wakalsprojectpage.github.io/covt-website/static/pdf/paper.pdf'
specs=[(1,0,(.093,.307,.91,.453)),(2,1,(.093,.09,.91,.256)),(3,3,(.093,.09,.91,.33))]
records=[]
def paperfig(n,caption):
    data=base64.b64encode((FIG/f'covt_figure_{n}.png').read_bytes()).decode()
    return f'<figure class="paper-figure"><img src="data:image/png;base64,{data}" alt="CoVT 原论文图 {n}：{caption}" loading="lazy"><figcaption><strong>原论文图 {n}。</strong>{caption} 图源：Qin 等，<a href="{url}">Chain-of-Visual-Thought 原论文</a>，版权归原作者。本图从论文页面裁出，未改动图内内容；无数值坐标轴，方框和箭头表示组件与流程。它说明原论文的方法，不能当作本项目的实验结果。</figcaption></figure>'
for n,page,box in specs:
    im=doc[page].render(scale=3).to_pil().convert('RGB');w,h=im.size
    crop=im.crop(tuple(round(v*(w if i%2==0 else h)) for i,v in enumerate(box)))
    dest=FIG/f'covt_figure_{n}.png';crop.save(dest)
    records.append({'figure':n,'pdf_page':page+1,'crop_normalized':box,'image':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})

intro='''<section id="start-name"><div class="eyebrow">从这里开始 · 先不用看实验数字</div><h2>CoVT 是什么？</h2><p><strong>论文方法叫 CoVT，全称 Chain-of-Visual-Thought，中文可以叫“视觉思维链”。</strong>本报告研究 CoVT 的公开深度版本，项目文件夹也统一使用 CoVT。</p><p>一句话理解：<strong>CoVT 是一种训练看图问答模型的方法，希望模型在回答前先形成与图像内容有关的内部表示，再利用这些表示回答。</strong>“思维”是方法名称，不表示模型具有人的意识或主观想象。</p><h3>先从普通问答开始</h3><p>给模型一张图片，问“A 和 B 哪个更近？”，模型接收图片和问题，经过计算，生成 A 或 B。能同时处理图片和文字的模型，叫视觉语言模型。我们使用的 CoVT 深度版本以 Qwen2.5-VL 为基础。</p><h3>再理解文字思维链</h3><p>模型也可以先写：“B 挡住了 A 的一部分，所以 B 可能更近”，再给答案。这种把中间步骤写成文字的方式，叫文字思维链。这个句子只是说明性例子，不是本项目某次运行的原始输出。</p><h3>CoVT 多提出了什么？</h3><p>它希望中间表示除了文字，还包含表达深度、物体区域等视觉信息的数值。这样的数值可以参与后续计算，也可以借助额外解码模块转成可查看的结果。我们要检查的正是：这些数值实际上起了什么作用。</p><p class="conclusion"><strong>先记住：</strong>CoVT 指一种让模型在中间生成视觉表示的方法；我们研究的是这套方法的公开深度版本。</p></section>'''

intro+='''<section id="start-figure1"><h2>读原论文图 1：文字和视觉表示怎样出现在同一段生成中？</h2><p>先看整体，不用记模型参数。从左到右依次看输入图片、带颜色的小方块，以及最后的答案。</p>'''+paperfig(1,'总览图。蓝色方块表示文字 token，黄色方块表示视觉 token，粉色标记界定中间生成段，灰色方块表示最后的文字回答。')+'''<ol class="steps"><li><strong>左边的图片和问题：</strong>是模型的输入。图里的示例问图片中有多少朵云。</li><li><strong>底部的 VLM：</strong>就是负责处理图片、问题并逐步生成输出的视觉语言模型。</li><li><strong>蓝色方块 text tokens：</strong>表示文字生成单位。token 可以是一个字、一个词的一部分或特殊标记，不能一律理解为一个完整词。</li><li><strong>黄色方块 visual tokens：</strong>表示与视觉信息有关的连续数值表示。它们本身不是几张缩小的图片。</li><li><strong>右边的 Answers：</strong>是最终的文字答案。论文希望前面的视觉表示能够帮助这一步。</li></ol><p>“连续”可以先理解为内部数值能取许多实数值，例如 0.12、-0.37。相对地，输出文字需要从有限的词表里选择 token。图中箭头表达计算和生成关系，不能仅凭箭头就认定某个表示对答案有因果作用。</p><p class="conclusion"><strong>先记住：</strong>视觉思维不是一定要在屏幕上画一张图；核心是模型内部存在一组被训练来表达视觉信息的数值。</p></section>'''

intro+='''<section id="start-figure2"><h2>读原论文图 2：这些数值怎样与我们能看的图联系起来？</h2><p>原论文给出三个示例。先读每个示例的问题，再看中间的表示和可选的解码结果。这里展示的是论文示例，不是我们的球体数据。</p>'''+paperfig(2,'三个示例从左到右为计数、相对深度和场景理解。粉色表示分割 token，紫色表示深度 token；蓝色 Decoder 方框是可选解码器。不同图的配色不表示统一的数值尺度。')+'''<ol class="steps"><li><strong>左边：计数。</strong>问题问有多少完整水果。分割结果标出图像中的物体区域，使人能看到视觉表示对应的区域信息。</li><li><strong>中间：比较远近。</strong>输入图片有两个标记点。深度表示可以解码成深度图，而文字分支输出哪个点更近。</li><li><strong>右边：理解场景。</strong>示例把分割与深度表示放在同一生成过程中，用于回答床后墙面是否挂画的问题。</li></ol><p>重点是图里的 <strong>Decoder (optional)</strong>，意思是“解码器，可选”。原方法允许模型只生成内部视觉表示和文字答案，不必每次都把可视化图画出来。我们恢复解码器，是为了检查这些表示。</p><p>在我们实际恢复的深度分支中，画图还需要深度专家提供当前图片的特征。因此，展示出一张正确的深度图，并不能单独证明四个状态已经独立保存了完整深度信息。后文会把这一计算展开。</p><p class="conclusion"><strong>先记住：</strong>内部视觉表示、解码后的可视化图、最后的文字答案，是三个不同对象，需要分别检查。</p></section>'''

intro+='''<section id="start-figure3"><h2>读原论文图 3：怎样让这些内部数值学到视觉信息？</h2><p>一个位置被叫作“深度 token”，不代表它天然懂深度。训练要给这些数值施加要求。图 3 画的是训练过程；初读只看中间蓝色的深度分支，再看最右侧的文字回答分支。</p>'''+paperfig(3,'训练流程图。中部是多种视觉任务的监督，右侧是文字答案监督；深度分支使用 DepthAny Encoder、Projection 和 BMM Layer。箭头表示信息或监督的连接，不是物理距离。')+'''<h3>先读深度分支的三个组件</h3><ul><li><strong>DepthAny Encoder：</strong>深度专家的图像编码器。它处理图片，为图像各位置提供特征。</li><li><strong>Projection：</strong>把 CoVT 内部状态转换成能与专家特征配合计算的向量。</li><li><strong>BMM Layer：</strong>批量矩阵乘法。直观上，是让向量与各位置的图像特征逐项相乘、求和，得到一张分数图。后文有两维手算例子。</li></ul><h3>再读训练误差</h3><p>图中 GT 指训练目标，Loss 指预测与目标的差异。深度分支比较重建深度与训练深度目标，文字分支比较生成文字与训练文字目标。训练根据误差调整参数，希望视觉表示和回答逐渐符合要求。训练目标不应与本项目的几何真值混淆：本项目的远近标签来自我们生成的三维场景。</p><p>这里讲的是<strong>原作者怎样训练 CoVT</strong>。我们没有重做这些训练；我们加载公开权重，恢复深度解码分支，然后做对照与干预。</p><p class="conclusion"><strong>先记住：</strong>视觉状态的含义来自训练要求。训练要求存在，并不保证每次回答都依赖这些状态，所以还需要后面的实验。</p></section>'''

intro+='''<section id="start-four"><h2>回到我们的版本：四个深度状态究竟是什么？</h2><p>我们使用 CoVT-7B-depth。正常生成时，它会出现四个专门的深度标记。每个标记所在位置有一组内部数值，我们把它们叫作“四个深度状态”。每组在当前模型中有 3584 个数。</p><p><strong>四个状态不是四个物体、四个距离，也不是四张现成的深度图。</strong>在恢复的解码器里，它们经过转换后，分别配合深度专家第 4、11、17、23 层的特征，生成四张分数图，再取平均。编号从 0 开始，指深度专家的层。</p><p>代码规定了这些对应关系，但没有规定“第一个专门表示近处、第二个专门表示远处”。每个状态具体学到了什么，需要通过分析和实验确认。</p><div class="walk"><b>读后面的报告，只需带着两个问题</b><p>① 画深度图时，当前图片自己的四个状态是否必需？<br>② 生成文字答案时，修改这些状态是否会改变回答？</p></div><p>第一个问题检查可视化分支，第二个问题检查答案生成。后面的实验会分别回答这两个问题。先看数据和模型原理，再看实验，不必先记住 E0、E1 或所有数值。</p><p class="conclusion"><strong>入门部分的结论：</strong>CoVT 希望通过学习视觉中间表示来帮助回答；我们的工作是检查已发布模型中的这些表示是否起到了预期作用。</p></section>'''

document=b['document']
document=document.replace('</header>','</header>'+intro,1)
nav=''.join(f'<a href="#{id}">{label}</a>' for id,label in [('start-name','入门 1：CoVT 的含义'),('start-figure1','入门 2：原论文图 1，总体过程'),('start-figure2','入门 3：原论文图 2，具体例子'),('start-figure3','入门 4：原论文图 3，怎样训练'),('start-four','入门 5：我们的四个深度状态')])
document=document.replace('<h2>阅读顺序</h2>','<h2>先理解方法，再读实验</h2>'+nav,1)
document=document.replace('先弄清模型怎么工作，<br>再看我们的实验改了什么','CoVT 到底是什么？<br>从原论文图开始读懂我们的实验',1)
document=document.replace('前五节讲清 CoVT、深度专家、四个状态和深度图的关系。后七组实验每次只问一个问题，按“先准备什么、改什么、看什么、结果是什么”解释。无需先读公式。','先读新增的五节入门讲解，用原论文图理解 CoVT 的名称、生成过程与训练目标。随后是数据展示、深度专家的原理和七组实验。无需先读公式。',1)
document=document.replace('</style>','.paper-figure{padding:12px;background:#fff;border:1px solid #dce4e9}.paper-figure img{border:0}.paper-figure figcaption{padding:0 8px}.paper-figure a{overflow-wrap:anywhere}</style>',1)
b['b']['target'].write_text(document,encoding='utf8')
manifest={'source_url':url,'source_pdf':str(pdf.relative_to(ROOT)),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'attribution':'Yiming Qin et al., Chain-of-Visual-Thought. Figure rights belong to the original authors.','figures':records,'builder':'scripts/build_paper_intro_report.py','new_model_experiments':False}
(FIG/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print('Added five introductory sections and three attributed original paper figures.')
