"""Publication layout from existing archived results only. No model execution."""
from pathlib import Path
import json, math, hashlib, html
import numpy as np
from PIL import Image as PILImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'; TMP=ROOT/'tmp/pdfs'; FIG=OUT/'figures'
for p in (OUT,TMP,FIG): p.mkdir(parents=True,exist_ok=True)
BF=ROOT/'runs/colab-20260918-backup/runs/colab-e0-20260918-032255'
FP=ROOT/'runs/colab-20260918-followup/runs/followup-fp32-20260918-043002'
AS=ROOT/'runs/colab-20260918-answer-shift/extracted/runs'
ANS=AS/'answer-diagnostic-20260918-050133'; SH=AS/'shift-validation-20260918-050713'
BAL=ROOT/'runs/colab-20260918-balanced-intervention/extracted/runs/balanced-intervention-20260918-090000'
LS=ROOT/'runs/colab-20260918-layer-scan/extracted/runs/layer-scan-20260918-093053'
BL=ROOT/'runs/colab-20260918-block-intervention/extracted/runs/block-intervention-20260918-100708'
def J(p): return json.loads(p.read_text(encoding='utf-8'))
def JL(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
bf=JL(BF/'e0/scored.jsonl'); fp=JL(FP/'e0/scored.jsonl'); e1=JL(FP/'e1/results.jsonl')
ans=JL(ANS/'results.jsonl'); bal=JL(BAL/'native.jsonl'); frozen=JL(BAL/'frozen.jsonl')
ls=J(LS/'final_audit.json'); block=J(BL/'final_audit.json'); blockrows=JL(BL/'results.jsonl')
assert len(bal)==32 and len(blockrows)==200 and block['identity_exact']==100
assert all(r['full_token_sequence_changes']==0 for r in block['details'])

FONT='C:/Windows/Fonts/msyh.ttc'; BOLD='C:/Windows/Fonts/msyhbd.ttc'
pdfmetrics.registerFont(TTFont('CN',FONT,subfontIndex=0))
pdfmetrics.registerFont(TTFont('CNB',BOLD,subfontIndex=0))
pdfmetrics.registerFontFamily('CN',normal='CN',bold='CNB',italic='CN',boldItalic='CNB')
fontManager.addfont(FONT)
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':10,'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
NAVY='#17324D'; BLUE='#2274A5'; ORANGE='#D78335'; TEAL='#26877D'; GRAY='#63758A'
def savefig(name,fig):
    path=FIG/(name+'.png'); fig.savefig(path,bbox_inches='tight',facecolor='white'); plt.close(fig); return path
def image_axes(ax,path,title,coords=False):
    ax.imshow(PILImage.open(path)); ax.set_title(title,fontsize=10)
    if coords: ax.set_xlabel('x（像素，向右）'); ax.set_ylabel('y（像素，向下）'); ax.set_xticks([0,168,335]); ax.set_yticks([0,168,335])
    else: ax.axis('off')

# Figures are report visualizations of saved results, never new scene/model experiments.
fig,axs=plt.subplots(1,3,figsize=(10,3.1),layout='constrained')
case=next(r for r in bf if r['id']=='scene_00002_base')
cache=np.load(BF/'e0'/case['artifact'])
image_axes(axs[0],BF/'scenes'/case['image'],'原始图像：真值 B，语言答案 A',True)
for ax,key,title in zip(axs[1:],['decoded_map','expert_map'],['原生读出：关系 B','专家读出：关系 B']):
    im=ax.imshow(cache[key],cmap='viridis'); ax.set_title(title); ax.set_xlabel('x（像素）'); ax.set_ylabel('y（像素）')
    cb=fig.colorbar(im,ax=ax,shrink=.78); cb.set_label('原始读出分数（非米）',fontsize=8)
casefig=savefig('01_native_case',fig)

fig,axs=plt.subplots(1,2,figsize=(8.8,3.1),layout='constrained')
image_axes(axs[0],BF/'scenes'/case['image'],'采样位置与原始输入',True)
for (x,y),letter in zip(case['points'],'AB'):
    for radius in (5,9): axs[0].add_patch(plt.Circle((x,y),radius,fill=False,color='#FFBF00',lw=1))
    axs[0].annotate(letter,(x,y),xytext=(x+15,y-22),color='#E39A00',weight='bold')
x,y=case['points'][0]; image_axes(axs[1],BF/'scenes'/case['image'],'A 点局部：只采样半径 5-9 像素环带',True)
axs[1].set_xlim(x-16,x+16); axs[1].set_ylim(y+16,y-16); axs[1].set_xticks([x-10,x,x+10]); axs[1].set_yticks([y-10,y,y+10])
for radius in (5,9): axs[1].add_patch(plt.Circle((x,y),radius,fill=False,color='#FFBF00',lw=2))
annulusfig=savefig('02_annulus',fig)

fig,axs=plt.subplots(2,2,figsize=(7.7,5.8),layout='constrained')
maps=[np.load(BF/f'e1/scene_00002_depth_swap_H{u}_F{v}.npy') for u in range(2) for v in range(2)]
lo=min(float(x.min()) for x in maps); hi=max(float(x.max()) for x in maps)
bfe1=JL(BF/'e1/results.jsonl')[0]
for idx,(ax,arr) in enumerate(zip(axs.flat,maps)):
    u,v=divmod(idx,2); im=ax.imshow(arr,cmap='viridis',vmin=lo,vmax=hi)
    ax.set_title(f'H{u} + F{v}；r{u}{v}={bfe1[f"r{u}{v}"]:.4f}'); ax.set_xlabel('x（原生 256 网格像素）'); ax.set_ylabel('y（像素）')
fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.8,label='共享色标：原始读出分数（非米）')
e1mapfig=savefig('03_e1_maps',fig)

fig,ax=plt.subplots(figsize=(8.4,3.0),layout='constrained')
x=np.arange(4); vals=[[-.375,-.35541153,.03961563],[0,.03059196,.03966713]]
labels=['BF16','BF16 + FP32 输出头','FP32']
for j,(v,c,l) in enumerate(zip(vals,[BLUE,ORANGE],['cached','uncached'])): ax.bar(np.arange(3)+(j-.5)*.32,v,.32,label=l,color=c)
ax.axhline(0,color=GRAY,lw=.8); ax.set_xticks(range(3),labels); ax.set_ylabel('logit(A) - logit(B)'); ax.legend(frameon=False)
precisionfig=savefig('04_precision',fig)

fig,axs=plt.subplots(1,2,figsize=(9.1,3.0),layout='constrained')
for ax,change,title in zip(axs,['depth_swap','nuisance'],['交换远近关系','只改颜色']):
    rr=[r for r in e1 if r['change']==change]; xx=np.arange(len(rr))
    ax.plot(xx,[abs(r['state_effect_Fa']) for r in rr],'o-',color=BLUE,label='|状态效应|')
    ax.plot(xx,[abs(r['expert_effect_Ha']) for r in rr],'s-',color=ORANGE,label='|专家特征效应|')
    ax.set_yscale('log'); ax.set_xlabel('场景族编号（scene_00002 起）'); ax.set_xticks(xx,[str(i+2) for i in xx]); ax.set_ylabel('归一化关系分数变化绝对值（对数轴）',fontsize=8); ax.set_title(title); ax.legend(fontsize=8,frameon=False)
e1effectfig=savefig('05_e1_effects',fig)

fig,ax=plt.subplots(figsize=(8.4,3.0),layout='constrained')
names=['原生','固定均值','固定示例','跨图状态','位置乱序','零 hidden','零投影']; values=[100,100,100,100,58.125,0,0]
ax.bar(range(7),values,color=[BLUE]*4+[ORANGE,GRAY,GRAY]); ax.set_xticks(range(7),names); ax.set_ylim(0,110); ax.set_ylabel('深度关系正确率（%）')
for i,v in enumerate(values): ax.text(i,v+2,f'{v:g}',ha='center',fontsize=9)
controlfig=savefig('06_state_controls',fig)

tasks=['original_depth','plain_near','plain_far','label_left','apparent_size','side_near','large_plain_near','large_label_left']
tasknames=['技术问法','直白问近','直白问远','识别左标签','比较面积','问近答左右','放大字母问近','放大字母识别']
modelnames=sorted(set(r['model'] for r in ans)); base_model=next(m for m in modelnames if m!='covt')
fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
for j,(model,color,name) in enumerate([('covt',BLUE,'CoVT'),(base_model,ORANGE,'Qwen 基座')]):
    vv=[sum(r['correct'] for r in ans if r['model']==model and r['task']==t) for t in tasks]
    bars=ax.barh(np.arange(8)+(j-.5)*.34,vv,.34,label=name,color=color)
    ax.bar_label(bars,fmt='%g',padding=2,fontsize=8)
ax.set_yticks(range(8),tasknames); ax.invert_yaxis(); ax.set_xlim(0,18.2); ax.set_xticks([0,4,8,12,16]); ax.set_xlabel('正确图像数（每项分母 16；8 个场景族）'); ax.legend(frameon=False,loc='lower right')
answerfig=savefig('07_answer_tasks',fig)

fig,axs=plt.subplots(1,2,figsize=(7.6,3.0),layout='constrained')
for ax,variant in zip(axs,['base','label_swap']):
    rr=next(r for r in ans if r['model']=='covt' and r['source_id']==f'scene_00002_{variant}' and r['task']=='plain_near')
    # Archived cases retain each source rendering under stable filenames.
    image_axes(ax,ANS/'cases'/f'scene_00002_{variant}_original.png',f'{variant}：真值 {rr["label"]}，直白问近答案 {rr["answer"]}')
answercasefig=savefig('08_answer_pair',fig)

shift_manifest=JL(SH/'scenes/manifest.jsonl')
strata=['size_variation','ellipsoids','boxes','partial_occlusion','camera_variation']
stratanames=['尺寸变化','椭球','盒体','部分遮挡','相机变化']
fig,axs=plt.subplots(1,5,figsize=(10.5,2.25),layout='constrained')
for ax,s,n in zip(axs,strata,stratanames):
    r=next(r for r in shift_manifest if r['stratum']==s and r['variant']=='base')
    image_axes(ax,SH/'scenes'/r['image'],f'{n}\n{r["id"]}',False)
shiftfig=savefig('09_shift_examples',fig)

fig,axs=plt.subplots(2,2,figsize=(7.4,5.5),layout='constrained')
for ax,r in zip(axs.flat,[r for r in bal if r['family']=='balanced_000']):
    image_axes(ax,BAL/'scenes'/r['image'],f'd{r["depth_order"]}_s{r["size_order"]}：真值 {r["label"]}，答案 {r["answer"]}\n'+('大小一致' if r['cue']=='aligned' else '大小冲突'))
balancedfig=savefig('10_balanced_family',fig)

layers=[0,4,8,14,20,24,27]
fig,axs=plt.subplots(1,2,figsize=(9,3.05),layout='constrained')
for ax,c,title,color in zip(axs,['opposite_depth','zero'],['自然相反深度供体','置零（有格式损坏）'],[BLUE,ORANGE]):
    vals=[next(r for r in ls['details'] if r['layer']==k and r['condition']==c)['max_delta_p_A']*100 for k in layers]
    ax.plot(layers,vals,'o-',color=color); ax.set_xticks(layers); ax.set_xlabel('层号（从 0 开始）'); ax.set_ylabel('最大 |ΔP(A | A或B)|（百分点）',fontsize=8); ax.set_ylim(bottom=0); ax.set_title(title)
layerfig=savefig('11_layer_probability',fig)

stages=['early','middle','late','all_layers','all_layers_thought_prefix']
stage_names=['0-8层\n4位置','9-17层\n4位置','18-27层\n4位置','0-27层\n4位置','0-27层\n15位置']
fig,ax=plt.subplots(figsize=(8.6,3.1),layout='constrained')
for j,(c,color,label) in enumerate([('opposite_depth',BLUE,'相反深度'),('same_depth',ORANGE,'同深度')]):
    vals=[next(r for r in block['details'] if r['stage']==s and r['condition']==c)['max_abs_delta_p_A']*100 for s in stages]
    bars=ax.bar(np.arange(5)+(j-.5)*.34,vals,.34,color=color,label=label)
    ax.bar_label(bars,fmt='%.4f',fontsize=8,padding=3)
ax.set_xticks(range(5),stage_names); ax.set_ylim(0,.061); ax.set_ylabel('最大条件概率变化（百分点）'); ax.legend(frameon=False)
blockfig=savefig('12_block_probability',fig)

best=max((r for r in blockrows if r['stage']=='all_layers_thought_prefix' and r['condition']=='opposite_depth'),key=lambda r:r['abs_delta_p_A'])
fig,axs=plt.subplots(1,2,figsize=(7.4,3),layout='constrained')
for ax,ident,title in zip(axs,[best['id'],best['donor']],['目标图像','相反深度供体']):
    r=next(r for r in bal if r['id']==ident)
    image_axes(ax,BAL/'scenes'/r['image'],f'{title}：{ident}\n真值 {r["label"]}，原答案 {r["answer"]}')
blockcasefig=savefig('13_block_pair',fig)

W,H=A4; M=43; CW=W-2*M
styles={
 'body':ParagraphStyle('body',fontName='CN',fontSize=10.3,leading=16,spaceAfter=9,wordWrap='CJK',textColor=colors.HexColor('#26384A')),
 'small':ParagraphStyle('small',fontName='CN',fontSize=8.4,leading=12,spaceAfter=6,wordWrap='CJK',textColor=colors.HexColor(GRAY)),
 'caption':ParagraphStyle('caption',fontName='CN',fontSize=8.6,leading=12.5,spaceAfter=10,wordWrap='CJK',textColor=colors.HexColor('#4D6073')),
 'title':ParagraphStyle('title',fontName='CNB',fontSize=20,leading=29,spaceAfter=15,textColor=colors.HexColor(NAVY),wordWrap='CJK'),
 'sub':ParagraphStyle('sub',fontName='CNB',fontSize=11.2,leading=17,spaceBefore=5,spaceAfter=6,textColor=colors.HexColor(BLUE)),
 'table':ParagraphStyle('table',fontName='CN',fontSize=8.5,leading=12,wordWrap='CJK'),
 'th':ParagraphStyle('th',fontName='CNB',fontSize=8.5,leading=12,wordWrap='CJK',textColor=colors.white),
 'conclusion':ParagraphStyle('conclusion',fontName='CNB',fontSize=10.2,leading=16,spaceAfter=9,wordWrap='CJK',textColor=colors.HexColor(TEAL)),
}
story=[]; sections=[]; textual=[]
def p(txt,kind='body'):
    story.append(Paragraph(txt,styles[kind])); textual.append(txt)
def sec(title,source=None):
    if sections: story.append(PageBreak())
    sections.append(title); p(f'{len(sections):02d} / {title}','title')
    if source: p('证据来源：'+source,'small')
def para(label,txt,conclusion=None):
    p(f'<b>{label}</b>　{txt}'+(f'<br/><font color="{TEAL}"><b>本段结论：</b>{conclusion}</font>' if conclusion else ''))
def conclude(txt): p('当前结论｜'+txt,'conclusion')
def table(headers,rows,widths=None):
    data=[[Paragraph(html.escape(str(x)),styles['th']) for x in headers]]+[[Paragraph(html.escape(str(x)),styles['table']) for x in row] for row in rows]
    widths=[CW*x/sum(widths) for x in widths] if widths else [CW/len(headers)]*len(headers)
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F5F8'),colors.white]),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor(NAVY))]))
    story.extend([t,Spacer(1,10)])
def figure(path,caption,maxheight=250):
    w,h=PILImage.open(path).size; width=min(CW,maxheight*w/h); height=width*h/w
    story.append(Image(str(path),width=width,height=height,hAlign='CENTER')); story.append(Spacer(1,5)); p(caption,'caption')
def formula(txt):
    t=Table([[Paragraph(txt,styles['body'])]],colWidths=[CW]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#EEF4F7')),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),2)]));story.extend([t,Spacer(1,9)])

sec('CoVT 原生深度读出与答案干预实验报告')
p('阶段性研究记录 · 截至 2026 年 9 月 18 日','sub')
p('从解码恢复、几何与行为对照，到单层、连续层和全层状态替换。本报告整理已完成的实验；不新增模型推理，不将计划中的验证写成既有结果。')
table(['核心问题','当前直接证据','结论范围'],[
 ['原生深度分支能否恢复？','公开参数恢复；算子对照误差 0；FP32 E0 为 40/40 通过','当前固定版本和合成场景可运行'],
 ['深度图是否依赖图像特异的 thought 状态？','旧固定状态在新 200 图中正确 199/200；大小冲突时 14/16','状态位置结构有作用；图像特异状态必要性未建立'],
 ['深度状态能否控制语言答案？','全 28 层、15 个思考前缀位置替换仍 0/20 翻转','自然供体替换效应很小；未覆盖全部信息路径'],
 ['置零翻转能否证明语义作用？','存在提前终止、重复及 A/B 歧义','必须分开审计格式与答案内容']],[1.2,2.2,1.6])
para('阅读原则','深度关系正确率、语言答案正确率、答案格式有效性和概率变化是不同指标。样本范围和聚合方式在各节单独标明。', '当前最可靠的发现是读出分支与语言回答表现可以分离；机制原因未查明。')
para('执行与范围','模型推理、场景生成、实验统计均来自 Colab A100 记录。本次仅在本地读取既有数据、绘制报告图表并排版。所有模型保持冻结，没有进行修复训练。','本报告可用于评估下一轮实验设计，尚不构成自然图像上的性能结论。')
p('项目：CoVT / covt-pilot　　报告版本：1.0　　语言：中文','small')

sec('阅读导航与实验依赖')
table(['页码','内容','回答的问题'],[
 ['3-5','计算链、几何真值与指标定义','H、F、深度分数、正确率和概率如何计算？'],
 ['6-8','原生恢复、首轮 E0、精度诊断与 FP32 E0','测量链是否可靠，哪些失败必须保留？'],
 ['9-11','E1 四组合与末端状态对照','图像专家特征和状态分别贡献多少？'],
 ['12-13','问法、标签和基座模型诊断','低答案准确率是否只是识别或提示问题？'],
 ['14-16','新几何分布、大小捷径和平衡对照','固定状态效果能否扩展，是否依赖大小线索？'],
 ['17-19','中间层干预、七层扫描和格式审计','状态替换是否影响后续答案？'],
 ['20-21','连续层、全层和 15 位置扩展','单层或四位置覆盖是否不足？'],
 ['22-24','综合结论、复现与证据索引','能下什么结论，原始记录在哪里？']],[.7,2,2.3])
para('实验编号','E0 表示恢复与一致性检查；E1 表示末端读出四组合。后续章节按具体干预命名，不把终端解码替换与完整语言生成混为同一实验。','E1 本身不重算语言答案；只有后续共享中间层实验才测量重新生成的回答。')
para('数据关系','首轮 BF16 为 seed=0 的 12 图；FP32 为 seed=18 的 40 图。答案诊断复用后者 16 图。扩展几何为 seed=1901 的 200 图；大小平衡为 seed=1902 的 32 图。层干预沿用其 20 图子集。','各阶段分母不同，不能把跨批正确率差直接当作某一改动的收益。')
conclude('整份报告按照“先验证仪器，再分解读出，再控制混杂，最后干预答案”的证据顺序组织。')

sec('模型计算链与原生解码定义','R01；decoder.py；expert.py；固定源码快照')
para('目的与对象','H 是四个自然生成 depth pad 输入位置在语言模型末端归一化后的状态，每个向量为 3584 维；F 是当前图像经 Depth Anything V2 Large 提取的四层 patch 特征，每个 patch 为 1024 维。','H 与 F 来自不同计算路径，正确的深度图不能仅归因于 H。')
formula('图像 + 问题 → CoVT → 四个状态 H → MLP → 四个向量 T<br/>图像 → 深度专家 → 四层特征 F<br/>对应位置的 T 与 F 点积 → 双线性上采样 → 四图均值 D')
formula('T<sub>i</sub> = W<sub>2</sub> GELU(W<sub>1</sub>H<sub>i</sub> + b<sub>1</sub>) + b<sub>2</sub><br/>M<sub>i</sub>(p) = Σ<sub>c=1…1024</sub> T<sub>i,c</sub> F<sub>i,p,c</sub><br/>D = (1/4) Σ<sub>i=1…4</sub> Bilinear(M<sub>i</sub>, 256 × 256)')
para('设定','MLP 为 3584→3584→1024，使用公开 checkpoint 的四个参数张量。专家取层 [4,11,17,23]；RGB 图像先缩至 256×256，再按官方 image2tensor 预处理。当前缓存的专家网格为 37×37。点积后没有 softmax、sigmoid 或逐图归一化。','D 是模型读出分数，未标定为米；报告中的热图不表示绝对测距精度。')
para('简单例子（说明性）','若一个二维 token 为 (1,0)，两个 patch 特征为 (2,5) 与 (1,9)，点积为 2 和 1。即使 token 不变，改变 patch 特征仍会改变空间图。','固定 H 得到不同深度图在计算上完全可能，需要分别控制 H 和 F。')
para('验证与限制','重建算子在合成张量、非方形网格和不同 batch 下与固定官方实现最大误差为 0。随后 E0 才验证真实参数、专家特征与生成流程。','算子对照通过验证了重建实现；它单独不能验证模型是否使用深度状态回答。')

sec('几何真值、采样环带与深度关系分数','R02/R03 的 scenes/manifest.jsonl；metrics.py')
figure(annulusfig,'图 1｜真实归档图 scene_00002_base。横轴 x 为向右增加的图像像素，纵轴 y 为向下增加的像素；黄色圆为距标记中心 5 与 9 像素的边界。右图放大 A 点邻域，评估只取两圆间的环带，尽量避开标记中心。图片未重新生成。',200)
para('真值与样本单位','336×336 解析渲染图提供可见表面相机轴向 z 值，较小者更近。一个场景族共享基础几何，包含原图、深度交换、颜色变化、标签交换等变体。环带需位于对应可见物体内。','同族变体有关联，不能视为完全独立的重复。')
formula('d = median(D[B 环带]) - median(D[A 环带])<br/>s ∈ {−1,+1}：仅由校准族选择方向；c = median(|d|)：仅由校准样本固定尺度<br/>r = s · d / c；r &gt; 0.05 判 A，r &lt; −0.05 判 B，|r| ≤ 0.05 弃权')
para('简单例子（说明性）','若 s=−1、c=100、d=−80，则 r=0.8，判 A。若 d=−3，则 r=0.03，弃权。弃权阈值 0.05 是 pilot 工程设定，未被证明是通用决策边界。','弃权与选错方向应分开报告；阈值不能根据测试结果重新调整。')
conclude('几何标签与读出关系分数分别来自渲染真值和校准后的模型输出，两者使用不同数值单位。')

sec('统计、答案解析与干预指标','metrics.py；layer_scan.py；block_intervention.py')
formula('逐图准确率 = 正确图像数 / 该项尝试或合格图像数（见表中分母）<br/>族平均 μ = (1/G) Σ<sub>g</sub> [(1/n<sub>g</sub>) Σ<sub>j</sub> x<sub>g,j</sub>]<br/>95% 区间：以场景族均值为单位有放回抽取 G 个族，重复 2000 次，取 2.5% 与 97.5% 分位数；seed=0。')
para('聚合例子（说明性）','一族 3/3 正确，另一族 0/1 正确，逐图准确率为 75%，族平均为 50%。五次随机打乱先在族内汇总，不能把五个 seed 当成五倍独立场景。','图像分母不等时必须区分两种口径；少量族或全成功会使经验区间退化。')
formula('q = P(A | A 或 B) = exp(l<sub>A</sub>) / [exp(l<sub>A</sub>) + exp(l<sub>B</sub>)]<br/>Δq = q<sub>干预</sub> − q<sub>基线</sub>；百分点变化 = 100 × Δq<br/>定向变化：目标真值 A 时为 −Δq，目标真值 B 时为 +Δq；正值表示朝相反真值移动。')
para('概率例子（说明性）','q 从 0.8400 变成 0.8405，变化为 0.0005，即 0.05 个百分点。概率取原始答案之前的固定上下文；自由生成则从最后一个 pad 后重新开始。','固定上下文概率只能作为诊断量，不能替代完整生成的答案效果。')
para('解析与翻转','早期解析器可取答案标签或最后一个 A/B 行。发现置零异常后，增加完整标签、唯一字母与 A/B 歧义审计；多层实验预先要求一个完整答案标签且无冲突 A/B。有效翻转要求干预后答案有效且与基线不同。','无效输出不计为干净的 A↔B 翻转，同时必须保留在尝试分母中。')
para('状态与地图距离','状态距离为 ||H供体−H目标||₂ / max(||H目标||₂,10⁻¹²)，按所选位置拼接计算。地图 RMSE 为 sqrt(mean((D干预−D基线)²))，使用未缩放的原始读出单位。','距离与 RMSE 描述数值变化，不自动等同于语义变化。')

sec('E0：首轮原生恢复与真实案例','R01/R02；BF16，seed=0，3 个场景族，12 图')
para('目的与设定','检查四个自然 pad、正常结束、cached/uncached token 一致、固定序列 hidden/logits 原样写回、专家重算与解码器原样写回。前 1 族校准，后 2 族 pilot；BF16 语言模型、FP16 原生解码器、FP32 专家、greedy、最多 192 新 token。','只有全部检查通过的样本进入配对分析；失败仍计入尝试数。')
figure(casefig,'图 2｜真实案例 scene_00002_base。左图为输入；中、右图为归档的 336×336 评估地图。横纵轴均为像素；色条为各分支原始输出分数，两个色条分别缩放，不能直接比较数值大小。在本次校准方向下，较高读出值对应较近关系。真值 B，原生和专家关系均为 B，语言回答 A。',190)
table(['检查 / 指标','结果','分母范围'],[['自然 pad、结束与固定序列 identity','12/12','所有图像'],['缓存 / 无缓存 token 一致','10/12','2 张不一致保留为失败'],['全部 E0 通过','10/12','3 校准 + 7 pilot'],['深度 / 专家关系正确','7/7；7/7','合格 pilot'],['语言答案正确','2/7','合格 pilot']],[2,1,2])
para('案例与分析','A、B 表面真值 z 分别为 5.3795、3.4078，故 B 更近；原生 r=−1.17084 判 B，语言却回答 A。两个缓存失败样本的差异均在最终 A/B token。','真实深度读出与语言答案可不一致；首轮缓存问题尚需独立诊断。')
conclude('原生分支已运行并恢复出可评估关系；仅 7 个合格 pilot 图和 2 个族不能支持总体可靠性或答案因果结论。')

sec('精度诊断：缓存模式为何影响答案？','R03；cache-diagnostic-20260918-042422')
para('目的与设定','选首轮 2 个失败图和 2 个通过图。每种精度重新加载固定 checkpoint，各执行 cached、cached repeat、uncached，比较同一前缀的 token 与 A/B logits；关闭 TF32，启用确定性算法。','这是定向故障诊断，样本选择不能估计一般失败率。')
table(['模式','token 不一致','A/B 不一致','cached 重复不一致'],[['BF16','2/4','2/4','0/4'],['BF16 主体 + FP32 输出头','2/4','2/4','0/4'],['FP16','4/4','0/4','0/4'],['FP32','0/4','0/4','0/4']],[2.3,1,1,1.4])
figure(precisionfig,'图 3｜真实 nuisance 失败样本的同一答案前缀。横轴为加载精度；纵轴为 logit(A)−logit(B)，正值偏向 A、负值偏向 B；蓝色为 cached，橙色为 uncached，无误差条。纵轴零线为两标签分数相等。完整 FP32 的两柱分别为 0.03961563 与 0.03966713。',210)
para('案例与分析','BF16 的 cached 差值 −0.375，uncached 为 0；仅提升输出头后仍跨越零点。FP16 的四次 token 差异来自句号，并非答案字母。','完整 FP32 消除了本批观察到的选择差异；最后输出头不是已被单独证实的唯一原因。')
conclude('后续采用 FP32 作为工程规避。参数表示和计算精度同时改变，具体层或算子的原因未查明。')

sec('FP32 E0：扩大独立场景族','R03；followup-fp32-20260918-043002')
para('目的与设定','检验精度规避是否能支持下一轮可重复测量。seed=18，10 个族×4 变体=40 图；2 个族、8 图校准，8 个族、32 图 pilot。语言模型 FP32、eager，专家在 Colab CPU；没有训练，未放宽 E0 阈值。','这是新批次验证，不是对首轮相同数据单独改变精度的准确率比较。')
table(['检查 / 结果','数值','解释'],[['E0 全部检查、缓存 token 一致','40/40；40/40','本批无排除'],['校准 / pilot','8 / 32 张','2 / 8 个场景族'],['原生深度关系','32/32','全部正确'],['专家深度关系','32/32','全部正确'],['语言答案','8/32 = 25%','族 bootstrap 95%：9.375%-40.625%'],['深度正确、答案错误','24/32','两种指标的联合错误']],[1.8,1.4,2])
para('简单例子（说明性）','若深度图依据 A/B 邻域判定 B 更近，而文本输出 A，该图记为“深度正确、答案错误”；不能因为答案错就删除深度结果，也不能因为深度对就认为推理成功。','联合统计保留了两个分支之间的差异。')
para('分析','本次深度关系全成功使经验区间退化为 [1,1]，独立族数仍只有 8。32 图答案低正确率可能受任务和输入设定影响，具体原因未查明；后续专门进行了问法、标签和基座对照。','目前只能确认测量链在该批次通过检查，以及两个输出分支的表现显著不同。')
conclude('FP32 E0 为后续受控比较提供了可用样本；尚不能将 25% 答案准确率解释为模型普遍忽略思考状态。')

sec('E1 四组合：把状态与专家特征分开','R02 的 e1/results.jsonl 和原生地图；R03 扩展 E1')
para('目的与计算','对目标 a 和供体 b，分别读出 D(Ha,Fa)、D(Hb,Fa)、D(Ha,Fb)、D(Hb,Fb)。rᵤᵥ 是相应地图经固定校准后的关系分数；使用 Fv 对应图像的 A/B 采样位置。','交换 F 时改变的是图像特征输入；这些组合不重新生成语言答案。')
formula('状态效应 = r10−r00；专家效应 = r01−r00<br/>交互项 = r11−r01−r10+r00<br/>深度交换的定向状态效应 = o·(r10−r00)，供体真值 A 时 o=+1，B 时 o=−1。')
figure(e1mapfig,'图 4｜首轮唯一合格 E1 族 scene_00002 的真实 depth_swap 四组合。行改变 H（0=原图，1=交换图），列改变 F；四图共享色标。轴为原生 256×256 地图像素，色条是非米制原始分数。标题 r 在 336×336 评估图的标记环带上计算，因此色条不是 r。',300)
conclude('该案例中交换专家特征使关系分数变号，交换状态未变号；一个场景族只能提供局部观察。')

sec('E1 扩展：八个族上的效应量','R03；16 配对，8 个 pilot 场景族，无排除')
para('设定与例子','相同 FP32 E0 缓存配合原生解码器在 Colab GPU 运行。每族含 base/depth_swap、base/nuisance 两对。真实 scene_00002 depth_swap：r00=−1.31915，r10=−1.31977，r01=0.77762，r11=0.77713。','该例的换状态效应约 −0.000620，换特征效应约 +2.09677；数量级差异需看跨族结果。')
figure(e1effectfig,'图 5｜FP32 扩展 E1 的逐族绝对效应。横轴为场景族编号；纵轴是归一化关系分数变化绝对值，使用对数刻度。蓝线固定 Fa 换 H，橙线固定 Ha 换 F。每点为一个族，不含额外独立重复；两面板分别为深度交换和颜色变化。连线仅帮助追踪编号。',235)
table(['变化','平均 |状态效应|','平均 |专家效应|','换 H / 换 F 关系翻转'],[['depth_swap','0.000311','1.802808','0/8；8/8'],['nuisance（颜色）','0.000502','0.049131','0/8；0/8']],[1.3,1.3,1.3,1.9])
para('分析','固定特征时 H 替换的变化很小；固定 H 时深度交换特征的变化较大。小效应未设预先等价边界，部分区间不跨零也不能自动变成实际重要效应。','本实验支持读出分支中 F 的变化占主导；不提供语言答案使用 H 的证据。')
conclude('跨 8 个族复现了首轮读出分解现象；尚未证明固定 H 与自然 H 在所有输入或精度下正式等价。')

sec('末端状态对照：固定、跨图、打乱与置零','R03；controls/results.jsonl，32 pilot 图，8 个族')
para('目的与设定','固定目标图像专家特征，比较原生状态、8 个校准样本均值、校准示例、跨族图像状态、四位置无固定点乱序、零 hidden、零投影向量。随机条件五个 seed，先在族内平均。','这是末端解码分支的对照，不是共享中间层的语言生成干预。')
figure(controlfig,'图 6｜原生 FP16 解码器上的深度关系正确率。横轴是状态条件，纵轴为正确率百分比；四个蓝柱为保留位置结构的条件，橙柱为乱序，灰柱为零条件。每个固定条件 32 图；随机条件 32×5 次测量、仍为 8 个独立族。柱顶数字是汇总值，无误差条。',190)
table(['条件','弃权率','平均 |Δr|','地图 RMSE（原始单位）'],[['固定均值','0%','0.000523','0.1465'],['固定示例','0%','0.000608','0.1649'],['跨图状态','0%','0.000726','0.1954'],['位置乱序','1.25%','0.841362','140.9208'],['零 hidden / 零投影','100%','0.891649','247.3345']],[1.5,.8,1,1.7])
para('例子与分析','固定均值条件在每张测试图使用完全相同的 H，但保留各自 F。乱序把第 i 个向量接到别的专家层，改变了位置对应关系。FP32 解码器复核后各条件正确率和弃权率相同。','位置对应关系有作用；当前图像特异 H 的必要性尚未建立。')
conclude('零 hidden 仍经过 MLP 偏置；零投影直接清空点积向量。两者在本批均弃权，不能称为方向选错。')

sec('答案诊断：八种任务与基座模型','R04；16 图，8 个族；CoVT 与 Qwen 各 128 次生成')
para('目的与设定','复用 FP32 pilot 的 base 与 label_swap，共 16 图；固定各自 processor 和原生模板，FP32、eager、greedy、最多 192 token。比较技术问法、直白近/远、标签识别、面积、左右答案及 24 px 字母。','相同输入上的任务比较可定位行为现象；基座比较不能识别某项训练成分的因果作用。')
figure(answerfig,'图 7｜横轴为正确图像数，所有条形分母均为 16；纵轴为八种任务，蓝色 CoVT、橙色 Qwen 基座。条末为正确数；无误差条。八任务复用相同图像，不能把条形当作互相独立的数据集。',275)
para('真实例子','scene_00002_base 的真值为 B。CoVT 对技术问法回答 A，对 closer to the camera 回答 B，对 farther 回答 A；同图的标签识别和面积比较也正确。','问法会改变该案例答案，但一个改善案例不能解释全部错误。')
para('结果与分析','两模型标签和面积任务均 16/16；直白近距离均 11/16。CoVT 近距离族区间为 56.25%-87.5%，基座为 50%-87.5%，区间重叠。','本批未显示 CoVT 在直白近距离任务上优于基座。')
conclude('低答案正确率不能单独归因于字母看不清；深度问题的表述与关系判断仍需单独检查。')

sec('答案诊断：配对一致性与实际标签交换','R04；paired_summary.json；cases 图像')
figure(answercasefig,'图 8｜真实 scene_00002 标签交换对。两图物体和几何保持不变，只互换 A/B 标记；不显示坐标轴，因为本图用于展示输入和标注变化，不编码新的数值指标。此族直白近距离答案随标记从 B 变 A，属于一致案例；全体结果见下表。',220)
table(['配对指标与计算规则','CoVT','Qwen'],[['近/远互补：同图两个答案互为 A/B','9/16','10/16'],['标签交换一致：交换标记后近答案翻转','3/8 族','3/8 族'],['左右回答一致：换标签后 left/right 不变','7/8 族','5/8 族']],[3,1,1])
para('简单例子（说明性）','原图 B 在左且较近；交换标记后 A 在左且较近。近距离的 A/B 答案应 B→A，而 left/right 应保持 left。问近与问远的互补只检查两答案是否相反。','一致性与正确性不同：两道互补题可能同时选错。')
para('结果与分析','CoVT 原问法到直白近距离有 6 张错→对、1 张对→错；基座为 4 和 1。原问法 6/16 不能与四种变体全集的 8/32 当作相同评估集。','提示改善是配对的探索性观察；整体标签交换一致性仍不足。')
conclude('简单识别任务通过并未保证可靠的关系回答。后续冻结使用直白问法和较大字母，但不把跨批改善归因于单一改动。')

sec('冻结状态扩展：五类新几何场景','R05；seed=1901，50 个族，200 图；全部用于评估')
para('目的与设定','保持上一轮校准方向、尺度、阈值及校准均值/示例 H 不变，只为每张新图提取当前 F。五类几何各 10 个族、40 图，专家和解码器 FP32。无需为全部 200 图生成各自 CoVT 状态。','这一设计直接检验旧固定状态在新几何上的读出表现。')
figure(shiftfig,'图 9｜五个真实 base 场景例图，标题给出类别与样本 ID。图像无坐标轴；仅展示输入分布，不用图像尺寸编码正确率。所有输入为 336×336，24 px 字母；几何真值来自归档 manifest。',160)
table(['范围','专家正确','固定均值正确','固定示例正确'],[['尺寸变化（40）','39','40','40'],['椭球（40）','39','39','39'],['盒体（40）','40','40','40'],['部分遮挡（40）','40','40','40'],['相机变化（40）','40','40','40'],['全部（200）','198','199','199']],[2,1,1,1])
para('真实失败例子','shift_00001_depth_swap 的真值 A，固定均值 r=0.03239，落在 |r|≤0.05 区间，因此弃权。固定状态的 1 个未正确样本及专家的 2 个未正确样本均为弃权。','应报告 199 正确 + 1 弃权，不能写成 1 次方向错误。')
conclude('固定状态在该新几何集合上仍有效；固定状态 99.5% 的族区间 [98.5%,100%] 与专家区间重叠，不据此主张优于专家。')

sec('扩展场景原生对照与大小线索检查','R05；预选 40 图原生子集；39 图状态合格')
para('目的与设置','在预先选定的 40 图上比较原生、固定均值、固定示例、跨图状态及 token 打乱。保留不生成四个 pad 的样本为失败。另用可见物体面积定义“面积更大者更近”的规则，不使用模型答案。','原生配对比较只覆盖子集，不能写成全部 200 图均有原生状态对照。')
table(['指标','结果','统计范围'],[['自然状态合格','39/40','1 张不生成 pad'],['语言答案正确','22/40；合格中 22/39','全部尝试与合格样本'],['原生 / 固定 / 跨图深度正确','族平均均 96.67%','39 图，10 个族；区间 90%-100%'],['固定均值与原生预测一致','逐样本一致','平均 |Δr|=0.00459'],['token 打乱','族平均 56.5%','95% 区间 48%-64.5%'],['面积规则正确率','84%；92.5%','200 图；预选 40 图']],[2,1.6,1.6])
para('真实失败例子','shift_00001_label_swap 直接输出 B，没有四个深度 pad，缓存模式 token 一致。该失败属于状态不可用，不属于之前的缓存数值不稳定。','两种失败机制需要分别计数。')
para('大小规则与分析','面积规则按图像中可见实例像素数选择 A 或 B。若一个球更大且确实更近，该图无需可靠推断三维深度也能被规则答对。原生子集的规则正确率达到 92.5%。','较高深度正确率尚未排除面积线索，需要预先平衡一致和冲突条件。')
conclude('固定状态现象在原生子集延续；下一轮必须控制明显几何线索，单纯增加同类图像不足以消除混杂。')

sec('大小平衡：远近顺序 × 表观大小顺序','R06；seed=1902，8 个族，32 图')
para('目的与设定','每族交叉深度顺序 d∈{0,1} 与大小顺序 s∈{0,1}，A/B 真值各 16，一致/冲突各 16。用半径=相机 z×角半径控制投影大小；交换深度时保持角半径和 x/z。世界半径、地面接触位置也会变化。','这是有已知标签的场景干预，未声称只改变单一物理因素。')
figure(balancedfig,'图 10｜真实 balanced_000 的四个变体，标题列出 d/s、真值、语言答案与大小线索类别。无坐标轴；四图同为 336×336。d0_s1 中较远的 B 占面积 3658 像素，较近 A 仅 1717 像素，是大小冲突例。',260)
table(['范围','语言正确','固定状态深度正确','专家正确'],[['一致（16）','12','16','16'],['冲突（16）','13','14','10'],['全部（32）','25','30','26']],[1.4,1,1.6,1])
para('案例与分析','d0_s1 的真值 A，语言答 A，固定状态 r=0.07139 判 A，专家 r=−0.03600 弃权。面积规则在冲突组正确率 0%，但上述分支仍有较高正确数。','本批判断不完全服从“更大即更近”，其他透视与接触线索仍未排除。')
conclude('大小冲突下固定状态读出仍有效；固定状态 2 个未正确含 1 弃权、1 错向，专家 6 个未正确含 4 弃权、2 错向。')

sec('中间层干预：替换后重新生成答案','R06；model.layers.14 输入，28 层编号从 0 开始')
para('目的与设置','检验四个深度位置的中间状态能否影响后续语言回答。32 张中 26 张自然生成四个 pad；6 张无 pad，均为缓存 token 一致。截断到最后一个自然 pad 后，逐步无缓存重算前缀，持续写入供体状态，不保留旧答案。','这一步测量后续行为，比末端解码替换更接近答案因果问题。')
formula('已生成前缀 … [depth pad × 4] │ 后续转接文本 │ &lt;answer&gt; A/B<br/>在第 14 层四个 pad 位置替换输入残差 → 从分隔处重新生成全部后续 token<br/>原样写回 = 把同一样本的状态放回相同位置，用作实现一致性检查。')
table(['条件','目标数','答案改变','说明'],[['原样写回','26','0','26/26 通过'],['同族相反深度','20','0','供体也需状态合格'],['同标签、同大小条件跨族','26','0','自然状态对照'],['四位置循环排列','26','0','干预位置对应关系']],[2,1,.8,1.7])
para('真实案例','balanced_000_d0_s0 真值 A、答 A；配对 d1_s0 真值 B、答 B。把后者第 14 层的四个状态放到前者后，重新生成仍为 A。','至少这个供体本来回答不同的配对未实现答案转移。')
para('分析与筛选','124 次含基线的生成均有效完成。相反深度替换后的下游 pad 状态确有变化，逐样本最大绝对差为 0.0472-0.1110；无答案变化不等于 hook 未执行。相反与同深度公平比较只取共同 20 图、6 个族。','20 图与全体 26 图分母不同，不可直接比较准确率改善。')
conclude('第 14 层自然状态替换不足以改变当前答案；单层结果尚不能排除层或位置覆盖不足。')

sec('七层扫描：同时测答案与概率','R07；同一 20 图、6 个族，4 个深度位置')
para('目的与设置','扫描层 0、4、8、14、20、24、27，逐层比较自然相反深度和置零。每次均替换四个位置，完整重新生成；另在原答案前固定上下文测 q。140 次原样写回通过，20 次最终答案读出正控制均能转为供体答案。','正控制证明读出测量可检测答案改变，不能证明深度路径具有语义作用。')
figure(layerfig,'图 11｜横轴为从零开始的层号；纵轴为 20 图中最大绝对条件概率变化，单位百分点。左为自然供体，右为置零；两面板纵轴范围不同，不能用线条高度直接比较。无误差条，最大值为描述性汇总。概率使用固定后续上下文，右图不能代表自由生成的答案质量。',235)
table(['层','相反深度答案翻转','最大自然供体变化（百分点）','置零原解析器无效数'],[[k,'0/20',f'{next(r for r in ls["details"] if r["layer"]==k and r["condition"]=="opposite_depth")["max_delta_p_A"]*100:.5f}',next(r for r in ls['details'] if r['layer']==k and r['condition']=='zero')['invalid']] for k in layers],[.6,1.4,1.9,1.6])
conclude('自然供体七层均无翻转，最大变化仅 0.00710 个百分点；置零行为必须进一步检查文本和格式。')

sec('置零结果复核：格式、字母与语义分开','R07；results.jsonl 与 format_layer_*.json；审计为事后补充')
para('目的与规则','原解析器可能把异常文本最后一个 A/B 行当成答案。逐条保留原文，检查完整答案标签、是否同时含 A/B、是否完全没有字母，并保留仅有一种字母的诊断。','事后格式审计用于限制解释，不覆盖原始结果。')
table(['层','完整无歧义标签','同时含 A/B','没有 A/B','典型真实输出'],[['0','20/20','0','0','完整标签，答案不变'],['4','0/20','12','0','保守；A；B（分行）'],['8','0/20','0','20','换行后结束'],['14','0/20','7','0',' A</answer> 或 A or B.</answer>'],['20','3/20','0','17','downloadable；重启思考段'],['24','0/20','0','20','立即结束'],['27','20/20','0','0','完整标签，答案不变']],[.5,1.2,.9,.8,2.4])
para('真实案例','balanced_001_d1_s0 第 4 层置零后输出“保守；B&lt;/answer&gt;&lt;/answer&gt; B；B”（分号表示换行）。该样本原答案 A，字母确有变化，但包含重复与损坏标签。','不能抹去内容变化，也不能将其解释为干净的深度语义转移。')
para('结果与分析','第 4 层原解析器报告 7/20 翻转；12 条同时含 A/B，剩余 8 条只含一种字母，其中 3 条改变了字母。第 14 层 13 条只含一种字母且与基线一致，但均无完整标签。','“答案改变”“格式无效”和“唯一字母改变”是不同统计。')
para('固定上下文与自由生成','置零时固定答案上下文内 A/B 总概率仍超过 99.5%，但自由生成可能提前结束。这是因为固定上下文保留了原有过渡 token。','高固定上下文 A/B 概率不保证干预后会自然走到该答案位置。')
conclude('置零确认这些位置可以影响生成过程；具体作用是否属于深度语义，原因未查明。')

sec('连续层、全层与思考前缀扩展','R08；同一 20 图、6 个族；200 次实际干预')
para('目的与设置','依次联合替换早层 0-8、中层 9-17、晚层 18-27、全部 28 层；前四阶段均为四个 pad。全层无有效翻转后，按预定规则扩大为提示结束至最后 pad 的 15 个位置。加入跨族同深度供体，匹配真值、深度/大小顺序与线索类别。','该扩展直接检验层数和思考前缀位置覆盖；供体答案 token 始终排除。')
figure(blockfig,'图 12｜横轴为阶段及替换位置数；纵轴为 20 图中最大绝对 A/B 条件概率变化，单位百分点。蓝色相反深度、橙色同深度供体；数字为描述性最大值，无误差条。所有阶段的两种供体均为 0/20 答案翻转，正确数均为 13/20。',245)
para('检查与结果','100/100 原样写回通过。200 次干预全部正常结束、格式有效、无 A/B 歧义，完整后续 token 序列均与基线相同。最宽范围最大变化为 0.0472 个百分点，同深度为 0.0422。','覆盖扩大后数值影响略增，尚未形成实际答案控制。')
para('统计比较','最宽范围的族平均绝对概率变化：相反深度 0.0126 个百分点，同深度 0.0133；相反减同深度的定向差为 0.00377，族区间 [−0.00573,0.0136] 个百分点。','本批未得到稳健的深度方向控制证据；零翻转不构成总体零效应证明。')
conclude('“只换一层太少”已不足以解释本批阴性结果；扩大到 15 个位置也没有改变语言输出。')

sec('多层具体案例与未覆盖的位置','R08；最大概率变化案例 balanced_003_d1_s0')
figure(blockcasefig,'图 13｜真实目标与供体，几何真值和模型原答案均相反；两图不显示数值轴，标题提供可追溯样本 ID。全层、15 位置替换保持目标图像，只移植供体对应层的思考前缀状态，不移植供体答案。',220)
table(['量','干预前','全层 + 15 位置替换后'],[['目标答案','A','A'],['P(A | A 或 B)',f'{best["conditional_p_A"]-best["delta_p_A"]:.8f}',f'{best["conditional_p_A"]:.8f}'],['概率变化','基线','−0.047183 个百分点'],['全词表 A−B logit 差变化','基线',f'{best["delta_logit_A_minus_B"]:.8f}']],[2,1.5,2])
para('供体答案的区分','20 个相反深度配对中只有 6 个目标的供体原本回答不同；这 6 个目标在五阶段也均无翻转。其余 14 个是几何真值相反，但模型本来回答相同。','不能只凭真值相反就认定所有配对都具备相反的行为输出。')
para('覆盖边界','15 个位置包括 11 个前置生成位置和 4 个 pad，不含图像 token、问题文本、最后 pad 之后的过渡位置及答案。最大逐层相对状态 L2 距离为 2.14%，说明替换值确有变化。','本轮不代表完整推理序列的全位置干预；距离也不能直接解释为语义强度。')
conclude('该真实案例说明即使供体本来回答相反，所测状态替换也只造成微小概率移动；下一步应定位未覆盖的信息路径。')

sec('综合结论、限制与下一步')
table(['当前能得到的结论','直接证据','尚不能得到的结论'],[['原生分支可恢复并运行','参数、算子、E0 与专家检查','任意输入均稳定、米制深度准确'],['固定旧状态可支持当前深度关系读出','32 图状态对照；200 图扩展；冲突 14/16','状态普遍无信息或可正式等价删除'],['位置对应结构影响深度读出','乱序降至 58.125%；零条件弃权','位置携带的就是任务所需深度语义'],['自然供体替换对当前答案影响小','单层、连续层、28 层与 15 位置结果','模型全部信息路径都不依赖思考状态'],['强干预可破坏生成结构','置零异常文本审计','异常输出即为可控的语义翻转']],[1.8,1.6,1.9])
para('限制汇总','主要是简单几何、单个新 seed、小量独立族、自然 pad 可用性筛选、供体复用和稀疏路径覆盖。多数极高正确率区间退化；多重探索比较没有用于确认性推断。','应把结果定位为 pilot 机制证据，不能推广为普遍模型能力或机制定理。')
para('下一步建议（未执行）','依次检查 pad 后、答案前的过渡位置，问题文本位置和视觉 token。优先保留供体实际答案不同的配对，并加入自然换图或视觉嵌入交换对照；避免直接移植答案。','下一轮应验证答案信息经过哪些路径，继续扩大置零范围的解释收益有限。')
para('研究方向','当前证据更适合支持“原生可视化读出与答案行为的关系需要独立验证”这一问题。是否开展训练修复，需先找到能稳定改变语义且不破坏格式的干预。','目前没有经过验证的修复方向，不把尚未执行的训练收益写入结论。')

sec('复现配置与计算细节')
table(['项目','固定配置'],[['模型','Wakals/CoVT-7B-depth；7B；28 层'],['CoVT 权重 revision','48f5a921c01db42b33ca0949b303b17a3e085bf4'],['CoVT 源码 commit','c3497a325b0fdda4d91ade456a01b0e325ee3150'],['深度专家 revision','cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9'],['Qwen 基座 revision','cc594898137f460bfe9f0759e9844b3ce807cfb5'],['环境','Colab A100-SXM4-40GB；Python 3.11.16；torch 2.5.1；transformers 4.50.1；torchvision 0.20.1'],['生成','greedy；E0/答案诊断最多 192 新 token；层扫描/多层重新生成最多 128'],['深度 token','token ID 151669；恰好四个自然 pad；不插入或补造'],['校准','首轮：方向 −1，尺度 137.54537964，3 图；FP32：方向 −1，尺度 151.85319519，8 图；阈值 0.05']],[1.2,4])
para('插值与边界','原生深度图输出 256×256；评估时双线性插值到输入 336×336，align_corners=False。环带 5-9 像素；取中位数后做 B−A 差。非有限值、空环带、网格不兼容及状态数量错误均不静默修正。','评估定义固定后再比较干预，图像展示的色彩归一化不参与关系评分。')
para('工程诊断修订','七层扫描最初对跨前缀状态要求逐位相同而停止，随后在干预前记录误差并固定容差。概率原样写回仍要求 logits 逐位一致；多层沿用该方案。','数值检查修订有记录，不能将工程容差当成科学等价界限。')
conclude('复现应使用归档源码和固定版本；报告绘图只读取已存结果，没有重新训练、推理或重新校准。')

sec('证据索引、归档状态与引用')
table(['编号','本地证据目录（相对 covt-pilot/runs）','主要文件'],[
 ['R01','docs/reference_parity.json；docs/resource_snapshot.json（相对 repo）','算子对照、固定资源元数据'],
 ['R02','colab-20260918-backup/runs/colab-e0-20260918-032255','e0/scored.jsonl；e1/results.jsonl；地图与特征'],
 ['R03','colab-20260918-followup/runs/','cache-diagnostic-*；followup-fp32-*/e0,e1,controls'],
 ['R04','colab-20260918-answer-shift/extracted/runs/answer-diagnostic-20260918-050133','results.jsonl；paired_summary.json；cases/'],
 ['R05','colab-20260918-answer-shift/extracted/runs/shift-validation-20260918-050713','frozen_results.jsonl；native/；geometry_baseline.json'],
 ['R06','colab-20260918-balanced-intervention/extracted/runs/balanced-intervention-20260918-090000','native.jsonl；frozen.jsonl；interventions.jsonl'],
 ['R07','colab-20260918-layer-scan/extracted/runs/layer-scan-20260918-093053','results.jsonl；final_audit.json；format_layer_*.json'],
 ['R08','colab-20260918-block-intervention/extracted/runs/block-intervention-20260918-100708','results.jsonl；summary.json；*_paired.json；final_audit.json']],[.5,2.8,1.9])
para('归档完整性','首轮完整归档已校验。FP32 后续本地只有精简包，约 1 GB 完整缓存包尚未核验；后续扩展包也不包含全部专家特征/几何缓存。每个阶段 INDEX.json 记录真实下载状态及 SHA-256。','有报告和源码不等于已有全部缓存；报告未使用未落盘的地图补造图像。')
para('官方来源','固定源码：github.com/Wakals/CoVT（commit 见第 23 页）；模型：huggingface.co/Wakals/CoVT-7B-depth；基座：huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct。模型事实以本项目归档源码和 checkpoint 审计为准。','本报告为自有实验整理，未新增外部论文结论。')
p('最新多层归档 SHA-256：<br/>445994011a7f3fe650a920af85e1e859b1f9ca9aef7d65e8ad4a1f3167ad4409','small')
conclude('真实案例、数据表、指标定义及归档索引共同构成当前结论的追溯依据。')

def footer(canvas,doc):
    canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#D7E1E8')); canvas.line(M,38,W-M,38)
    canvas.setFont('CN',8); canvas.setFillColor(colors.HexColor(GRAY)); canvas.drawString(M,25,'CoVT 原生深度与答案干预 · 阶段性报告 · 2026-09-18')
    canvas.drawRightString(W-M,25,f'{doc.page} / {len(sections)}'); canvas.restoreState()
target=OUT/'CoVT_实验综合报告_20260918.pdf'
doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=M,leftMargin=M,topMargin=40,bottomMargin=52,
                      title='CoVT 原生深度读出与答案干预实验报告',author='CoVT Research',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'report_text_source.txt').write_text('\n\n'.join(textual),encoding='utf-8')
sources=[BF/'e0/scored.jsonl',FP/'e1/results.jsonl',FP/'controls/results.jsonl',ANS/'results.jsonl',SH/'frozen_results.jsonl',BAL/'native.jsonl',LS/'results.jsonl',BL/'results.jsonl']
(OUT/'report_sources.json').write_text(json.dumps({'report':target.name,'expected_pages':len(sections),'sections':sections,'data_sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(target),'expected_pages':len(sections),'figures':len(list(FIG.glob('*.png')))},ensure_ascii=False))
