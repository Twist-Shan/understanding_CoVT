"""Add an archived-data gallery and ground-truth explanation; no experiments."""
from pathlib import Path
import runpy,json,base64,html,re
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
b=runpy.run_path(str(ROOT/'scripts/build_explained_report.py'))
OUT=ROOT/'output/html'
specs=[
('first','首轮基础球体',12,ROOT/'runs/colab-20260918-backup/runs/colab-e0-20260918-032255/scenes'),
('fp32','FP32 基础球体',40,ROOT/'runs/colab-20260918-followup/runs/followup-fp32-20260918-043002/scenes'),
('shift','五类扩展场景',200,ROOT/'runs/colab-20260918-answer-shift/extracted/runs/shift-validation-20260918-050713/scenes'),
('balanced','大小与远近平衡',32,ROOT/'runs/colab-20260918-balanced-intervention/extracted/runs/balanced-intervention-20260918-090000/scenes')]
groups={}
for key,title,count,path in specs:
    rows=[json.loads(s) for s in (path/'manifest.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    assert len(rows)==count
    for r in rows:
        assert (path/r['image']).is_file(),path/r['image']
        assert r['label']==('A' if r['point_depths'][0]<r['point_depths'][1] else 'B')
    groups[key]=rows
variants={'base':'原图','depth_swap':'交换远近','nuisance':'只换颜色','label_swap':'交换 A/B 字母'}
strata={'size_variation':'物体尺寸变化','ellipsoids':'椭球','boxes':'盒体','partial_occlusion':'部分遮挡','camera_variation':'相机变化'}
def esc(x):return html.escape(str(x))
def card(key,row):
    path=next(s[3] for s in specs if s[0]==key)
    title=next(s[1] for s in specs if s[0]==key)
    label=variants.get(row.get('variant'),'')
    if key=='balanced':label=('大小与远近一致' if row['cue']=='aligned' else '大小与远近冲突')+f' · 远近组合 {row["depth_order"]} / 大小组合 {row["size_order"]}'
    category=strata.get(row.get('stratum'),'')
    search_terms=' '.join([row['id'],title,label,category,row.get('stratum',''),row.get('variant','')])
    data=base64.b64encode((path/row['image']).read_bytes()).decode()
    za,zb=row['point_depths']
    return f'''<article class="sample" data-batch="{key}" data-label="{row['label']}" data-search="{esc(search_terms)}"><img src="data:image/png;base64,{data}" loading="lazy" alt="{esc(row['id'])}，{label}，{row['label']} 更近"><div class="sample-text"><b>{esc(label)}</b><p>{esc(category)}<br><strong>正确答案：{row['label']} 更近</strong><br>A 深度 {za:.3f} · B 深度 {zb:.3f}</p><small>{title} / {esc(row['id'])}<br>A 像素 {row['points'][0]}；B 像素 {row['points'][1]}</small></div></article>'''
case=next(r for r in groups['first'] if r['id']=='scene_00002_base')
za,zb=case['point_depths']
intro=f'''<h2>为什么截图里是 B 更近？</h2><p>我们比较的是<strong>字母旁白色小圆点所标记的物体表面</strong>。A 在右边的粉色球上，B 在左边的紫色球上。判断依据来自合成场景中的几何位置：相机沿着前方看去，表面点的轴向深度 z 越小，表示它沿相机前方方向越近。</p>
{b['table'](['标记','图中位置（像素）','表面点深度 z','判断'],[['A','(222,191)，右侧粉色球',f'{za:.6f}','更远'],['B','(113,201)，左侧紫色球',f'{zb:.6f}','更近']])}
<div class="walk"><b>相机轴向位置示意：只是数值顺序，未按比例绘制</b><p>相机 z=0 → B 表面 z≈3.408 → A 表面 z≈5.380</p></div>
<p>这些数值是<strong>场景坐标单位</strong>，没有标定成真实世界的米；也不同于深度模型输出的颜色条分数。它们来自渲染时的射线与物体表面交点，不是球心深度。严格说，本数据比较相机轴向 z，而非到相机中心的欧氏距离。</p>
<p>这张图里两个球的实际半径很接近：紫球约 0.828，粉球约 0.822；更近的紫球投影较大，所以视觉上也容易看出 B 更近。但“看起来更大”不能作为通用标签规则，后面的大小冲突图专门检验这个问题。字母 A/B 只负责标记位置，A 并不默认代表近。</p>
<div class="sample-grid single">{card('first',case)}</div><p class="conclusion"><strong>结论：</strong>B 更近是场景几何给出的标签；专家图和 CoVT 图是否预测正确，都要与这个标签比较。</p>'''
scope='''<h2>实验只能做深度吗？</h2><p><strong>不必。</strong>“恢复视觉输出 → 改中间状态 → 检查输出和文字答案是否改变”的方法可以用于其他视觉任务。当前我们只恢复和测试了公开的 CoVT 深度版本，不能把这些结果直接推广到其他分支。</p>'''+b['table'](['可扩展任务','一个容易理解的问题','需要准备与比较什么'],[
['分割、位置与计数','图里有几个球？左边物体是哪一个？','使用包含分割能力的 CoVT 权重；比较区域掩码、数目/位置答案，再做相应状态干预。重建分支若使用图像特征，也必须单独控制。'],
['边缘与形状','图里有几条边界线？物体轮廓是否闭合？','使用边缘分支权重和解码器，准备明确的线条或轮廓标签；同时检查边缘图与文字答案。'],
['视觉特征与对象对应','两幅图中哪个物体相对应？','使用相应视觉特征分支，先规定特征匹配指标和文字任务；特征本身不是直接可读的图像标签。']])+'''<p>CoVT 官方框架包含分割、深度、边缘和 DINO 特征等表示；不同能力需要对应的权重与分支。当前四个深度状态不能直接改名当成四个分割状态。<a href="https://github.com/Wakals/CoVT">官方仓库</a>。</p><p>先选深度，是因为我们能在合成场景中精确知道 A、B 的远近，问题可以简化成二选一，并且已经找到了可恢复的公开深度权重。分割也适合后续扩展：当前场景生成过程有物体身份信息，已归档的部分几何文件包含逐像素物体编号，但新任务仍要重新定义标签、核验归档、准备对应模型并实际运行。</p><p class="conclusion"><strong>结论：</strong>方法不限于深度，当前证据只覆盖深度；其他任务尚未实验。</p>'''
overview='''<h2>我们的数据集里有什么？</h2><p>当前基础场景都是程序生成的简单几何图片，尺寸为 <strong>336×336</strong>。它们不是自然照片，也不是原论文完整评测集。下面四批共 <strong>284 条基础场景记录</strong>；同一场景的变体有关联，所以不能当作 284 个独立场景。</p>'''+b['table'](['批次','基础图片 / 场景族','内容','用在哪里'],[
['首轮','12 / 3','球体：原图、交换远近、换颜色、交换字母','恢复和首轮一致性检查；4 张校准、8 张评估尝试'],
['FP32','40 / 10','另一个种子生成的球体，同样四种变体','8 张校准、32 张评估；其中 16 张复用于问法诊断'],
['几何扩展','200 / 50','尺寸、椭球、盒体、部分遮挡、相机变化；每类 40 张','固定状态测试；其中 40 张做自然状态对照'],
['大小平衡','32 / 8','远近顺序 × 表观大小顺序，共四种组合','检验大小线索；其中 20 张进入后续匹配干预']])+'''<p>“256 次答案诊断”“200 次多层干预”表示运行次数，不能在 284 张上继续相加。它们复用了已有场景。放大字母的图片属于已有场景的派生版本，本展示不把它们另算为独立基础场景。不同批次可能复用相同编号，所以这里始终同时标明批次。</p><p>每条记录保存图片路径、问题、正确标签、A/B 像素位置、两个表面点的深度、所属场景族，以及用于生成的物体和相机参数。不是所有批次都下载了完整逐像素几何与模型特征缓存，完整性限制见报告附录。</p>'''
firstfour=[r for r in groups['first'] if r['family']=='scene_00002']
shiftfive=[next(r for r in groups['shift'] if r['stratum']==s and r['variant']=='base') for s in strata]
balancedfour=[r for r in groups['balanced'] if r['family']=='balanced_000']
curated='''<h3>展示一：同一场景的四种变体</h3><p>下面四张取自截图所属场景族。交换远近会重新渲染，物体投影大小和接地位置也可能变化；换颜色保持几何不变；交换字母只改变 A/B 对应关系，正确字母随之变化。每张都列出真实深度，直接比较数值即可核对标签。</p><div class="sample-grid">'''+''.join(card('first',r) for r in firstfour)+'''</div><h3>展示二：扩展数据的五种变化</h3><p>每类各取一张原图，只用于说明有什么内容，不是按模型成功或失败挑选。</p><div class="sample-grid">'''+''.join(card('shift',r) for r in shiftfive)+'''</div><h3>展示三：近的物体也可以看起来更小</h3><p>下面四张来自同一大小平衡场景。标为“冲突”的图中，看起来较大的物体反而更远。请比较图像外观、A/B 深度和正确答案。</p><div class="sample-grid">'''+''.join(card('balanced',r) for r in balancedfour)+'''</div>'''
gallery='''<h3>全部 284 张基础图片</h3><p>可按批次或正确答案筛选，也可搜索编号、类别或“交换字母”“冲突”等关键词。图片保持归档原样。每张图下面的数值来自场景记录，未运行新模型。</p><div class="gallery-controls"><label>批次 <select id="batch-filter"><option value="all">全部批次</option>'''+''.join(f'<option value="{k}">{t}（{n} 张）</option>' for k,t,n,_ in specs)+'''</select></label><label>正确答案 <select id="answer-filter"><option value="all">全部</option><option>A</option><option>B</option></select></label><label>搜索 <input id="sample-search" type="search" placeholder="例如 boxes 或 balanced_000"></label><p id="gallery-count" role="status">显示 284 / 284 张</p></div><div class="sample-grid" id="all-samples">'''+''.join(card(k,r) for k,_,_,_ in specs for r in groups[k])+'''</div>'''
css='''.sample-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px;margin:24px 0}.sample{border:1px solid #dce4e9;background:#fff;min-width:0;border-radius:6px;overflow:hidden}.sample img{display:block;width:100%;height:auto;image-rendering:auto}.sample-text{padding:15px;font-size:14px;line-height:1.7}.sample-text small{color:#617080;overflow-wrap:anywhere}.single{max-width:430px;grid-template-columns:1fr}.gallery-controls{padding:18px;background:#eef5f7;display:flex;flex-wrap:wrap;gap:16px}.gallery-controls label{display:flex;flex-direction:column;font-size:14px}.gallery-controls select,.gallery-controls input{font:inherit;padding:8px;border:1px solid #9ab0bc;max-width:100%}#gallery-count{width:100%;margin:0}[hidden]{display:none!important}@media(max-width:600px){.sample-grid{grid-template-columns:1fr}}@media print{#all-samples{display:none}.gallery-controls{display:none}.sample{break-inside:avoid}}'''
js='''
const sampleCards=[...document.querySelectorAll('#all-samples .sample')];
function filterSamples(){const batch=document.getElementById('batch-filter').value, answer=document.getElementById('answer-filter').value, term=document.getElementById('sample-search').value.toLowerCase().trim();let count=0;for(const card of sampleCards){const ok=(batch==='all'||card.dataset.batch===batch)&&(answer==='all'||card.dataset.label===answer)&&card.dataset.search.toLowerCase().includes(term);card.hidden=!ok;if(ok)count++}document.getElementById('gallery-count').textContent=`显示 ${count} / ${sampleCards.length} 张`;}
for(const id of ['batch-filter','answer-filter','sample-search'])document.getElementById(id).addEventListener('input',filterSamples);
'''
fragment=f'<section id="dataset-scope">{scope}</section><section id="dataset-case">{intro}</section><section id="dataset-show">{overview}{curated}<details><summary>展开全部 284 张基础图片与筛选工具</summary>{gallery}</details></section>'
document=b['document'].replace('</style>',css+'</style>',1)
document=document.replace('</header>','</header>'+fragment,1)
document=document.replace('<h2>阅读顺序</h2>','<h2>阅读顺序</h2><a href="#dataset-scope">补充：不限于深度</a><a href="#dataset-case">截图为什么 B 更近？</a><a href="#dataset-show">数据集实际图片展示</a>',1)
document=document.replace('</script>',js+'</script>',1)
b['target'].write_text(document,encoding='utf8')
standalone='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CoVT 数据集：284 张基础场景</title><style>'+b['base']['css']+css+'main{max-width:1100px;margin:30px auto;padding:15px}</style></head><body><main><header><h1>我们的数据集有哪些图片？</h1><p>归档基础场景展示 · 284 张 · 不包含新增实验</p></header><section>'+intro+'</section><section>'+overview+curated+'</section><section>'+gallery+'</section></main><script>'+js+'</script></body></html>'
(OUT/'CoVT_数据集展示.html').write_text(standalone,encoding='utf8')
(OUT/'dataset_manifest.json').write_text(json.dumps({'base_scene_records':284,'families':71,'gallery_is_not_new_experiment':True,'batches':[{'key':k,'title':t,'images':n,'families':len(set(r['family'] for r in groups[k])),'labels':dict(Counter(r['label'] for r in groups[k])),'manifest':str((p/'manifest.jsonl').relative_to(ROOT))} for k,t,n,p in specs]},ensure_ascii=False,indent=2),encoding='utf8')
print('Saved dataset gallery and updated full report; all 284 image paths and geometric labels verified.')
