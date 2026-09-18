"""English figures from saved experiment artifacts. Does not run a model."""
from pathlib import Path
import sys, shutil
sys.path.append('C:/anaconda3/Lib/site-packages')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/research/notes_figures'
OUT.mkdir(parents=True,exist_ok=True)
# Reuse only the archived-data loading preamble, before any rendering or PDF work.
source=(ROOT/'scripts/build_research_report.py').read_text(encoding='utf8')
scope={'__file__':str(ROOT/'scripts/build_research_report.py')}
exec(source.split("FONT='C:/Windows/Fonts/msyh.ttc'")[0],scope)
BF,FP,ANS,SH,BAL=[scope[k] for k in ['BF','FP','ANS','SH','BAL']]
JL=scope['JL']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
def save(name,fig):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)
def scene(ax,path,title):
    ax.imshow(Image.open(path));ax.set_title(title,fontsize=9);ax.axis('off')
for i in (1,2,3):
    original=ROOT/f'output/html/paper_figures/covt_figure_{i}.png'
    if original.exists():shutil.copy2(original,OUT/f'paper_{i}.png')
    elif not (OUT/f'paper_{i}.png').exists():raise FileNotFoundError(f'Missing attributed paper figure {i}')
case=next(r for r in scope['bf'] if r['id']=='scene_00002_base')
cache=np.load(BF/'e0'/case['artifact'])
fig,axs=plt.subplots(1,3,figsize=(9,2.8),layout='constrained')
axs[0].imshow(Image.open(BF/'scenes'/case['image']));axs[0].set_title('Input: truth B; text answer A')
for ax,key,title in zip(axs[1:],['decoded_map','expert_map'],['CoVT conditional readout: B','Depth expert: B']):
    im=ax.imshow(cache[key],cmap='viridis');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.72)
for ax in axs:ax.set_xlabel('x (pixels)');ax.set_ylabel('y (pixels)')
save('native_case',fig)
fig,axs=plt.subplots(1,5,figsize=(9.5,2.1),layout='constrained')
rows=JL(SH/'scenes/manifest.jsonl')
for ax,s,t in zip(axs,['size_variation','ellipsoids','boxes','partial_occlusion','camera_variation'],['Size','Ellipsoids','Boxes','Occlusion','Camera']):
    r=next(r for r in rows if r['stratum']==s and r['variant']=='base')
    scene(ax,SH/'scenes'/r['image'],t)
save('dataset',fig)
fig,axs=plt.subplots(1,2,figsize=(8.7,2.7),layout='constrained')
for ax,change,title in zip(axs,['depth_swap','nuisance'],['Depth-order change','Color change']):
    rr=[r for r in scope['e1'] if r['change']==change]
    xx=np.arange(len(rr))
    ax.plot(xx,[abs(r['state_effect_Fa']) for r in rr],'o-',label='Change H only')
    ax.plot(xx,[abs(r['expert_effect_Ha']) for r in rr],'s-',label='Change F only')
    ax.set_yscale('log');ax.set_xlabel('Scene family (index)');ax.set_ylabel('Absolute change in r (log scale)');ax.set_title(title);ax.legend(fontsize=8)
save('factorial',fig)
fig,ax=plt.subplots(figsize=(8.4,2.7),layout='constrained')
tasks=['original_depth','plain_near','plain_far','label_left','apparent_size','side_near','large_plain_near','large_label_left']
names=['Original','Near','Far','Left label','Area','Left/right','Large: near','Large: label']
ans=scope['ans'];base=next(m for m in {r['model'] for r in ans} if m!='covt')
for j,(m,l) in enumerate([('covt','CoVT'),(base,'Qwen base')]):
    ax.bar(np.arange(8)+(j-.5)*.36,[sum(r['correct'] for r in ans if r['model']==m and r['task']==t) for t in tasks],.36,label=l)
ax.set_xticks(range(8),names,rotation=20,ha='right');ax.set_ylabel('Correct images / 16');ax.set_ylim(0,18);ax.legend(ncol=2,fontsize=8)
save('prompts',fig)
fig,ax=plt.subplots(figsize=(7.7,2.5),layout='constrained')
for j,(l,vals) in enumerate([('Text answer',[25,12,13]),('Fixed-state depth',[30,16,14]),('Expert depth',[26,16,10])]):
    b=ax.bar(np.arange(3)+(j-1)*.24,vals,.24,label=l);ax.bar_label(b,padding=2,fontsize=8)
ax.set_xticks(range(3),['All (n=32)','Size agrees (n=16)','Size conflicts (n=16)']);ax.set_ylim(0,35);ax.set_ylabel('Correct images');ax.legend(fontsize=8,ncol=3)
save('balanced',fig)
print(OUT)
