"""Geometry-validated distribution shifts for the frozen depth readout."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .io import new_output, write_json, write_jsonl
from .scenes import annulus


STRATA=('size_variation','ellipsoids','boxes','partial_occlusion','camera_variation')


def render(centers,axes,kinds,colors,focal_scale,ground,size=336):
    focal=focal_scale*size
    yy,xx=np.mgrid[:size,:size]
    rays=np.stack([(xx+.5-size/2)/focal,(yy+.5-size/2)/focal,np.ones_like(xx)],-1)
    depth=np.full((size,size),100.)
    ids=np.full((size,size),-1,dtype=np.int16)
    rgb=np.full((size,size,3),[.55,.65,.75])
    floor=np.divide(ground,rays[...,1],out=np.full_like(depth,100.),where=rays[...,1]>0)
    visible=(floor>0)&(floor<depth)
    depth[visible]=floor[visible]
    xyz=rays*floor[...,None]
    checker=((np.floor(xyz[...,0])+np.floor(xyz[...,2]))%2)*.08+.55
    rgb[visible]=np.repeat(checker[...,None],3,-1)[visible]
    light=np.array([-.4,-.8,-.6]); light/=np.linalg.norm(light)
    marks=[]
    for i,(center,axis,kind,color) in enumerate(zip(centers,axes,kinds,colors)):
        center,axis=np.asarray(center),np.asarray(axis)
        if kind=='box':
            lo=np.divide(center-axis,rays,out=np.full_like(rays,-np.inf),where=rays!=0)
            hi=np.divide(center+axis,rays,out=np.full_like(rays,np.inf),where=rays!=0)
            entry=np.minimum(lo,hi).max(-1); leave=np.maximum(lo,hi).min(-1)
            hit=(leave>=entry)&(entry>0)
            t=np.where(hit,entry,100.)
            coords=(rays*t[...,None]-center)/axis
            face=np.argmax(np.abs(coords),-1)
            normals=np.eye(3)[face]*np.take_along_axis(np.sign(coords),face[...,None],axis=-1)
        else:
            a=((rays/axis)**2).sum(-1)
            dot=np.einsum('hwc,c->hw',rays,center/axis**2)
            disc=dot**2-a*((center/axis)@(center/axis)-1)
            t=(dot-np.sqrt(np.maximum(disc,0)))/a
            hit=(disc>=0)&(t>0)
            normals=(rays*t[...,None]-center)/axis**2
            normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-12)
        visible=hit&(t<depth)
        shading=.3+.7*np.maximum(0,normals@light)
        rgb[visible]=(np.asarray(color)*shading[...,None])[visible]
        depth[visible]=t[visible]; ids[visible]=i
        marks.append([int(round(focal*center[0]/center[2]+size/2-.5)),
                      int(round(focal*center[1]/center[2]+size/2-.5))])
    return Image.fromarray((np.clip(rgb,0,1)*255).astype('uint8')),depth.astype('float32'),ids,marks


def generate_shift(out,families=50,seed=1901,size=336):
    if families<10 or families%5:
        raise ValueError('Use at least ten families, balanced over five strata.')
    out=new_output(out); rng=np.random.default_rng(seed); rows=[]; rejected=0
    try: font=ImageFont.truetype('DejaVuSans-Bold.ttf',24)
    except OSError: font=ImageFont.load_default(size=24)
    for family in range(families):
        stratum=STRATA[family%5]
        for attempt in range(200):
            z=np.array([rng.uniform(4.1,4.9),rng.uniform(6.0,7.2)])
            if family%2: z=z[::-1]
            radius=rng.uniform(.45,1.15,2)
            axes=np.repeat(radius[:,None],3,axis=1)
            kinds=['ellipsoid','ellipsoid']
            focal=.9; ground=1.3; screen=np.array([-.18,.18])
            if stratum=='ellipsoids': axes*=rng.uniform(.7,1.4,(2,3))
            if stratum=='boxes': kinds=['box','box']; axes*=.8
            if stratum=='partial_occlusion': screen=np.array([-.095,.095]); axes*=.85
            if stratum=='camera_variation': focal=rng.uniform(.7,1.15); ground=rng.uniform(1.05,1.65)
            colors=rng.uniform(.25,.95,(2,3))
            order0=[0,1] if family%4<2 else [1,0]
            prepared=[]
            for variant in ('base','depth_swap','nuisance','label_swap'):
                current_z=z[::-1] if variant=='depth_swap' else z
                centers=np.column_stack([screen*current_z,ground-axes[:,1],current_z])
                current_colors=1-colors*.65 if variant=='nuisance' else colors
                image,depth,ids,marks=render(centers,axes,kinds,current_colors,focal,ground,size)
                order=order0[::-1] if variant=='label_swap' else order0
                points=[marks[i] for i in order]
                valid=all(20<x<size-45 and 20<y<size-25 for x,y in points)
                masks=[annulus(depth.shape,xy) for xy in points]
                valid=valid and all(np.all(ids[mask]==obj) for mask,obj in zip(masks,order))
                if not valid: break
                point_z=[float(depth[y,x]) for x,y in points]
                patch_z=[float(np.median(depth[m])) for m in masks]
                if abs(point_z[0]-point_z[1])<.4 or (point_z[0]<point_z[1])!=(patch_z[0]<patch_z[1]): break
                # Occlusion stratum must contain actual overlap in projected isolated masks.
                if stratum=='partial_occlusion' and variant=='base':
                    _,_,id0,_=render(centers[:1],axes[:1],kinds[:1],current_colors[:1],focal,ground,size)
                    _,_,id1,_=render(centers[1:],axes[1:],kinds[1:],current_colors[1:],focal,ground,size)
                    if np.count_nonzero((id0==0)&(id1==0))<15: break
                prepared.append((variant,image,depth,ids,points,point_z,centers,current_colors))
            if len(prepared)==4:
                labels=['A' if p[5][0]<p[5][1] else 'B' for p in prepared]
                if labels[0]!=labels[1] and labels[0]!=labels[3] and labels[0]==labels[2]: break
            rejected+=1
        else: raise RuntimeError(f'Failed geometry validation: {stratum}')
        for variant,image,depth,ids,points,point_z,centers,current_colors in prepared:
            sample=f'shift_{family:05d}_{variant}'; folder=out/sample; folder.mkdir()
            image.save(folder/'unmarked.png')
            draw=ImageDraw.Draw(image)
            for label,(x,y) in zip(('A','B'),points):
                draw.ellipse((x-3,y-3,x+3,y+3),fill='white',outline='black')
                draw.text((x+12,y-18),label,font=font,fill='white',stroke_width=2,stroke_fill='black')
            image.save(folder/'image.png')
            np.savez_compressed(folder/'geometry.npz',depth=depth,object_ids=ids)
            rows.append(dict(id=sample,family=f'shift_{family:05d}',split='pilot',stratum=stratum,
                variant=variant,image=f'{sample}/image.png',geometry=f'{sample}/geometry.npz',
                points=points,point_depths=point_z,label='A' if point_z[0]<point_z[1] else 'B',
                question='Which marked point is closer to the camera, A or B? Answer only A or B.',
                camera={'focal_px':focal*size,'size':[size,size],'ground_y':ground},
                objects={'centers':centers.tolist(),'axes':axes.tolist(),'kinds':kinds,'colors':current_colors.tolist()},
                inner_radius=5,outer_radius=9,native_audit=family<10,
                renderer='analytic_shapes_v2',marker_font_size=24))
    write_jsonl(out/'manifest.jsonl',rows)
    write_json(out/'dataset.json',dict(seed=seed,families=families,strata=STRATA,examples=len(rows),
        geometry_rejections=rejected,native_audit_families=10,
        selection='First two families of each stratum, before any model results. All others fixed-state/expert only.',
        calibration='Frozen prior run calibration; all new scenes are evaluation.'))
    return rows
