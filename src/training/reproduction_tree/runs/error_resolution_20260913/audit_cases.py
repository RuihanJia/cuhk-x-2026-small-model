"""Complete held-user error inventory and raw/input image evidence. No learning."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,csv,json,hashlib,sys,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

R=Path(__file__).resolve().parent;REPO=R.parents[1];OLD=REPO/'research/next_route_error_audit_20260913';J=REPO/'runs/joint_relation_20260913'
PKG=REPO/'deliverables/activity_token_stopgrad_candidate_20260913'
sys.path.insert(0,str(PKG));from activity_inputs import load_activity_five
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))

def prepare():
    assert not (R/'audit/INVENTORY.json').exists()
    source=J/'analysis/ALL_1094_COMPARISONS.csv';base=rows(source);recent=rows(J/'independent_fusion/PAIRED_CASES.csv');rec={x['ids']:x for x in recent};assert len(base)==len(rec)==1094
    verified=read(OLD/'metrics/metrics.json')
    assert sha(OLD/'metrics/cases.csv')==verified['case_table_sha256']
    for v in verified['sources']:
        assert sha(v['video_source'])==v['video_sha256'] and sha(v['final_array'])==v['final_sha256']
    originals={x['clip_id']:x for x in rows(J/'analysis/ALL_275_ERRORS.csv')}
    cases=[];extra=[];counts={n:0 for n in ['R6','P0','C128','J3']}
    for r in base:
        j=rec[r['clip_id']];assert int(r['true_class'])==int(j['y']) and int(r['user'])==int(j['u']) and int(r['prediction'])==int(j['R6'])
        pred=dict(R6=int(r['prediction']),P0=int(r['P0_prediction']),C128=int(r['C128_prediction']),J3=int(j['J3']));y=int(r['true_class']);wrong=[k for k,v in pred.items() if v!=y]
        for k in wrong:counts[k]+=1
        if wrong:
            x=dict(r,predictions=pred,wrong_submitted_recipes=wrong,joint_independent_prediction=int(j['head']),scope='Training held users; supplied truth unchanged')
            if r['clip_id'] in originals:x.update({k:v for k,v in originals[r['clip_id']].items() if k not in x});x['case_id']=originals[r['clip_id']]['error_id'];cases.append(x)
            else:extra.append(x)
    cases.sort(key=lambda x:x['case_id'])
    for i,x in enumerate(sorted(extra,key=lambda x:x['clip_id']),276):x['case_id']=f'E{i:03d}';cases.append(x)
    assert counts==dict(R6=275,P0=273,C128=273,J3=274) and len(cases)==285
    cache=dict(np.load(J/'data/sensors.npz',allow_pickle=False));ix={str(c):i for i,c in enumerate(cache['clip_ids'])}
    for r in cases:
        i=ix[r['clip_id']];assert int(cache['targets'][i])==int(r['true_class']);x=cache['x'][i];sk=cache['skeleton'][i]
        r['sensor_numeric']=dict(acc_observed_fraction=cache['acc_mask'][i].mean(-1).tolist(),gyro_observed_fraction=cache['gyro_mask'][i].mean(-1).tolist(),raw_resampled_axis_std=x.std(-1).tolist(),skeleton_temporal_mean_abs_delta=float(np.abs(np.diff(sk,axis=0)).mean()),all_finite=bool(np.isfinite(x).all() and np.isfinite(sk).all()),interpretation='Mask and resampled variability facts, not evidence that labels are identifiable')
    inventory=dict(status='285_CASES_REGISTERED_BEFORE_VISUAL_REVIEW',at_unix=time.time(),records=1094,error_counts=counts,union=285,cases=cases,sources={str(source):sha(source),str(J/'independent_fusion/PAIRED_CASES.csv'):sha(J/'independent_fusion/PAIRED_CASES.csv'),str(OLD/'metrics/cases.csv'):sha(OLD/'metrics/cases.csv')},script_sha256=sha(Path(__file__)),activity_input_code_sha256=sha(PKG/'activity_inputs.py'),Test_read=False,training_fits=0,visual_reviews_completed=0)
    save(R/'audit/INVENTORY.json',inventory);print(json.dumps(dict(status=inventory['status'],error_counts=counts,union=len(cases))),flush=True)

def tile(pic,size):
    pic=pic.convert('RGB');pic.thumbnail(size,Image.Resampling.LANCZOS);o=Image.new('RGB',size,(225,227,232));o.paste(pic,((size[0]-pic.width)//2,(size[1]-pic.height)//2));return o
def render_one(r):
    case=r['case_id'];path=R/'audit/images'/f'{case}.jpg';meta_path=R/'audit/images'/f'{case}.json'
    if path.exists() and meta_path.exists():
        m=read(meta_path);assert sha(path)==m['image_sha256'];return case
    canvas=Image.new('RGB',(1792,1340),'white');d=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    d.text((5,5),f"{case} {r['clip_id']} | True {r['true_class']} {r['true_name']} | predictions {r['predictions']}",font=font,fill='black')
    d.text((5,33),f"V/S/T/I top1: {r['V_prediction']}/{r['S_prediction']}/{r['T_prediction']}/{r['I_prediction']} | True rank {r['true_rank']} | Rescue {r['rescuing_branches'] or 'none'} | Jhead {r['joint_independent_prediction']}",font=small,fill='black')
    sources=[];regions={};invalid={};nframes={};exact_input={}
    for mod,top in [('IR',86),('Depth_Color',302)]:
        directory=Path(r[mod+'_directory']);assert 'Training' in directory.parts;files=sorted(directory.glob('*.png'))
        x,sub,bad,reg=load_activity_five(directory);assert x is not None
        regions[mod]=reg;invalid[mod]=dict(substitutions=sub,invalid=bad);nframes[mod]=len(files)
        d.text((5,top-23),f"Native {mod}: {len(files)} frames | {reg['reason']}; red = current local crop",font=small,fill='black')
        for j,k in enumerate(np.linspace(0,len(files)-1,8).round().astype(int)):
            with Image.open(files[k]) as h:pic=h.convert('RGB')
            if reg.get('box'):
                xx,yy,ww,hh=reg['box'];ImageDraw.Draw(pic).rectangle((xx,yy,xx+ww,yy+hh),outline=(255,35,35),width=4)
            canvas.paste(tile(pic,(224,168)),(224*j,top));d.text((224*j+4,top+170),f'frame {k}',font=small,fill='black');sources.append(dict(path=str(files[k]),sha256=sha(files[k])))
        if mod=='IR':
            for spatial,y,label in [(0,536,'Actual global IR input, center phase'),(1,807,'Actual local IR input, center phase')]:
                d.text((5,y-23),label,font=font,fill='black')
                values=(x[0,spatial]*.225+.45).clamp(0,1).mul(255).round().byte()
                exact_input[label]=hashlib.sha256(values.numpy().tobytes()).hexdigest().upper()
                for j,k in enumerate([0,2,4,6,8,10,12,15]):canvas.paste(Image.fromarray(values[:,k].permute(1,2,0).numpy()),(224*j,y))
        del x
    ir=Path(r['IR_directory']);thermal=ir.parents[3]/'Thermal'/ir.parents[1].name/ir.parent.name/ir.name
    tf=sorted(f for f in thermal.glob('*') if f.suffix.lower() in {'.jpg','.jpeg','.png'});d.text((5,1056),f'Native Thermal: {len(tf)} frames (first/middle/last); not assumed synchronized with IR',font=small,fill='black')
    for j,k in enumerate(np.linspace(0,len(tf)-1,4).round().astype(int) if tf else []):
        with Image.open(tf[k]) as h:pic=h.convert('RGB')
        canvas.paste(tile(pic,(224,216)),(224*j,1080));sources.append(dict(path=str(tf[k]),sha256=sha(tf[k])))
    sn=r['sensor_numeric'];d.text((920,1088),'Measured sensor coverage (five devices):',font=small,fill='black')
    d.text((920,1120),'ACC '+str([round(v,2) for v in sn['acc_observed_fraction']]),font=small,fill='black');d.text((920,1150),'GYRO '+str([round(v,2) for v in sn['gyro_observed_fraction']]),font=small,fill='black')
    d.text((920,1180),'Skeleton mean abs delta '+str(round(sn['skeleton_temporal_mean_abs_delta'],4)),font=small,fill='black')
    d.text((920,1210),'All finite '+str(sn['all_finite'])+' | Truth labels retained',font=small,fill='black')
    d.text((5,1312),'Evidence montage: sampled frames and exact input pixels; not a full video replay or a causal diagnosis.',font=small,fill='black')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,quality=87)
    save(meta_path,dict(case_id=case,clip_id=r['clip_id'],source_images=sources,regions=regions,decode=invalid,native_frames=nframes,thermal_frames=len(tf),exact_IR_center_input_u8_hashes=exact_input,image_sha256=sha(path),script_sha256=sha(Path(__file__)),Test_read=False,visually_reviewed=False))
    return case

def render():
    inv=read(R/'audit/INVENTORY.json');assert inv['script_sha256']==sha(Path(__file__)) and inv['activity_input_code_sha256']==sha(PKG/'activity_inputs.py')
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i,name in enumerate(pool.map(render_one,inv['cases']),1):
            if i%10==0:print(json.dumps(dict(rendered=i,total=len(inv['cases']),last=name)),flush=True)
    save(R/'audit/RENDER_DONE.json',dict(status='ALL_285_EVIDENCE_MONTAGES_RENDERED',count=len(inv['cases']),at_unix=time.time(),inventory_sha256=sha(R/'audit/INVENTORY.json'),visual_review_still_required=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--render',action='store_true');a=ap.parse_args()
    if a.prepare:prepare()
    if a.render:render()
