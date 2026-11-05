"""Label-preserving descriptive input audit over all 1094 held Training clips."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv, json, sys, hashlib, datetime, re, time
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent
REPO=R.parents[1]
sys.path.insert(0,str(REPO/'deliverables/activity_token_stopgrad_candidate_20260913'))
from activity_inputs import box_from_thumbnails
import torch
torch.set_num_threads(1)

def span(files):
    times=[]
    for p in files:
        m=re.search(r'(20\d\d-\d\d-\d\d)[_ ](\d\d)[-:](\d\d)[-:](\d\d)(?:[._](\d{3,6}))?',p.name)
        if m:
            a,b,c,d,e=m.groups();times.append(datetime.datetime.fromisoformat(a+'T'+b+':'+c+':'+d+'.'+(e or '000')).timestamp())
    return dict(parsed=len(times),first=min(times) if times else None,last=max(times) if times else None,duration_s=max(times)-min(times) if times else None)

def one(row):
    out=dict(clip_id=row['clip_id'],user=int(row['user']),target=int(row['true_class']),R6_correct=int(row['prediction'])==int(row['true_class']),fold=int(row['fold']))
    for mod in ['IR','Depth_Color','Thermal']:
        directory=Path(row['IR_directory'])
        if mod!='IR':directory=directory.parents[3]/mod/directory.parents[1].name/directory.parent.name/directory.name
        files=sorted(p for p in directory.glob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg'])
        if not files:out[mod]=dict(frames=0);continue
        frames=[];native=None;failures=[]
        indices=sorted(set(min(len(files)-1,int((k+.5)*len(files)/16)) for k in range(16)))
        for i in indices:
            try:
                with Image.open(files[i]) as im:
                    im=im.convert('RGB');native=im.size;scale=120/min(native)
                    frames.append(np.asarray(im.resize(tuple(round(v*scale) for v in native),Image.Resampling.BILINEAR)))
            except Exception as e:failures.append(dict(path=str(files[i]),error=str(e)))
        d=dict(frames=len(files),timestamp=span(files),decode_failures=failures)
        if frames:
            a=np.stack(frames).astype(np.float32)/255
            d.update(native_size=list(native),gray_mean=float(a.mean()),gray_std=float(a.mean(-1).std()),rgb_mean=a.mean((0,1,2)).tolist(),rgb_std=a.std((0,1,2)).tolist(),p05_p50_p95=np.quantile(a.mean(-1),[.05,.5,.95]).tolist(),region=box_from_thumbnails(np.stack(frames),native),sampled_unique_frames=len(frames))
        out[mod]=d
    return out

if __name__=='__main__':
    source=REPO/'runs/joint_relation_20260913/analysis/ALL_1094_COMPARISONS.csv'
    rows=list(csv.DictReader(source.open(encoding='utf-8-sig',newline='')))
    path=R/'audit/ALL_1094_INPUT_GEOMETRY.json'
    assert not path.exists()
    start=time.time()
    with ProcessPoolExecutor(max_workers=3) as pool:
        values=[]
        for i,v in enumerate(pool.map(one,rows),1):
            values.append(v)
            if i%100==0:print(json.dumps(dict(completed=i,total=len(rows))),flush=True)
    summaries={}
    for correct in [False,True]:
        ss=[v for v in values if v['R6_correct']==correct];d=dict(n=len(ss))
        for mod in ['IR','Depth_Color','Thermal']:
            available=[v[mod] for v in ss if v[mod].get('region')]
            reasons={r:sum(v['region']['reason']==r for v in available) for r in sorted({v['region']['reason'] for v in available})}
            d[mod]=dict(available=len(available),crop_reasons=reasons,mean_gray_mean=float(np.mean([v['gray_mean'] for v in available])),median_gray_std=float(np.median([v['gray_std'] for v in available])))
        summaries[str(correct)]=d
    payload=dict(status='DESCRIPTIVE_ALL_HELD_INPUT_AUDIT_COMPLETE',at_unix=time.time(),seconds=time.time()-start,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),Test_read=False,training_fits=0,causal_interpretation=False,summary=summaries,cases=values)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries),flush=True)
