"""Source-only bounded CPU QA for the exact fixed photometric transform."""
from pathlib import Path
import os,sys,json,time,hashlib,random,threading,traceback
P=Path(__file__).resolve().parent;D=P.parent;C=D.parents[1];O=C.parent/'error_resolution_20260913';out=P/'CPU_QA_FAST.json';assert not out.exists()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(r):
    tmp=out.with_suffix('.partial');tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,out)
beg=time.monotonic();ids=['10_Stir_drinks/1/2-1-3','35_Do_lunges/20/1-2-1','7_Eat_food/1/4-1-2','8_Take_and_use_tableware/1/4-1-2']
r=dict(status='REGISTERED_CPU_PHOTOMETRIC_QA',registered_at_unix=time.time(),wall_cap_seconds=180,qa_source_sha256=sha(__file__),data_sha256=sha(P/'photometric_data_numpy.py'),source_clip_ids=ids,selection='Fixed2 bright/2 dark cases from the previous8source-only visuals; no new clip search.',checks={})
save(r)
def timeout():
    r.update(status='PARTIAL_CPU_QA_TIMEOUT',seconds=time.monotonic()-beg);save(r);os._exit(124)
timer=threading.Timer(180,timeout);timer.daemon=True;timer.start()
try:
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';sys.path[:0]=[str(P),str(D),str(C),str(O),str(O/'bundle')]
    import numpy as np
    import torch
    from torch.utils.data._utils.collate import default_collate
    from photometric_data_numpy import PhotometricClips,PhotometricFiveViews,stretch_normalized
    from pose_view_data import PoseClips,PoseFiveViews
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    ip=O/'POSE_ADAPTER_INPUTS.json';inp=json.loads(ip.read_text(encoding='utf-8'));byid={v['clip_id']:v for v in inp['rows']};rows=[byid[k] for k in ids];root='D:/Datasets/CUHK-X_Small_Model_Track/Small-Model-Track';seed=20260914
    evidence=json.loads((P/'EVIDENCE.json').read_text(encoding='utf-8'));previous={v['clip_id']:v for v in evidence['records']};assert set(ids)<=set(previous)
    assert all(v['user'] in ({1,3,4,5,6,7,8,9,17,18,19,21,22,23,24} if v['user']!=20 else {1,2,3,4,6,9,16,17,18,19,20,22,23,24}) for v in rows)
    def independent(x):
        v=torch.clamp(x*.225+.45,0,1);bounds=torch.quantile(v.reshape(-1),torch.tensor([.01,.99],dtype=torch.float32),interpolation='linear');den=torch.clamp_min(bounds[1]-bounds[0],1/255);return (((v-bounds[0])/den).clamp(0,1)-.45)/.225,bounds,den
    records=[];timing=[];batch=[];evalbatch=[]
    for training,epoch in [(False,0),(True,1)]:
        old=PoseClips(rows,root,seed,inp['regions'],True,training,epoch);raw=PhotometricClips(rows,root,seed,inp['regions'],training,epoch,'RAW_CONTROL');new=PhotometricClips(rows,root,seed,inp['regions'],training,epoch,'CLIP_STRETCH')
        for i,row in enumerate(rows):
            reference=old[i];a=raw[i];b=new[i];expected,lohi,den=independent(reference['x']);assert torch.equal(a['x'],reference['x']) and torch.equal(b['x'],expected)
            assert (a['stretch_low'],a['stretch_high'],a['denominator'])==(0.,1.,1.) and b['stretch_low']==float(lohi[0]) and b['stretch_high']==float(lohi[1]) and b['denominator']==float(den)
            rng=random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{epoch}:{row["clip_id"]}'.encode()).digest()[:8],'big'));relative=[rng.random() if training else .5 for _ in range(16)];flip=bool(training and rng.random()<.5);n=len(row['streams'][0]['files']);indices=[min(n-1,int((k+relative[k])*n/16)) for k in range(16)]
            assert a['frame_indices'].tolist()==b['frame_indices'].tolist()==indices and a['flip']==b['flip']==flip
            assert a['y']==b['y']==row['label'] and a['user']==b['user']==row['user']
            records.append(dict(clip_id=row['clip_id'],training=training,epoch=epoch,raw_bitwise=True,stretch_maxabs=float((b['x']-expected).abs().max()),frame_indices=indices,flip=flip,lo=b['stretch_low'],hi=b['stretch_high'],denominator=b['denominator']))
            if not training:
                assert b['stretch_low']==previous[row['clip_id']]['low01'] and b['stretch_high']==previous[row['clip_id']]['high99'] and b['denominator']==previous[row['clip_id']]['denominator']
                v=(reference['x']*.225+.45).clamp(0,1);flat=v.reshape(-1);qs=torch.tensor([.01,.99],dtype=torch.float32);t=time.perf_counter();torch.quantile(flat,qs,interpolation='linear');ms=(time.perf_counter()-t)*1000
                t=time.perf_counter();stretch_normalized(reference['x']);fullms=(time.perf_counter()-t)*1000
                timing.append(dict(clip_id=row['clip_id'],float32_values=flat.numel(),torch_threads=2,quantile_only_ms=ms,full_transform_ms=fullms,already_warmed_by_exact_qa=True));print(json.dumps(dict(phase='QUANTILE_COST',**timing[-1])),flush=True)
            batch.append(b)
    old=PoseFiveViews(rows,root,inp['regions']);raw=PhotometricFiveViews(rows,root,inp['regions'],'RAW_CONTROL');new=PhotometricFiveViews(rows,root,inp['regions'],'CLIP_STRETCH')
    for i,row in enumerate(rows):
        reference=old[i];a=raw[i];b=new[i];assert torch.equal(a['x'],reference['x'])
        for j in [0,3,4]:
            expected,lohi,den=independent(reference['x'][j]);assert torch.equal(b['x'][j],expected) and b['stretch_low'][j]==lohi[0] and b['stretch_high'][j]==lohi[1] and b['denominator'][j]==den
        assert torch.equal(b['x'][0],b['x'][1]) and torch.equal(b['x'][0],b['x'][2]);evalbatch.append(b)
    collated=default_collate(batch[:4]);ec=default_collate(evalbatch);assert collated['x'].shape==(4,3,16,224,224) and collated['stretch_low'].shape==(4,) and ec['x'].shape==(4,5,3,16,224,224) and ec['stretch_low'].shape==(4,5)
    constant=torch.full((3,16,2,2),(.2-.45)/.225,dtype=torch.float32);v,lo,hi,den=stretch_normalized(constant);assert den==float(torch.tensor(1/255)) and torch.equal(v,torch.full_like(v,-2.))
    assert sha(P/'photometric_data_numpy.py')==r['data_sha256'] and not torch.cuda.is_initialized()
    r.update(status='PASS_CPU_PHOTOMETRIC_DATA_QA',records=records,quantile_timing=timing,quantile_only_median_ms=float(np.median([v['quantile_only_ms'] for v in timing])),full_transform_median_ms=float(np.median([v['full_transform_ms'] for v in timing])),checks=dict(raw_train_and_eval_bitwise=True,stretch_preview_formula_bitwise=True,preview_midpoint_quantiles_exact=True,sampling_and_flip_exact=True,each_view_16_frames_shared_transform=True,first_three_eval_views_bitwise=True,constant_denominator_floor=True,default_collate=True,all_finite=True,hash_stable=True,no_CUDA=True),input_sha256=sha(ip),preview_evidence_sha256=sha(P/'EVIDENCE.json'),algorithm='NUMPY_PARTITION_TORCH_FLOAT32_LINEAR_V1; independently compared against original torch.quantile full-population definition',no_training=True,no_model=True,no_held_predictions=True,no_Test=True,limits='Quantile timing is4warm real source samples on this host with2CPUthreads, not full DataLoader throughput. All four real examples have non-null headarm boxes; raw branch calls original functions for null boxes as well but that geometry is not separately exercised here.')
except BaseException as exc:r.update(status='FAILED_CPU_PHOTOMETRIC_DATA_QA',error=str(exc),traceback=traceback.format_exc())
finally:
    timer.cancel();r['seconds']=time.monotonic()-beg;save(r)
print(json.dumps(dict(status=r['status'],seconds=r['seconds'],data_sha256=r['data_sha256'],qa_source_sha256=r['qa_source_sha256'],qa_sha256=sha(out),quantile_only_median_ms=r.get('quantile_only_median_ms'),full_transform_median_ms=r.get('full_transform_median_ms')),indent=2));raise SystemExit(0 if r['status']=='PASS_CPU_PHOTOMETRIC_DATA_QA' else 2)
