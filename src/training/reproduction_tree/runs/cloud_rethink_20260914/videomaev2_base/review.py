"""Fixed two-outer-fold audit; source-unmatched fusion is diagnostic only."""
from pathlib import Path
import importlib.util,json,sys,time
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1];F=C/'videomae_kinetics_fusion';sys.path[:0]=[str(F),str(C)]
sp=importlib.util.spec_from_file_location('v2_base_endpoint_helpers',F/'worker.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);a=m.a;np=m.np
def main():
 reg=a.read(E/'REVIEW_REGISTRATION.json');assert a.sha(E/'review.py')==reg['worker_sha256'] and time.time()<reg['deadline_unix'];ph=reg['protocol_sha256'];s=a.read(a.check(E/'protocol.json',ph));out=E/'review';out.mkdir(exist_ok=True);assert not(out/'DONE.json').exists();start=time.time()
 m.torch.set_num_threads(2);root=E/'cloud';m.terminal(root,ph)
 current=a.arr(a.check(C/'videomaev2_fusion/output/PAIRED.npz',reg['current_v2_paired_sha256']));oldall=a.arr(a.check(C/'videomae_base_kinetics/review/PAIRED.npz',reg['old_base_paired_sha256']));parts=[];sampling={}
 for f in [0,3]:
  job=next(j for j in s['jobs'] if j['kind']=='outer' and j['fold']==f);p=root/job['id'];fr=a.read(p/'FROZEN.json');assert fr['epochs']==16 and fr['train_users']==job['train_users'] and fr['held_users']==job['held_users'] and fr['FP16_reload_exact']
  n=a.arr(p/'predictions.npz');oldroot=C/'videomae_base_kinetics/cloud'/('BASE_K400_F'+str(f));old=a.align(a.arr(oldroot/'predictions.npz'),n)
  hist=a.read(p/'HISTORY.json');oh=a.read(oldroot/'HISTORY.json');assert len(hist)==len(oh)==16 and [x['sampling_sha256'] for x in hist]==[x['sampling_sha256'] for x in oh];sampling[str(f)]=True
  subset=lambda v:{k:x[np.isin(v['users'],job['held_users'])] for k,x in v.items()}
  cur=a.align(subset(current),n);oldmatched=a.align(subset(oldall),n);assert np.array_equal(old['videomae_logits'],oldmatched['base'])
  soft=lambda z:np.exp(z-m.cw.logsumexp(z,-1,True));v={k:n[k] for k in ['clip_ids','targets','users']};v.update(old_base=old['videomae_logits'],v2_base=n['videomae_logits'],v2_small=cur['raw_v2'],v2_global7=cur['global7'])
  v['fixed_diagnostic']=np.log(.8*soft(cur['global7'].astype(np.float64))+.2*soft(n['videomae_logits'].astype(np.float64))).astype(np.float32);parts.append(v)
 v={k:np.concatenate([x[k] for x in parts]) for k in parts[0]};assert len(v['targets'])==1094 and len(set(v['clip_ids']))==1094
 result=dict(status='COMPLETE_V2_BASE_TWO_OUTER_AUDIT',protocol_sha256=ph,private_vs_K400Base=a.pair(v['targets'],v['users'],v['old_base'],v['v2_base']),private_vs_V2Small=a.pair(v['targets'],v['users'],v['v2_small'],v['v2_base']),private_vs_current_whole=a.pair(v['targets'],v['users'],v['v2_global7'],v['v2_base']),fixed_diagnostic=a.pair(v['targets'],v['users'],v['v2_global7'],v['fixed_diagnostic']),source_sampling_exact=sampling,local_provisional_source_not_used=True,no_Test=True,no_new_CSV=True,limitations='Repeated development users. Fixed mixture is source-unmatched and not eligible promotion. Actual INT8 and source-only TI calibration required; no Public guarantee.',seconds=time.time()-start,at_unix=time.time())
 assert time.time()<reg['deadline_unix'] and time.time()-start<reg['wall_cap_seconds'];np.savez_compressed(out/'PAIRED.npz',**v);a.atomic(out/'RESULT.json',result);a.atomic(out/'DONE.json',dict(status='PASS',review_registration_sha256=a.sha(E/'REVIEW_REGISTRATION.json'),artifacts={n:a.sha(out/n) for n in ['RESULT.json','PAIRED.npz']}))
 with(R/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='V2_BASE_TWO_OUTER_AUDIT_COMPLETE',result=str((out/'RESULT.json').relative_to(R)),result_sha256=a.sha(out/'RESULT.json'),at_unix=time.time()))+'\n')
 print(json.dumps({k:{n:v for n,v in x.items() if n not in ['per_user','per_class']} for k,x in result.items() if isinstance(x,dict)}))
if __name__=='__main__':main()
