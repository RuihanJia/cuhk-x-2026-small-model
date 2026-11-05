"""Read-only paired diagnostics of a frozen, exposed-development candidate."""
from pathlib import Path
import json,csv,time,hashlib,html,collections
import numpy as np
import calibrate_pose_complement as c
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    recipe=next(x for x in read(R/'POSE_COMPLEMENT_CALIBRATION_PROTOCOL.json')['recipes'] if x['id']=='POSE_DUAL6')
    samples=[];refs={};branches=['V_context','Skeleton','Thermal','IMU','Pose_frozen','Pose_private'];tags={}
    with (R/'audit/CASE_REVIEW.csv').open(encoding='utf-8-sig') as f:
        for q in csv.DictReader(f):
            refs.setdefault(q['clip_id'],[]).append(dict(type='original',case_id=q['case_id'],image='images/'+q['case_id']+'.jpg',note=q['visible_evidence'],uncertain=q['uncertain_interpretation']));tags[q['clip_id']]=q['tags']
    for fn,kind,key in [('ENDPOINT_OBSERVATIONS.jsonl','endpoint','observation'),('crop_contrasts/OBSERVATIONS.jsonl','crop','note')]:
        latest={q['case_id']:q for q in [json.loads(line) for line in (R/'audit'/fn).read_text(encoding='utf-8').splitlines()]}
        for q in latest.values():
            assert q.get('actually_viewed',q.get('viewed',False));im=('crop_contrasts/' if kind=='crop' else 'images/')+q['case_id']+'.jpg';assert sha(R/'audit'/im)==q.get('image_sha256',q.get('viewed_image_sha256'));refs.setdefault(q['clip_id'],[]).append(dict(type=kind,case_id=q['case_id'],image=im,note=q.get(key,q.get('visible_evidence',''))))
    for f in [0,3]:
        data,ref,prior=c.outer(f,recipe);p=R/'pose_complement_calibration'/f'F{f}_POSE_DUAL6/predictions.npz';a=dict(np.load(p,allow_pickle=False));assert np.array_equal(a['clip_ids'],data['clip_ids']);pred=a['fused'].argmax(1);base=ref.argmax(1)
        for i,k in enumerate(a['clip_ids']):
            y=int(a['targets'][i]);votes=data['logits'][i].argmax(1);av=data['available'][i].astype(bool);correct=[branches[j] for j in range(6) if av[j] and votes[j]==y];q=dict(clip_id=str(k),fold=f,user=int(a['users'][i]),target=y,class_name=str(k).split('/')[0],R6=int(base[i]),candidate=int(pred[i]),before_correct=bool(base[i]==y),after_correct=bool(pred[i]==y),delta=int(pred[i]==y)-int(base[i]==y),candidate_truth_rank=int(1+np.sum(a['fused'][i]>a['fused'][i,y])),candidate_top2=[int(v) for v in np.argsort(-a['fused'][i])[:2]],correct_branches=correct,branch_top1={name:int(votes[j]) if av[j] else None for j,name in enumerate(branches)},available_count=int(av.sum()),tags=tags.get(str(k),''),viewed_evidence=refs.get(str(k),[]));samples.append(q)
    assert len(samples)==1094 and len({q['clip_id'] for q in samples})==1094
    def paired(rows):
        return dict(n=len(rows),before=sum(q['before_correct'] for q in rows),after=sum(q['after_correct'] for q in rows),fix=sum(q['delta']==1 for q in rows),harm=sum(q['delta']==-1 for q in rows),net=sum(q['delta'] for q in rows))
    per_user={u:paired([q for q in samples if q['user']==u]) for u in sorted({q['user'] for q in samples})};per_class={y:dict(name=next(q['class_name'] for q in samples if q['target']==y),**paired([q for q in samples if q['target']==y])) for y in range(40)};errors=[q for q in samples if not q['after_correct']];assert len(errors)==239
    unviewed=[q['clip_id'] for q in errors if not q['viewed_evidence']];assert not unviewed,unviewed
    original_latest={}
    for line in (R/'audit/OBSERVATIONS.jsonl').read_text(encoding='utf-8').splitlines():
        q=json.loads(line);original_latest[q['case_id']]=q
    assert len(original_latest)==285
    for k,q in original_latest.items():assert q['viewed'] and sha(R/'audit/images'/f'{k}.jpg').upper()==q['viewed_image_sha256'].upper(),k
    for q in errors:
        for e in q['viewed_evidence']:
            if e['type']=='original':assert e['case_id'] in original_latest
    uv=np.array([[v['net'],v['n']] for v in per_user.values()]);rng=np.random.default_rng(20260914);idx=rng.integers(0,len(uv),(20000,len(uv)));draw=uv[idx].sum(1);boot=draw[:,0]/draw[:,1];sensitivity={u:dict(net=36-v['net'],n=1094-v['n'],delta_accuracy=(36-v['net'])/(1094-v['n'])) for u,v in per_user.items()}
    conf=collections.Counter((q['target'],q['candidate']) for q in errors);correct_hist=collections.Counter(x for q in errors for x in q['correct_branches']);no_branch=[q for q in errors if not q['correct_branches']]
    result=dict(at_unix=time.time(),recipe='POSE_DUAL6',selection_frozen_before_Test=True,pooled=paired(samples),per_user=per_user,per_class=per_class,remaining_errors=dict(n=len(errors),all_6_available_branch_top1_wrong=len(no_branch),at_least_one_available_branch_correct=len(errors)-len(no_branch),fused_truth_in_top2=sum(q['candidate_truth_rank']<=2 for q in errors),fused_truth_in_top5=sum(q['candidate_truth_rank']<=5 for q in errors),correct_branch_support=dict(correct_hist),largest_confusions=[dict(target=y,predicted=p,n=n) for (y,p),n in conf.most_common(20)],all_have_actually_viewed_evidence=True),leave_one_user_out_sensitivity=sensitivity,user_cluster_bootstrap=dict(seed=20260914,resamples=20000,percentile_95_interval_delta_accuracy=np.quantile(boot,[.025,.975]).tolist(),interpretation='Descriptive resampling of seven repeatedly exposed development users, conditional on selected candidate. NOT an unbiased confidence interval for unseen users or Kaggle Public; does not account for architecture selection.'),source_hashes={str(p.relative_to(R)):sha(p) for p in [R/'pose_complement_calibration/ALL_PAIRED_CASES.csv',R/'audit/OBSERVATIONS.jsonl',R/'audit/ENDPOINT_OBSERVATIONS.jsonl',R/'audit/crop_contrasts/OBSERVATIONS.jsonl']},no_fits=True,no_Test_labels=True,no_Public_tuning=True)
    out=R/'audit/selected_candidate';out.mkdir(exist_ok=True);(out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(out/'ALL_1094_CASES.json').write_text(json.dumps(samples,ensure_ascii=False));
    for fn,rows in [('ALL_1094_CASES.csv',samples),('REMAINING_239_ERRORS.csv',errors)]:
        with (out/fn).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in q.items()} for q in rows)
    cards=[]
    for q in sorted(samples,key=lambda q:(q['after_correct'],-abs(q['delta']),q['target'],q['user'],q['clip_id'])):
        ev=''.join('<details><summary>'+html.escape(e['case_id']+' '+e['type'])+'</summary><img loading="lazy" src="../'+html.escape(e['image'])+'"><p>'+html.escape(e['note'])+'</p>'+('<p>不确定：'+html.escape(e.get('uncertain',''))+'</p>' if e.get('uncertain') else '')+'</details>' for e in q['viewed_evidence']);h=html.escape(q['clip_id']);status='仍错' if not q['after_correct'] else ('修正' if q['delta']==1 else '正确');cards.append('<article data-status="'+status+'"><h3>'+h+' · '+status+'</h3><p>真值 '+str(q['target'])+'｜R6 '+str(q['R6'])+' → 新模型 '+str(q['candidate'])+'｜真值排名 '+str(q['candidate_truth_rank'])+'</p><p>正确分支：'+html.escape(', '.join(q['correct_branches']) or '所有可用分支都错')+'</p><p>'+html.escape(json.dumps(q['branch_top1'],ensure_ascii=False))+'</p>'+ev+'</article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逐例错误与改进证据</title><style>body{font:16px/1.6 system-ui;max-width:1180px;margin:32px auto;background:#f5f7fa;color:#152334}article{background:white;border:1px solid #dbe1e7;padding:18px;margin:14px 0;border-radius:10px}img{max-width:100%}input,select{font:inherit;padding:10px}header{position:sticky;top:0;background:#f5f7fa;padding:12px 0}h1{font-size:28px}summary{cursor:pointer;color:#174e83}aside{padding:18px;background:#e9eef5}</style><h1>冻结候选：逐例错误与改进证据</h1><aside>开发集1094条：819→855，修68、损32、剩239。七个用户均净正；开发折反复暴露，不能解释为Kaggle/Public已达0.75。全部239剩余错误都有实际看图记录。分支中有人答对只是诊断上限，不是可直接实现的规则。所有观察仅供分析，没有修改标签或写入模型。<br>可展开原始多模态图和旧/新裁剪对照；正确但没有观察记录的普通例不声称已看。</aside><header><input id="q" placeholder="搜索类别、用户、片段或观察"><select id="s"><option value="">全部1094</option><option selected>仍错</option><option>修正</option><option>正确</option></select><span id="count"></span></header>'''+''.join(cards)+'''<script>const q=document.getElementById('q'),s=document.getElementById('s');function update(){let n=0;for(const c of document.querySelectorAll('article')){const ok=(!s.value||c.dataset.status===s.value)&&c.textContent.toLowerCase().includes(q.value.toLowerCase());c.hidden=!ok;n+=ok;}document.getElementById('count').textContent=' '+n+' 条';}q.addEventListener('input',update);s.addEventListener('change',update);update();</script></html>''';(out/'ERROR_ATLAS.html').write_text(page,encoding='utf-8')
    with (R.parents[1]/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='SELECTED_CANDIDATE_PAIRED_ERROR_AUDIT',artifact=str((out/'RESULT.json').relative_to(R.parents[1])),at_unix=time.time(),pooled=result['pooled'],remaining_errors=result['remaining_errors'],no_new_fits=True))+'\n')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
