"""Publish diagnostic observations; this output is never a model input."""
from pathlib import Path
import json,time,hashlib,csv,html,collections
R=Path(__file__).resolve().parent;A=R/'audit';ROOT=R.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
allnotes=[json.loads(x) for x in (A/'OBSERVATIONS.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
notes={n['case_id']:n for n in allnotes};assert set(notes)=={f'E{i:03d}' for i in range(1,286)}
full={n['case_id']:n for n in map(json.loads,(A/'FULL_FRAME_OBSERVATIONS.jsonl').read_text(encoding='utf-8').splitlines())}
assert len(full)==20
cards=[];flat=[]
for k,n in sorted(notes.items()):
 assert n['viewed'] and n['used_as_training_label_or_feature'] is False
 img=A/'images'/f'{k}.jpg';assert sha(img).upper()==n['viewed_image_sha256'].upper()
 f=full.get(k);extra=''
 if f:
  assert f['viewed'] and f['used_as_training_label_or_feature'] is False
  for s in f['sheets']:assert sha(s['path']).lower()==s['sha256'].lower()
  links=' · '.join(f'<a href="full_frames/{html.escape(Path(s["path"]).name)}">{html.escape(Path(s["path"]).stem)}</a>' for s in f['sheets'])
  extra=f'<div class="full"><b>完整帧复核</b><p>{html.escape(f["visible_evidence"])}</p><p>仍不确定：{html.escape(f["uncertainty"])}</p>{links}</div>'
 tag=' '.join(n.get('tags',[]));content=' '.join([k,n['clip_id'],tag,n['visible_evidence'],n['uncertain_interpretation']])
 cards.append(f'''<article id="{k}" data-search="{html.escape(content,quote=True)}"><h2>{k} · {html.escape(n['clip_id'])}</h2><p class="tags">{html.escape(tag)}</p><a href="images/{k}.jpg"><img loading="lazy" width="1792" height="1340" src="images/{k}.jpg" alt="{k} 原始与模型输入对照"></a><div class="notes"><p><b>图上可见</b>　{html.escape(n['visible_evidence'])}</p><p><b>仍不确定</b>　{html.escape(n['uncertain_interpretation'])}</p><p><b>待检验机制</b>　{html.escape(n['proposed_program_or_experiment'])}</p>{extra}</div></article>''')
 flat.append(dict(case_id=k,clip_id=n['clip_id'],visible_evidence=n['visible_evidence'],uncertain_interpretation=n['uncertain_interpretation'],proposed_program_or_experiment=n['proposed_program_or_experiment'],tags=tag,full_frame_review=f['visible_evidence'] if f else '',full_frame_uncertainty=f['uncertainty'] if f else '',used_for_training=False,image_sha256=sha(img)))
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>285例错误审阅证据</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#18283b;background:#edf1f5;margin:0}header,main{max-width:1250px;margin:auto;padding:24px}header{position:sticky;top:0;background:#edf1f5ed;z-index:2;backdrop-filter:blur(8px);border-bottom:1px solid #bbc7d6}h1{font-size:24px;margin:0}header p{margin:8px 0}input{padding:12px;width:75%;font:inherit;border:1px solid #78899e;border-radius:6px}article{background:white;border:1px solid #cbd5df;margin:24px 0;border-radius:10px;overflow:hidden}h2,.tags,.notes{padding:0 22px}h2{font-size:20px}img{display:block;max-width:100%;height:auto}.tags{color:#536d88;font-size:13px}.notes{padding-bottom:20px}.full{border-left:4px solid #00796b;padding-left:16px;background:#eef8f4}a{color:#086bc0}[hidden]{display:none}</style><header><h1>285例错误与受损样本 · 逐例证据</h1><p>285 张对照图均已实际查看；20 例另有完整帧复核。这里的观察与解释仅用于诊断，绝不进入训练标签、特征、伪标签或逐例预测。</p><input id="q" placeholder="搜索编号、动作、用户、问题，例如 E096、wash、取景"><span id="count"> 285 / 285</span></header><main>'''+''.join(cards)+'''</main><script>const cards=[...document.querySelectorAll('article')];document.querySelector('#q').addEventListener('input',e=>{let q=e.target.value.trim().toLowerCase(),n=0;for(let c of cards){c.hidden=!c.dataset.search.toLowerCase().includes(q);if(!c.hidden)n++}document.querySelector('#count').textContent=` ${n} / 285`})</script></html>'''
(A/'CASE_REVIEW.html').write_text(page,encoding='utf-8')
with (A/'CASE_REVIEW.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
receipt=dict(status='285_UNIQUE_REVIEWED_AND_IMAGE_HASHES_VERIFIED',at_unix=time.time(),reviewed=285,raw_observation_records=len(allnotes),correction_records=len(allnotes)-285,full_frame_cases=20,full_frame_sheets=sum(len(x['sheets']) for x in full.values()),source_hashes={n:sha(A/n) for n in ['OBSERVATIONS.jsonl','FULL_FRAME_OBSERVATIONS.jsonl']},outputs={n:sha(A/n) for n in ['CASE_REVIEW.html','CASE_REVIEW.csv']},tags_descriptive_overlapping=dict(collections.Counter(t for n in notes.values() for t in n.get('tags',[]))),training_use=False)
(A/'REVIEW_COVERAGE_VERIFIED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='ERROR_AUDIT_COVERAGE_VERIFIED',at_unix=time.time(),artifact='runs/error_resolution_20260913/audit/REVIEW_COVERAGE_VERIFIED.json',reviewed=285,full_frame_cases=20,no_training_use=True))+'\n')
print(json.dumps({k:v for k,v in receipt.items() if not k.startswith('tags')}))
