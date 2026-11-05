import json, sys, time, hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
out = root / 'audit' / 'OBSERVATIONS.jsonl'
existing = {json.loads(s)['case_id'] for s in out.read_text(encoding='utf-8').splitlines()} if out.exists() else set()
rows = json.load(sys.stdin)
with out.open('a', encoding='utf-8') as f:
    for case, visible, uncertain, proposal, tags in rows:
        assert case not in existing, case
        meta = json.loads((root/'audit'/'images'/f'{case}.json').read_text(encoding='utf-8'))
        p = root/'audit'/'images'/f'{case}.jpg'
        rec = dict(case_id=case, clip_id=meta['clip_id'], viewed_image_sha256=hashlib.sha256(p.read_bytes()).hexdigest().upper(), viewed=True, at_unix=time.time(), visible_evidence=visible, uncertain_interpretation=uncertain, proposed_program_or_experiment=proposal, tags=tags, review_scope='Eight native IR/Depth frames, exact center IR inputs, four Thermal samples where available, numeric sensor coverage; not full video', used_as_training_label_or_feature=False)
        f.write(json.dumps(rec, ensure_ascii=False)+'\n')
        existing.add(case)
print(json.dumps({'reviewed': len(existing), 'appended': [r[0] for r in rows]}))
