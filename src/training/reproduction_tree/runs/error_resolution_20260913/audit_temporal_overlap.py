"""Read-only raw Training frame-name overlap audit. No relabeling or Test access."""
from pathlib import Path
import csv,json,time,hashlib,itertools
R=Path(__file__).resolve().parent
DATA=Path('D:/Datasets/CUHK-X_Small_Model_Track/Small-Model-Track/Training/data/HAR/data/IR')
start=time.time();byuser={};clips=[]
for d in sorted(DATA.glob('*/*/*')):
    if not d.is_dir():continue
    fs=sorted(d.glob('*.png'))
    if not fs:continue
    keys={x.name[3:-4] for x in fs};row=dict(path=str(d),activity=d.parent.parent.name,user=d.parent.name,clip=d.name,frames=len(fs),first=min(keys),last=max(keys))
    byuser.setdefault(row['user'],[]).append((row,keys));clips.append(row)
overlaps=[]
for user,seq in byuser.items():
    for (a,ka),(b,kb) in itertools.combinations(seq,2):
        if a['activity']==b['activity']:continue
        shared=ka&kb
        if not shared:continue
        byte_equal=0;checked=[]
        for k in sorted(shared):
            pa=Path(a['path'])/('IR_'+k+'.png');pb=Path(b['path'])/('IR_'+k+'.png')
            ha=hashlib.sha256(pa.read_bytes()).hexdigest();hb=hashlib.sha256(pb.read_bytes()).hexdigest();byte_equal+=ha==hb
            checked.append(dict(frame=k,sha_a=ha,sha_b=hb))
        overlaps.append(dict(a=a,b=b,shared_timestamp_and_index=len(shared),identical_IR_bytes=byte_equal,frames=checked))
result=dict(at_unix=time.time(),seconds=time.time()-start,clip_count=len(clips),frame_count=sum(r['frames'] for r in clips),cross_class_overlaps=overlaps,clips=clips,Test_read=False,training_fits=0,automatic_label_changes=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(R/'audit/ALL_TRAINING_TEMPORAL_OVERLAP.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(clips=len(clips),cross_class_overlap_pairs=len(overlaps),user_counts={u:sum(x['a']['user']==u for x in overlaps) for u in byuser},seconds=result['seconds'])))
