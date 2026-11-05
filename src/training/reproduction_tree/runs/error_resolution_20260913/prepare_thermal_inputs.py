from pathlib import Path
import sys,json,time,hashlib
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'bundle'))
from activity_view_data import compute_stream_box
DATA='D:/Datasets/CUHK-X_Small_Model_Track/Small-Model-Track'
def run(row):
 s=row['views'][3]
 if not s['files']:return None
 stream=dict(directory=s['directory'],files=s['files']);roi=compute_stream_box(DATA,stream)
 return dict(clip_id=row['clip_id'],label=row['label'],user=row['user'],streams=[stream]),roi
def main():
 p=R/'THERMAL_INPUTS.json';assert not p.exists();rows=json.loads((R/'DETAIL_INPUTS.json').read_text(encoding='utf-8'));start=time.time();selected=[];regions={}
 with ThreadPoolExecutor(max_workers=4) as pool:
  for item in pool.map(run,rows):
   if item:r,roi=item;selected.append(r);regions[r['clip_id']]=[roi]
 p.write_text(json.dumps(dict(rows=selected,regions=regions),ensure_ascii=False)+'\n',encoding='utf-8')
 result=dict(at_unix=time.time(),seconds=time.time()-start,source_rows=len(rows),thermal_available=len(selected),missing_thermal=len(rows)-len(selected),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),roi_method='existing label-independent ACTIVITY_MAXMASS_SQUARE060_SHORTSIDE_V1',Test_read=False)
 (R/'THERMAL_INPUTS_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
