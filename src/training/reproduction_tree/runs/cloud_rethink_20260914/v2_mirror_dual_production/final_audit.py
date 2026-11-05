from pathlib import Path
import os,csv,json,hashlib,time,zipfile,shutil,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1];D=Path(json.loads((E/'protocol.json').read_text(encoding='utf-8'))['package'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 start=time.time();s=read(E/'inference_protocol.json');ph=sha(E/'inference_protocol.json');out=E/'output/local';cd=read(out/'CONTROLLER_DONE.json');assert cd['status']=='COMPLETE' and cd['protocol_sha256']==ph and cd['terminal_sha256']==sha(out/'DONE.json')
 for n,h in read(out/'DONE.json')['artifacts'].items():assert sha(out/n)==h
 result_forward=read(out/'RESULT.json');assert sha(D/'submission.csv')==result_forward['csv_sha256'].lower() and sha(D/'submission.csv.logits.npz')==result_forward['logits_npz_sha256']
 manifest=read(D/'inference_manifest.json')
 for p,v in manifest['files'].items():assert sha(D/p)==v['sha256'] and (D/p).stat().st_size==v['bytes'],p
 config=read(D/'release.json');cal=read(D/'calibrator.json');size=sum((D/v['path']).stat().st_size for v in config['checkpoints'].values())+(D/'calibrator.json').stat().st_size;assert size==config['serialized_weight_bytes']<=100000000
 with np.load(D/'submission.csv.logits.npz',allow_pickle=False) as ar:v={k:ar[k] for k in ar.files}
 assert v['branch_logits'].shape==(405,4,40) and v['available'].shape==(405,4) and v['fused'].shape==(405,40) and v['fused'].dtype==np.float32 and v['available'].any(1).all()
 from audit_math import replay
 check=replay(v['branch_logits'],v['available'],cal)
 assert np.isfinite(check).all() and np.max(np.abs(check-v['fused']))<1e-6 and np.array_equal(check.argmax(1),v['fused'].argmax(1))
 with(D/'submission.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.reader(f))
 assert rows[0]==['path','prediction'] and len(rows)==406 and all(len(x)==2 for x in rows)
 paths=[x[0] for x in rows[1:]];pred=np.array([int(x[1]) for x in rows[1:]])
 assert len(set(paths))==405 and np.array_equal(np.asarray(paths),v['output_paths']) and np.array_equal(pred,check.argmax(1)) and ((0<=pred)&(pred<40)).all()
 sys.path.insert(0,str(D));from template_contract import verify_paths;verify_paths(paths)
 with(R/'deliverables/submission_videomaev2_base_classwise_20260915.csv').open(newline='',encoding='utf-8-sig') as f:prior=list(csv.reader(f))
 assert [x[0] for x in prior[1:]]==paths
 result=dict(status='PASS_V2_BASE6_SMALL8_FINAL405',protocol_sha256=ph,csv_sha256=sha(D/'submission.csv'),rows=405,unique_paths=405,classes=40,exact_sample_order=True,all_frozen_files_unchanged=True,all_frozen_files=len(manifest['files']),all_serialized_weights_and_calibrator_bytes=size,headroom_bytes=100000000-size,independent_numpy_max_abs=float(np.max(np.abs(check-v['fused']))),argmax_mismatches=0,Public_score_unknown=True,Test_labels_read=False,uploaded=False,model_sha256=sha(D/'checkpoints/PAIR.bin'),calibrator_sha256=sha(D/'calibrator.json'),prediction_changes_vs_current_best=int((pred!=np.array([int(x[1]) for x in prior[1:]])).sum()),at_unix=time.time())
 (D/'FINAL_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
 destination=Path(read(E/'protocol.json')['csv']);assert not destination.exists();shutil.copy2(D/'submission.csv',destination);assert sha(destination)==result['csv_sha256']
 archive=Path(read(E/'protocol.json')['zip']);assert not archive.exists()
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as zf:
  for p in sorted(D.rglob('*')):
   rel=p.relative_to(D)
   if p.is_file() and '__pycache__' not in rel.parts and 'runtime' not in rel.parts and p.suffix!='.pyc':zf.write(p,rel.as_posix())
 with zipfile.ZipFile(archive) as zf:assert zf.testzip() is None
 result.update(csv=str(destination),package=str(D),zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=sha(archive),seconds=time.time()-start)
 (E/'FINAL_DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n')

 print(json.dumps(result))
if __name__=='__main__':main()
