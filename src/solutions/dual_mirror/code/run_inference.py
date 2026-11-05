"""Portable entry for the exact selected multi-branch checkpoint container."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--micro',type=int,default=4);a=p.parse_args()
 data=Path(a.data_dir).resolve();output=Path(a.output).resolve();artifact=ROOT.parents[2]/'artifacts'/ROOT.name;cp=artifact/'model.pth';meta=json.loads((artifact/'CONTAINER.json').read_text())
 assert cp.stat().st_size==meta['bytes']<100000000
 assert hashlib.sha256(cp.read_bytes()).hexdigest()==meta['sha256']
 release=json.loads((ROOT/'code/runtime/release.json').read_text())
 required={v['path']:v['sha256'].lower() for v in release['checkpoints'].values()};required['calibrator.json']=release['calibrator_sha256'].lower()
 with tempfile.TemporaryDirectory(prefix='cuhkx_verified_') as tmp:
  runtime=Path(tmp)/'runtime';shutil.copytree(ROOT/'code/runtime',runtime,ignore=shutil.ignore_patterns('checkpoints','calibrator.json','runtime','__pycache__','*.pyc','submission.csv*','geometry_cache.json'))
  with zipfile.ZipFile(cp) as z:
   assert set(z.namelist())==set(required)
   for name,h in required.items():
    payload=z.read(name);assert hashlib.sha256(payload).hexdigest()==h
    path=runtime/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
  command=[sys.executable,'-B',str(runtime/'infer.py'),'--data-dir',str(data),'--output',str(output),'--workers',str(a.workers),'--micro',str(a.micro)]
  subprocess.run(command,check=True)
if __name__=='__main__':main()
