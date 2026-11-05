"""Single CPU callback: source calibration then independent NumPy replay."""
from pathlib import Path
import json,hashlib,os,subprocess,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
E=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=E/'protocol.json';s=json.loads(p.read_text('utf-8-sig'));ph=sha(p);assert s['status']=='FROZEN_FOR_EXECUTION'and time.time()<s['deadline_unix']
 for n,h in s['code_bindings'].items():assert sha(n)==h,n
 out=E/'output';out.mkdir(exist_ok=True)
 for name in ['calibrate.py','independent_audit.py']:
  remain=s['deadline_unix']-time.time();assert remain>0
  with (out/(name+'.log')).open('ab')as f:subprocess.run([sys.executable,'-X','utf8','-B',str(E/name),'--protocol',str(p),'--expected-protocol-sha',ph],check=True,timeout=min(remain,300),stdout=f,stderr=subprocess.STDOUT)
 a=json.loads((out/'INDEPENDENT_AUDIT.json').read_text());assert a['status']=='PASS_INDEPENDENT_MIRROR_DUAL_SOURCE_AND_OUTER_REPLAY'and a['protocol_sha256']==ph
 q=E/'CALLBACK_DONE.json';assert not q.exists();q.write_text(json.dumps(dict(status='COMPLETE_MIRROR_DUAL_SOURCE_FIT_AND_INDEPENDENT_AUDIT',protocol_sha256=ph,result_sha256=sha(out/'RESULT.json'),audit_sha256=sha(out/'INDEPENDENT_AUDIT.json'),at_unix=time.time()),indent=2)+'\n','utf8')
if __name__=='__main__':main()
