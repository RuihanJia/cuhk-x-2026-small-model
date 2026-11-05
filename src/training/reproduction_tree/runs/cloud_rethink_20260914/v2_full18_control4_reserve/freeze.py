from pathlib import Path
import argparse,json,hashlib,time,os
E=Path(__file__).resolve().parent;C=E.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(host):
 candidate=E/f'protocol.{host}.candidate.json';dest=E/f'protocol.{host}.json';s=json.loads(candidate.read_text());assert s['status']=='CPU_PREPARED_AWAITING_ROOT_LAUNCH_DECISION'and not dest.exists();assert (os.name=='nt')==(host=='local')
 h=s['hosts'][host]
 for path,value in h['input_bindings'].items():assert sha(path)==value,path
 for rel,value in s['code_sha256'].items():assert sha(C/rel)==value,rel
 pred=h['predecessor'];assert 'PENDING'not in json.dumps(pred);assert sha(pred['protocol_file'])==pred['protocol_sha256']
 now=time.time();s.update(status='FROZEN_FOR_EXECUTION',registered_at_unix=now,candidate_sha256=sha(candidate),queue_deadline_unix=now+s['queue_cap_seconds'],deadline_unix=now+s['queue_cap_seconds']+s['execution_cap_seconds'])
 dest.write_text(json.dumps(s,indent=2),encoding='utf-8');print(json.dumps(dict(status='FROZEN_NO_LAUNCH',protocol_sha256=sha(dest),protocol_file=str(dest),host=host,execution_cap_seconds=s['execution_cap_seconds'])))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--host',required=True,choices=['local','cloud']);main(**vars(a.parse_args()))
