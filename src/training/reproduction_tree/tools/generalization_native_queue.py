"""Bounded, restartable job queue using the project's established thermal/IO helpers."""
from pathlib import Path
import argparse, json, os, subprocess, sys, time, traceback

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);args=a.parse_args()
    root=Path(args.root).resolve();sys.path.insert(0,str(root/'bundle'))
    from dual_host_runtime.io import atomic,read
    from dual_host_runtime.health import NVML,Pause,identity,stop_owned
    from research_runtime.common import sha
    config=read(root/'QUEUE.json');deadline=config['deadline_unix'];active={};finished={}
    if (root/'QUEUE_STATUS.json').exists():finished=read(root/'QUEUE_STATUS.json').get('finished',{})
    started=time.time();nv=NVML();local=os.name=='nt';lastwrite=0
    atomic(root/'CONTROLLER.json',dict(**identity(os.getpid()),started_unix=started,deadline_unix=deadline))
    try:
        while time.time()<deadline:
            config=read(root/'QUEUE.json');sample=nv.sample();sample['control']='RUN'
            if not local and sample['temperature_c']>=88:sample['control']='CLOUD_TEMPERATURE_STOP'
            atomic(root/'TELEMETRY.json',sample)
            if sample['control']=='CLOUD_TEMPERATURE_STOP':raise RuntimeError(sample['control'])
            if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
            for key,v in list(active.items()):
                rc=v['process'].poll()
                if rc is None:
                    v['control']=v['pause'].tick(sample,local);continue
                v['pause'].close();v['log'].close();out=root/'output'/key
                finished[key]=dict(exit_code=rc,finished_unix=time.time(),status='COMPLETE' if rc==0 and (out/'DONE.json').exists() else 'FAILED',owner=v['owner'])
                del active[key]
            waiting=[]
            for job in config['jobs']:
                key=job['id']
                if key in active or key in finished:continue
                if not all((root/v).is_file() for v in job.get('requires',[])):
                    waiting.append(key);continue
                if len(active)>=config['concurrency']:break
                protocol=root/job['protocol']
                if sha(protocol)!=job['protocol_sha256']:raise ValueError('PROTOCOL_HASH_CHANGED:'+key)
                p=read(protocol)
                for name,expected in p['code_sha256'].items():
                    if sha(root/'bundle'/name)!=expected:raise ValueError('CODE_HASH_CHANGED:'+name)
                out=root/'output'/key;out.mkdir(parents=True,exist_ok=True)
                log=(out/'worker.log').open('a',encoding='utf-8')
                command=[sys.executable,'-X','utf8','-B',str(root/'bundle'/job['worker']),'--root',str(root),'--job',key,'--protocol',str(protocol),'--deadline',str(deadline)]
                proc=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
                owner=identity(proc.pid);atomic(out/'LAUNCH.json',dict(**owner,started_unix=time.time(),command=command,protocol_sha256=job['protocol_sha256']))
                active[key]=dict(process=proc,owner=owner,pause=Pause(owner),log=log,control='RUN')
            now=time.time()
            if now-lastwrite>=10 or not active:
                atomic(root/'QUEUE_STATUS.json',dict(status='RUNNING' if active else 'WAITING_INPUT' if waiting else 'COMPLETE',at_unix=now,started_unix=started,deadline_unix=deadline,active={k:dict(owner=v['owner'],control=v['control']) for k,v in active.items()},finished=finished,waiting_inputs=waiting,remaining=[j['id'] for j in config['jobs'] if j['id'] not in finished and j['id'] not in active],telemetry=sample))
                with (root/'telemetry.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(sample)+'\n')
                lastwrite=now
            if not active and not waiting:
                atomic(root/'QUEUE_DONE.json',dict(status='COMPLETE' if all(v['status']=='COMPLETE' for v in finished.values()) else 'COMPLETED_WITH_FAILURES',finished=finished,at_unix=now));return
            time.sleep(2)
        raise TimeoutError('QUEUE_BUDGET_EXHAUSTED')
    except BaseException as e:
        atomic(root/'QUEUE_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),at_unix=time.time()))
        raise
    finally:
        for v in active.values():
            v['pause'].close();stop_owned(v['owner']);v['log'].close()

if __name__=='__main__':main()
