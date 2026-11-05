"""Production-only wrapper around the exact matched video and measured-sensor recipes."""
from pathlib import Path
import argparse,sys
import numpy as np

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--job',required=True);a.add_argument('--protocol',required=True);a.add_argument('--deadline',type=float,required=True);args=a.parse_args();root=Path(args.root);sys.path.insert(0,str(root/'bundle'))
    from dual_host_runtime.io import read
    from research_runtime.common import configure,sha
    from optional_time_budget import Guard
    import research_runtime.r3_full18_train as train
    configure();p=read(args.protocol);p.update(_sha=sha(args.protocol),_bundle=str(root/'bundle'));spec=p['jobs'][args.job];out=root/'output'/args.job;out.mkdir(parents=True,exist_ok=True);guard=Guard(out,args.deadline,root/'TELEMETRY.json')
    assert not spec['predict_users'] and sorted(spec['train_users'])==sorted(p['training_users'])
    if spec['kind']=='video':
        def split(rows,train_users,predict_users):
            assert not predict_users;result=[r for r in rows if r['user'] in train_users];assert len(result)==2931;return result,[]
        def predictions(model,rows,*args,**kwargs):
            assert not rows;return dict(clip_ids=np.asarray([],dtype='U1'),targets=np.asarray([],dtype=np.int64),users=np.asarray([],dtype=np.int64),logits=np.empty((0,40),np.float32))
        train.split_rows=split;train.video_predictions=predictions;train.train_video(p,spec,p['remote_data'],out,root/'feature_cache',guard)
    elif spec['kind']=='sensor':
        p['calibration']=dict(encoder_fit_users=spec['train_users'],calibration_users=[],assessment_users=[]);train.train_sensor(p,out,guard)
    else:raise ValueError('UNREGISTERED_PRODUCTION_KIND')

if __name__=='__main__':main()
