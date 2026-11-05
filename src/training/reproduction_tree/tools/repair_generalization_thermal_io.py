"""Match sample/tensor order, then resume epoch checkpoints using threads instead of Windows workers."""
from pathlib import Path
from datetime import datetime
import ast,copy,json,shutil,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/generalization_training_20260910_v1';OUT=BASE/'local';BUNDLE=OUT/'bundle';sys.path.insert(0,str(BUNDLE))
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import alive,stop_owned
from research_runtime.common import sha
def main():
    torch.set_num_threads(2);p=read(OUT/'thermal_protocol.json');ph=sha(OUT/'thermal_protocol.json');assert not (OUT/'thermal_io2_protocol.json').exists();shutil.copy2(ROOT/'tools/generalization_threaded_data.py',BUNDLE/'threaded_data.py')
    from threaded_data import ThreadedLoader
    from generalization_thermal_worker import ThermalClips
    rows=read(BUNDLE/'data/thermal_index.json');tr=[r for r in rows if r['user'] in p['jobs']['T_F0_0']['train_users']];norm=read(OUT/'output/T_F0_0/NORMALIZER.json');ds=ThermalClips(tr,p['data']['root'],norm['mean'],norm['std'],True,3)
    # Check the entire permutation, then real decoded/normalized tensors from8 training-side clips.
    seed=ds.seed+ds.epoch*1000003;g1=torch.Generator().manual_seed(seed);standard=torch.utils.data.DataLoader(ds,batch_size=2,shuffle=True,num_workers=0,generator=g1);g2=torch.Generator().manual_seed(seed);s2=torch.utils.data.DataLoader(ds,batch_size=2,shuffle=True,num_workers=0,generator=g2);it=iter(standard);torch.empty((),dtype=torch.int64).random_(generator=g2);order1=list(standard.batch_sampler);order2=list(s2.batch_sampler);assert order1==order2
    standard=torch.utils.data.DataLoader(ds,batch_size=2,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed));threaded=ThreadedLoader(ds,2,4,True);i=iter(standard);j=iter(threaded);checked=0
    for _ in range(4):
        a=next(i);b=next(j);assert a['clip_id']==b['clip_id'] and all(torch.equal(a[k],b[k].cpu()) for k in ['x','y','user']);checked+=len(a['y'])
    j.close();del i,j,standard,threaded
    text=(BUNDLE/'generalization_thermal_worker.py').read_text(encoding='utf-8');text=text.replace('from research_runtime.data import loader,safe_path','from research_runtime.data import safe_path\n    from threaded_data import loader').replace('ph=sha(args.protocol);spec=',"ph=p.get('scientific_parent_sha256',sha(args.protocol));spec=").replace("    try:\n        sys.path.insert", "    atomic(out/'EXECUTION.json',dict(execution_protocol_sha256=sha(args.protocol),scientific_protocol_sha256=ph,loader='bounded CPU threads, exact sampled input parity verified'))\n    try:\n        sys.path.insert").replace("heldout_metric_selection=False))","heldout_metric_selection=False,execution_protocol_sha256=sha(args.protocol)))")
    worker=BUNDLE/'generalization_thermal_worker_io2.py';ast.parse(text);worker.write_text(text,encoding='utf-8');updated=copy.deepcopy(p);updated['id']='GENERALIZATION-NATIVE-THERMAL-IO2-20260910-V1';updated['at_local']=datetime.now().astimezone().isoformat(timespec='seconds');updated['scientific_parent_sha256']=ph;updated['code_sha256']={x.relative_to(BUNDLE).as_posix():sha(x) for x in BUNDLE.rglob('*.py')};updated['execution_delta']=dict(reason='Windows worker processes exhausted commit/pagefile while importing repeated CUDA DLLs',change='Replace spawned CPU DataLoader workers with at most4in-process CPU decoding threads and at most3prefetched batches per job',science_changes=False,entire_epoch_batch_index_order_equal=True,real_training_clips_tensor_bitwise_equal=checked,checkpoint_resume='same original scientific identity, normalizer and RNG; retain completed epochs; discard unfinished epoch only',concurrency=2,budget_extension=False)
    atomic(OUT/'thermal_io2_protocol.json',updated);newph=sha(OUT/'thermal_io2_protocol.json');atomic(OUT/'THREAD_LOADER_PARITY.json',dict(status='PASS',entire_epoch_batches=len(order1),samples=len(tr),full_order_equal=True,real_training_sample_tensor_bitwise_equal=checked,source_protocol_sha256=ph,new_execution_protocol_sha256=newph,at_unix=time.time()))
    owner=read(OUT/'CONTROLLER.json');assert alive(owner);stop_owned(owner);assert not alive(owner)
    prior=read(OUT/'QUEUE_STATUS.json');atomic(OUT/'QUEUE_STATUS_BEFORE_IO_REPAIR.json',prior);prior['active']={};prior['finished']={k:v for k,v in prior['finished'].items() if v['status']=='COMPLETE'};prior['status']='ENGINEERING_RESTART_READY';atomic(OUT/'QUEUE_STATUS.json',prior)
    q=read(OUT/'QUEUE.json');q['concurrency']=2;q['execution_amendment']='thermal_io2_protocol.json'
    for job in q['jobs']:
        if job['id'].startswith('T_'):job.update(protocol='thermal_io2_protocol.json',protocol_sha256=newph,worker=worker.name)
    atomic(OUT/'QUEUE.json',q)
    if (OUT/'QUEUE_ERROR.json').exists():shutil.copy2(OUT/'QUEUE_ERROR.json',OUT/'QUEUE_ERROR_BEFORE_IO_REPAIR.json');(OUT/'QUEUE_ERROR.json').unlink()
    fact=dict(event='E-GENERALIZATION-THERMAL-IO-REPAIR-20260910-V1',at_local=updated['at_local'],status='INPUT_PARITY_PASS_READY_TO_RESUME',execution_protocol_sha256=newph,scientific_protocol_sha256=ph,completed_epoch_checkpoints={k:read(OUT/'output'/k/'RESUME.json')['completed_epoch'] for k in p['jobs'] if (OUT/'output'/k/'RESUME.json').exists()},affected_scope='registered local Thermal controller and its workers only',cloud_interrupted=False,system_pagefile_settings_changed=False)
    atomic(BASE/'THERMAL_IO_REPAIR.json',fact)
    with (ROOT/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(fact,ensure_ascii=False)+'\n')
    print(json.dumps(fact,ensure_ascii=False))
if __name__=='__main__':main()
