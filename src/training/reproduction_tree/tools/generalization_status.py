"""One bounded, compact two-host snapshot for the20-minute assistant."""
from pathlib import Path
from datetime import datetime
import json,time
import psutil
import generalization_cloud as cloud
ROOT=cloud.ROOT;BASE=cloud.BASE
def read(p):return cloud.read(p)
def ident_alive(owner):
    if not owner or 'pid' not in owner or 'create_time' not in owner:return False
    try:p=psutil.Process(owner['pid']);return p.is_running() and p.status()!=psutil.STATUS_ZOMBIE and abs(p.create_time()-owner['create_time'])<.01
    except psutil.NoSuchProcess:return False
def joint_delivery():
    run=ROOT/'runs/r3_imu_fusion_release_20260911_v1';stage=run/'submission_stage'
    if not (run/'DELIVERY_LAUNCH.json').exists():return None
    value=dict(controller_alive=ident_alive(read(run/'DELIVERY_LAUNCH.json')))
    if (run/'DELIVERY_STATUS.json').exists():value.update(read(run/'DELIVERY_STATUS.json'))
    if (run/'DELIVERY_ERROR.json').exists():value['error']=read(run/'DELIVERY_ERROR.json')['error']
    if (run/'DELIVERY_DONE.json').exists():value['validated']=True
    if (stage/'STATUS.json').exists():
        s=read(stage/'STATUS.json');repeat=s['repeat'];value['repeat']=repeat
        value['inference_worker_alive']=ident_alive(s.get('worker'))
        path=stage/f'submission_{repeat}.csv.status.json'
        if path.exists():
            p=read(path);value['progress']={k:p[k] for k in ['phase','records','total']}
            value['progress_age_seconds']=round(time.time()-p['at_unix'],1)
        if (stage/'TELEMETRY.json').exists():
            sample=read(stage/'TELEMETRY.json');value['gpu']={k:sample[k] for k in ['temperature_c','gpu_utilization_percent','memory_used_bytes','at_unix']}
            value['gpu']['sample_age_seconds']=round(time.time()-sample['at_unix'],1)
    return value
def compact(snapshot,queue):
    deadline=queue.get('deadline_unix') or 0
    observed=snapshot.get('at_unix',time.time())
    qs=snapshot.get('QUEUE_STATUS.json',{});r=dict(status=qs.get('status'),controller_alive=snapshot.get('controller_alive'),gpu=snapshot.get('TELEMETRY.json'),active={},completed=sum(v['status']=='COMPLETE' for v in qs.get('finished',{}).values()),failed={k:v for k,v in qs.get('finished',{}).items() if v['status']!='COMPLETE'},pending_count=len(qs.get('remaining',[])),next_jobs=qs.get('remaining',[])[:2],wall_time_cap_enabled=deadline>0,budget_remaining_minutes=round((deadline-time.time())/60,1) if deadline>0 else None)
    if r['gpu']:r['gpu']={k:r['gpu'][k] for k in ['temperature_c','gpu_utilization_percent','memory_used_bytes','at_unix']};r['gpu']['sample_age_seconds']=round(max(0,observed-r['gpu']['at_unix']),1)
    for k in qs.get('active',{}):
        v=snapshot.get('workers',{}).get(k,{});s=v.get('STATUS.json',{});r['active'][k]={key:s[key] for key in ['phase','epoch','epochs','records','total','updates','amp_skips','at_unix'] if key in s};r['active'][k]['progress_age_seconds']=round(max(0,observed-s.get('at_unix',observed)),1)
        metrics=v.get('metrics.json',[])
        if metrics:
            seconds=sum(row['seconds'] for row in metrics)/len(metrics);r['active'][k]['recent_epoch_seconds']=round(seconds,1)
            if s.get('epoch') and s.get('epochs') and s.get('total'):r['active'][k]['approx_remaining_fit_minutes']=round((s['epochs']-s['epoch']+1-s.get('records',0)/s['total'])*seconds/60,1)
        if v.get('ERROR.json'):r['active'][k]['error']=v['ERROR.json'].get('error')
    if snapshot.get('QUEUE_ERROR.json'):r['error']=snapshot['QUEUE_ERROR.json'].get('error')
    return r
def main():
    state=read(ROOT/'PROJECT_STATE.json');local=BASE/'local';snapshot={}
    for n in ['QUEUE_STATUS.json','QUEUE_DONE.json','QUEUE_ERROR.json','TELEMETRY.json']:
        if (local/n).exists():snapshot[n]=read(local/n)
    snapshot['controller_alive']=ident_alive(read(local/'CONTROLLER.json'));snapshot['workers']={}
    for k in snapshot.get('QUEUE_STATUS.json',{}).get('active',{}):
        v={}
        for n in ['STATUS.json','ERROR.json','metrics.json']:
            if (local/'output'/k/n).exists():
                value=read(local/'output'/k/n);v[n]=value[-3:] if n=='metrics.json' else value
        snapshot['workers'][k]=v
    value=dict(at_local=datetime.now().astimezone().isoformat(timespec='seconds'),phase=state['phase'],automation=state['runtime']['automation']['status'],local=compact(snapshot,read(local/'QUEUE.json')))
    delivery=joint_delivery()
    if delivery:
        value['joint_delivery']=delivery
        if delivery.get('inference_worker_alive'):
            value['local']['status']='JOINT_CSV_INFERENCE_ACTIVE_NATIVE_TRAINING_RECOVERY_AFTER_DELIVERY'
            value['local']['active']={};value['local']['gpu']=delivery.get('gpu')
        elif not value['local']['controller_alive'] and value['local']['active']:
            value['local']['status']='INTERRUPTED_NATIVE_TRAINING_RECOVERY_AFTER_JOINT_DELIVERY'
    try:
        m,c=cloud.access.connection()
        try:r=cloud.snapshot(m,c)
        finally:c.close()
        value['cloud']=compact(r,read(BASE/'cloud/QUEUE.json'))
    except Exception as e:value['cloud']=dict(status='SNAPSHOT_CONNECTION_FAILED',error=str(e)[:300])
    transfer=BASE/'THERMAL_TRANSFER.json'
    if transfer.exists():
        t=read(transfer);value['thermal_handoff_data']={k:t[k] for k in ['status','sent_bytes','total_bytes','at_unix','error'] if k in t}
    value['next_preparation']=state['successor_preparation']['next'];cloud.write(BASE/'COMPACT_STATUS.json',value);print(json.dumps(value,ensure_ascii=False))
if __name__=='__main__':main()
