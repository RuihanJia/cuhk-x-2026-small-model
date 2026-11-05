from pathlib import Path
import shutil
E=Path(__file__).resolve().parent;C=E.parent
source=(C/'v2_window_production/worker.py').read_text()
source=source.replace('Prepared full18 four-epoch continuation; activation requires reviewed paired evidence and a new frozen protocol.','Authorized fixed full18 CONTROL4 reserve and one 50/50 weight average; not selected.')
source=source.replace("assert WARM_ARM in ('CONTROL','WINDOW')","assert WARM_ARM=='CONTROL' and s['average_alpha']==.5")
source=source.replace("runtime=dict(s);runtime.update(h);k.fit(job,runtime,ph,out,inp,guard)","assert not(out/'resume.pt').exists(),'NEW_STAGE_MUST_START_FRESH_ADAMW'\n runtime=dict(s);runtime.update(h);k.fit(job,runtime,ph,out,inp,guard)")
pos=source.index(" names=['PARENT_BINDING.json'")
end=source.index("if __name__=='__main__':",pos)
new=''' # Fixed average, retaining both alternative endpoints outside any submitted package.
 raw=torch.load(WARM_PARENT,map_location='cpu',weights_only=True);averaged={};changed=0
 for name,value in raw['model'].items():
  other=cp['model'][name];assert value.dtype==other.dtype==torch.float16 and value.shape==other.shape
  averaged[name]=((value.float()+other.float())*.5).half();changed+=int(not torch.equal(value,other))
 assert changed>0
 avg=dict(model=averaged,protocol_sha256=ph,pretrained_sha256=s['pretrained_sha256'],train_users=job['train_users'],held_users=[],epochs=4,seed=s['seed'],alpha=.5,raw_sha256=s['parent_checkpoint_sha256'],control_sha256=fr['checkpoint_sha256'],arithmetic='FP32 midpoint then FP16')
 save_torch(out/'average_model_fp16.pt',avg);packed=pack(avg);save_torch(out/'average_model_int8.pt',packed)
 reloaded=torch.load(out/'average_model_int8.pt',map_location='cpu',weights_only=True);expected=unpack(packed);replayed=unpack(reloaded)
 assert set(expected)==set(replayed) and all(torch.equal(expected[n],replayed[n]) for n in expected)
 assert len(reloaded['quantized_names'])==50 and (out/'average_model_int8.pt').stat().st_size<87000000
 # All finite source-interface checks use known training examples and a single seven-view readout.
 qa_row=next(r for r in inp['rows'] if r['user']==1);source=wd.SevenViews([qa_row],h['data_root'],inp['regions'])[0]
 model=BASE_MODEL(classes=40,fast=True,seed=s['seed']).cuda();model.load_state_dict(replayed,strict=True);model.eval()
 with torch.inference_mode():
  logits=torch.cat([model(source['x'][i:i+4].cuda()).cpu()for i in range(0,7,4)])
 assert logits.shape==(7,40) and logits.dtype==torch.float32 and torch.isfinite(logits).all()
 qa=dict(protocol_sha256=ph,source_clip_id=qa_row['clip_id'],source_user=1,views=7,FP32_logits_finite=True,logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),no_Test=True)
 atomic(out/'SOURCE_INTERFACE_QA.json',qa)
 atomic(out/'AVERAGE_FROZEN.json',dict(protocol_sha256=ph,alpha=.5,arithmetic='FP32 midpoint of saved raw16/control4 FP16 then FP16 then originalALL50INT8',checkpoint_sha256=sha(out/'average_model_int8.pt'),checkpoint_bytes=(out/'average_model_int8.pt').stat().st_size,raw_checkpoint_sha256=s['parent_checkpoint_sha256'],control_checkpoint_sha256=fr['checkpoint_sha256'],average_fp16_sha256=sha(out/'average_model_fp16.pt'),tensors=len(expected),changed_tensors=changed,quantized_matrices=50,actual_FP32_storage_replay_exact=True,model_plus_aux_bytes=(out/'average_model_int8.pt').stat().st_size+s['retained_aux_bytes'],calibration_reserved_bytes=s['calibration_reserved_bytes'],no_selection_claim=True,at_unix=time.time()))
 assert (out/'average_model_int8.pt').stat().st_size+s['retained_aux_bytes']+s['calibration_reserved_bytes']<=100000000
 names=['PARENT_BINDING.json','TRAIN_DONE.json','model_fp16.pt','model_int8.pt','QUANTIZED.json','INITIAL.json','SPLIT.json','HISTORY.json','FIRST_UPDATE.json','FROZEN.json','average_model_fp16.pt','average_model_int8.pt','AVERAGE_FROZEN.json','SOURCE_INTERFACE_QA.json'];guard.check(force=True,phase='FULL18_CONTROL_AND_AVERAGE_RESERVE_COMPLETE')
 atomic(out/'DONE.json',dict(status='COMPLETE_FULL18_CONTROL4_AND_FIXED_AVERAGE_RESERVE',protocol_sha256=ph,job=job['id'],artifacts={n:sha(out/n) for n in names},no_Test=True,no_selection_claim=True,held_evaluation_performed=False,at_unix=time.time()))
'''
source=source[:pos]+new+source[end:]
(E/'worker.py').write_text(source)
for n in ('freeze.py','deploy.py','resilient_io.py'):shutil.copyfile(C/'v2_true_fullframe_readout'/n,E/n)
ctl=(C/'v2_true_fullframe_readout/controller.py').read_text().replace('COMPLETE_TRUE_FULLFRAME_READOUT','COMPLETE_FULL18_CONTROL4_AND_FIXED_AVERAGE_RESERVE').replace('FULLFRAME_QUEUE_CAP','FULL18_CONTROL_QUEUE_CAP')
(E/'controller.py').write_text(ctl)
print('Prepared fixed runtime copies')
