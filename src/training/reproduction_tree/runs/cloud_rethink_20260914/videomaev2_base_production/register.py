from pathlib import Path
import argparse,copy,hashlib,json,time,ast
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1];W=C/'videomaev2_base';F=C/'videomaev2_base_fusion';Q=C/'videomaev2_base_quant';CAL=C/'videomaev2_base_calibration'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main(selected):
 assert not(E/'protocol.json').exists() and not(CAL/'protocol.json').exists()
 result=read(F/'output/RESULT.json');terminal=read(F/'output/DONE.json');assert terminal['status']=='COMPLETE' and result['status']=='COMPLETE_V2_BASE_TI_INT8_SOURCE_ONLY_FUSION' and result['no_Test']
 for n,h in terminal['artifacts'].items():assert sha(F/'output'/n)==h,n
 assert read(C/'v2_base_release_source_qa/DONE.json')['status']=='PASS'
 s=read(C/'videomae_base_production/protocol.json');w=read(W/'protocol.json');now=time.time()
 s.update(id='VIDEOMAEV2_BASE_FULL18_RAW16',registered_at_unix=now,deadline_unix=now+1920,wall_cap_seconds=1800,hypothesis='Actual fixed50 INT8 and source-isolated three-branch development results reviewed before this one full18 raw16 fit. Use exact previously matched V2Base recipe; replacement candidate still requires actual full18 serialization, source-only calibration, independent deployment checks. No Public parameter fit.',control_protocol_sha256=sha(W/'protocol.json'))
 for k in ['epochs','evaluation_points','seed','trainable_parameters','backbone_lr','head_lr','layer_decay','warmup_epochs','min_lr_ratio','weight_decay','label_smoothing','effective_batch','gradient_clip','amp_dtype','amp_initial_scale','TF32','dropout','drop_path','checkpoint_cap_bytes','pretrained_sha256','admission_sha256','qa_protocol_sha256','qa_result_sha256','original_protocol_sha256','pose_inputs_sha256']:
  if k in w:s[k]=w[k]
 s['jobs'][0]['id']='V2_BASE_FULL18_RAW16'
 h=copy.deepcopy(w['hosts']['cloud']);h.update(output_directory='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914/videomaev2_base_production/output/cloud',wall_cap_seconds=1800,workers=12,micro_clips=8,eval_micro=4)
 h['predecessor']=dict(directory='/home/featurize/cuhkx_motion_transfer_20260911_v1/cloud_rethink_20260914/videomaev2_base_quant/output/cloud',protocol_sha256=sha(Q/'protocol.json'),success_only=True)
 s['hosts']={'cloud':h}
 s['controls']=dict(current_whole_correct=900,current_whole_rows=1094,selected=selected,source_validation_result_sha256=sha(F/'output/RESULT.json'),selected_paired=result['pooled'][selected],limitations=result['limitations'])
 s['limits']=dict(no_Test=True,no_Public_tuning=True,no_new_instance=True,final_total_bytes=100000000,provisional_projected_weight_bytes_before_calibrator=99427234)
 s['validation_evidence']={str(p.relative_to(C)).replace('\\','/'):sha(p) for p in [F/'output/RESULT.json',F/'output/DONE.json',C/'v2_base_release_source_qa/RESULT.json']}
 s['code_sha256']={k:v for k,v in w['code_sha256'].items() if not k.startswith('videomaev2_base/') or k in ['videomaev2_base/worker.py','videomaev2_base/model.py']}
 s['code_sha256']['videomaev2_base_quant/codec.py']=sha(Q/'codec.py')
 for n in ['worker.py','controller.py','launch.py','watch.py']:
  ast.parse((E/n).read_text());s['code_sha256']['videomaev2_base_production/'+n]=sha(E/n)
 s['worker_sha256']=sha(E/'worker.py');s['controller_sha256']=sha(E/'controller.py')
 for n,hv in s['code_sha256'].items():assert sha(C/n)==hv,n
 for n,hv in s['protected_local_bindings'].items():assert sha(n)==hv
 write(E/'protocol.json',s)
 files=[dict(name=n,bytes=(E/n).stat().st_size,sha256=sha(E/n)) for n in ['worker.py','controller.py','launch.py','watch.py','protocol.json']]
 write(E/'TRANSFER_MANIFEST.json',dict(remote_directory=str(Path(h['output_directory']).parent.parent).replace('\\','/'),files=files,total_bytes=sum(v['bytes'] for v in files),no_raw_data=True,no_credentials=True))
 access=C/'post_aux_cloud_access.py';text=access.read_text();needle="'videomaev2_base_quant/protocol.json'";assert needle in text
 if "'videomaev2_base_production/protocol.json'" not in text:text=text.replace(needle,needle+",'videomaev2_base_production/protocol.json'");access.write_text(text,encoding='utf-8')
 # Single source-only production calibration, selected mechanism already reviewed.
 source=F/'output/SOURCE_FITTED.npz';base=C.parent/'noon_push_20260914/production_meta_input/source_oof_v2.npz'
 bound=[CAL/'worker.py',C/'frozen_fusion_math.py',F/'classwise3.py',source,base,F/'output/DONE.json']
 cs=dict(status='FROZEN_FOR_EXECUTION',registered_at_unix=now,deadline_unix=now+240,cpu_fit_seconds=180,source_npz=str(source.relative_to(R)),base_source_npz=str(base.relative_to(R)),bindings={str(p.relative_to(R)):sha(p) for p in bound},hypothesis='Fit selected three-branch calibrated classifier to all already source-isolated predictions, using same fixed500 global and optional fixed0.1 classwise mechanism. No new candidate scan, outer labels never used in fitting.',selected=selected,fit_count=1 if selected=='global3' else 2,optimizer='FP64_LBFGS500_userbalanced',no_Test=True)
 write(CAL/'protocol.json',cs)
 decision=dict(event='V2_BASE_FULL18_AND_FINAL_SOURCE_CALIBRATION_REGISTERED',at_unix=now,selected=selected,protocol_sha256=sha(E/'protocol.json'),calibration_protocol_sha256=sha(CAL/'protocol.json'),validation_result_sha256=sha(F/'output/RESULT.json'),gpu_cap_seconds=1800,cpu_cap_seconds=180,registered_before_updates=True)
 write(E/'REGISTRATION.json',decision)
 with(R/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(decision)+'\n')
 print(json.dumps(decision))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--selected',choices=['global3','classwise3'],required=True);main(**vars(p.parse_args()))
