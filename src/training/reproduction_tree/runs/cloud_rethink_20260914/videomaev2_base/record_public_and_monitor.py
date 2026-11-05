from pathlib import Path
import ast,copy,csv,hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime
E=Path(__file__).resolve().parent;C=E.parent;R=C.parents[1]
sys.path.insert(0,str(C.parent/'error_resolution_20260913/bundle'))
from dual_host_runtime.io import atomic,read
from dual_host_runtime.health import identity
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=time.time();stamp=datetime.now().astimezone().isoformat();s=read(E/'protocol.json');ph=sha(E/'protocol.json')
ev=C/'public_v2_global_20260915';ev.mkdir(exist_ok=True);image=ev/'USER_SCORE.png'
src=Path('C:/Users/贾瑞涵/AppData/Local/Temp/codex-clipboard-a289ba54-8c9d-4993-8b9e-536ff2ac89dd.png')
if not image.exists():shutil.copyfile(src,image)
assert sha(image)==sha(src)
csvp=R/'deliverables/submission_videomaev2_global_20260915.csv';assert sha(csvp)=='ab20d15d10f71f00bd25937f72b5587431c72a2983c76b8fbaaecb529d2b4112'
with csvp.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
assert len(rows)==405 and len({r['path'] for r in rows})==405 and all(0<=int(r['prediction'])<40 for r in rows)
old=R/'deliverables/submission_k400_classwise_20260915.csv';assert sha(old)=='1016cb58c7b1021e5f15b569a0cb223aea8eec3a14a43a428702ea0fe3997837'
feedback=dict(event='USER_CONFIRMED_V2_GLOBAL_PUBLIC_083582',at_unix=now,at_local=stamp,evidence=str(image.relative_to(R)),evidence_sha256=sha(image),csv=str(csvp.relative_to(R)),csv_sha256=sha(csvp),new_Public=.83582,previous_best_Public=.79104,delta=.04478,uploaded_by_user=True,platform_file_sha_unavailable=True,Test_labels_known=False,user_request='成绩进步巨大 继续努力 注意今天截止 以及最终取最优的两次成绩',status='USER_SUBMITTED_NEW_BEST_PUBLIC',select_boxes_unchecked_in_user_screenshot=True)
if not(ev/'PUBLIC_FEEDBACK.json').exists():
 atomic(ev/'PUBLIC_FEEDBACK.json',feedback)
 with(R/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(feedback,ensure_ascii=False)+'\n')
regp=E/'REVIEW_REGISTRATION.json'
if not regp.exists():
 ast.parse((E/'review.py').read_text(encoding='utf-8'));ast.parse((E/'activate_review.py').read_text(encoding='utf-8'))
 reg=dict(status='FROZEN_FOR_EXECUTION',registered_at_unix=now,deadline_unix=s['deadline_unix']+150,wall_cap_seconds=180,protocol_sha256=ph,worker_sha256=sha(E/'review.py'),current_v2_paired_sha256=sha(C/'videomaev2_fusion/output/PAIRED.npz'),old_base_paired_sha256=sha(C/'videomae_base_kinetics/review/PAIRED.npz'),source_unmatched_diagnostic='.8 current V2global7 probabilities +.2 new Base; cannot promote directly.',both_new_outer_fits_must_complete=True,no_new_outer_outputs_exist=(not(E/'cloud/DOWNLOAD_DONE.json').exists()),no_parameter_fits=True,no_Test=True)
 assert reg['no_new_outer_outputs_exist'];atomic(regp,reg)
 with(R/'EXPERIMENT_RESULTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='V2_BASE_CPU_AUDIT_REGISTERED',registration=str(regp.relative_to(R)),sha256=sha(regp),at_unix=now))+'\n')
 if not(E/'review_wait/CLAIM').exists():
  with(E/'review_launcher.log').open('ab') as log:p=subprocess.Popen([sys.executable,'-X','utf8','-B','-u',str(E/'activate_review.py')],cwd=C,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  atomic(E/'REVIEW_WATCH_LAUNCH.json',dict(owner=identity(p.pid),at_unix=time.time()))
state=read(R/'PROJECT_STATE.json')
if not(ev/'PRIOR_CURRENT_FIELDS.json').exists():atomic(ev/'PRIOR_CURRENT_FIELDS.json',{k:state.get(k) for k in ['latest_submission','best_confirmed_submission','candidate_delivery','backup_candidate_delivery','pending_user_action','latest_ready_submission','runtime','new_public_followup_20260915']})
ready=copy.deepcopy(state['latest_ready_submission']);ready.update(status='USER_SUBMITTED_CURRENT_BEST_PUBLIC',Public_score=.83582,uploaded=True,uploaded_by_user=True,feedback_evidence=str((ev/'PUBLIC_FEEDBACK.json').relative_to(R)),package='deliverables/videomaev2_global_candidate_20260915_v2',best_confirmed_Public=.83582)
state['latest_ready_submission']=ready;state['candidate_delivery']=copy.deepcopy(ready);state['best_confirmed_submission']=copy.deepcopy(ready);state['latest_submission']=feedback
prior=read(ev/'PRIOR_CURRENT_FIELDS.json')['candidate_delivery'];prior.update(Public_score=.79104,uploaded=True,uploaded_by_user=True,status='USER_SUBMITTED_SECOND_BEST_PUBLIC',best_confirmed_Public=.83582);state['backup_candidate_delivery']=prior
state['pending_user_action']=dict(kind='CONFIRM_EXACT_DEADLINE_AND_PERSONALLY_SELECT_FINAL_TWO',scope='Research continues on already authorized dual hosts. No upload pending for existing 0.83582 CSV. User must personally confirm final two submissions; exact deadline time requested asynchronously.',current_recommended_files=[csvp.name,old.name],select_boxes_empty_in_latest_user_screenshot=True,no_assistant_Kaggle_action=True,blocks_authorized_training=False)
state['final_two_submissions']=[dict(csv=str(csvp.relative_to(R)),Public_score=.83582,csv_sha256=sha(csvp)),dict(csv=str(old.relative_to(R)),Public_score=.79104,csv_sha256=sha(old))]
state.update(phase='V2_GLOBAL_083582_CONFIRMED_BEST_V2_BASE_DUAL_RUNNING',updated_at_local=stamp,updated_at_unix=now,last_updated_local=stamp,updated_at=stamp,next_frozen_candidate=None)
snapshot=read(E/'watch/SNAPSHOT.json');state['runtime'].update(checked_at_local=stamp,snapshot=str((E/'watch/SNAPSHOT.json').relative_to(R)),latest_snapshot=str((E/'watch/SNAPSHOT.json').relative_to(R)),current_video_stage=dict(root=str(E.relative_to(R)),protocol_sha256=ph,launch=str((E/'LAUNCH.json').relative_to(R)),watch=str((E/'watch/SNAPSHOT.json').relative_to(R))),latest_controller_completion='KD_FOUR_FITS_AND_PAIRED_REVIEW_COMPLETE')
for host in ['cloud','local']:
 v=snapshot[host];status=v.get('JOB_STATUS.json',v.get('STATUS.json',{}));h=v.get('HISTORY.json',[])
 state['runtime'][host]=dict(root=str(E.relative_to(R)),status='RUNNING',deadline_unix=s['deadline_unix'],last_observed_at_unix=snapshot['at_unix'],job=status.get('job'),epoch=status.get('epoch'),phase=status.get('phase'),telemetry=v.get('TELEMETRY.json'),history_last=h[-1] if h else None)
 for run in state['active_runs']:
  if run['host']==host:run.update(status='RUNNING',last_observed_at_unix=snapshot['at_unix'])
state['runtime']['engineering_incidents_current']['local_Base_micro8']='Zero-update QA gradient-equivalence assertion failed; no change to criteria, no micro8 adoption; original error retained and owned worker verified dead. Current fit uses known micro4.'
state['runtime']['last_guardian_check']=dict(at_unix=now,cloud_and_local_real_updates=True,current_snapshot=str((E/'watch/SNAPSHOT.json').relative_to(R)),action='Continue exact registered Base stages; review automatic on cloud pair download; no budget reset.')
state['current_research']['next_work']='Review exact V2Base two cloud outer endpoints against old Base and current V2Small/global7. Local provisional sourceF3inner2 is running. If worthwhile, separately register remaining five source fits and actual fixed50INT8 replay; no fusion by Public score.'
state['current_research']['next_research_direction']='V2Base capacity plus stronger public K710 vision-only pretraining; old exact matched KD and compact dual completed without promotion.'
state['evidence_scope']['future_CSV_timing']='Current V2global confirmed Public0.83582. Future CSV conditional on completed source-isolated calibrated and serialized-size-checked evidence; competition date Sep15 verified, exact hour pending.'
state['new_public_followup_20260915'].update(best_Public=.83582,feedback=str((ev/'PUBLIC_FEEDBACK.json').relative_to(R)),v2_quant='All eight actual49-tensor replays completed before current qualified release',ongoing='V2Base cloud two outer16 and local one provisional source-inner16; finite bounds and matched controls frozen before updates.')
atomic(R/'PROJECT_STATE.json',state)
print(json.dumps(dict(public_score=.83582,final_two=[x['csv'] for x in state['final_two_submissions']],review_registration_sha256=sha(regp),snapshot_at_unix=snapshot['at_unix'],at_unix=now)))
