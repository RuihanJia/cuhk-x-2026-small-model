from pathlib import Path
import shutil
E=Path(__file__).resolve().parent;C=E.parent;OLD=C/'v2_base6_small8_production'
for n in ('resilient_io.py','dispatch.py'):shutil.copyfile(OLD/n,E/n)
s=(E/'prepared_infer.py').read_text().replace("PAIR_BASE6_SMALL8_FP32","WEIGHT_AVERAGED_BASE_FP32")
(E/'selected_infer.py').write_text(s)
s=(OLD/'audit_math.py').read_text().replace('(4, 40)','(3, 40)').replace('four-branch','three-branch');(E/'audit_math.py').write_text(s)
s=(OLD/'worker.py').read_text().replace('COMPLETE_V2_PAIRED_INFERENCE','COMPLETE_WEIGHT_AVERAGE_INFERENCE');(E/'worker.py').write_text(s)
s=(OLD/'controller.py').read_text().replace('COMPLETE_V2_PAIRED_INFERENCE','COMPLETE_WEIGHT_AVERAGE_INFERENCE').replace("<=3600 and s['deadline_unix']<=s['registered_at_unix']+3720","<=5400 and s['deadline_unix']<=s['registered_at_unix']+5520")
s=s.replace("own=read(prior/'CONTROLLER.json')['worker'];assert not alive(own);break","own=read(prior/'CONTROLLER.json');assert pd['exit_code']==0;assert read(prior/'DONE.json')['status']==predecessor['success_status']\n    if not alive(own['worker']) and not alive(own['controller']):break")
(E/'controller.py').write_text(s)
s=(OLD/'final_audit.py').read_text().replace('(405,4,40)','(405,3,40)').replace('(405,4)','(405,3)').replace('PASS_V2_BASE6_SMALL8_FINAL405','PASS_WEIGHT_AVERAGED_BASE_FINAL405').replace("checkpoints/PAIR.bin","checkpoints/B.pt")
(E/'final_audit.py').write_text(s)
s=(OLD/'source_qa.py').read_text()
s=s.replace("for kind,relative,cls in [('Base','videomaev2_base/model.py','VideoMAEBase'),('Small','videomaev2_small/model.py','VideoMAESmall')]:","for kind,relative,cls in [('Base','videomaev2_base/model.py','VideoMAEBase')]:")
s=s.replace("model=infer.constructor(D/cfg['checkpoints']['PAIR']['path'],kind,cfg)","packed=torch.load(D/cfg['checkpoints']['B']['path'],map_location='cpu',weights_only=True);model=infer.constructor(packed,cfg)")
s=s.replace('PASS_PAIRED_FULL18_SOURCE_CPU_PARITY','PASS_WEIGHT_AVERAGED_FULL18_SOURCE_CPU_PARITY')
(E/'source_qa.py').write_text(s)
print('Runtime and audit copies prepared; originals unchanged')
