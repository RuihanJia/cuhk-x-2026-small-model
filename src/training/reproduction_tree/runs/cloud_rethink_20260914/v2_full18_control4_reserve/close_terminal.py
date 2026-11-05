from pathlib import Path
import hashlib,json,time
E=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.loads((E/'INDEPENDENT_STORAGE_AUDIT.json').read_text());assert a['status']=='PASS_INDEPENDENT_FULL18_CONTROL4_AVERAGE_STORAGE'
assert sha(E/'cloud/average_model_int8.pt')==a['model_sha256']
files=['protocol.cloud.json','CPU_QA.json','SAMPLER_INDEPENDENT_QA.json','INDEPENDENT_STORAGE_AUDIT.json','cloud/AVERAGE_FROZEN.json','cloud/CONTROLLER_DONE.json','cloud/REMOTE_MODEL_BINDINGS.json','collector/DONE.json']
record=dict(status='COMPLETE_FULL18_CONTROL4_AND_AVERAGE_RESERVE_WITH_INDEPENDENT_AUDIT',artifacts={p:sha(E/p)for p in files},local_average_int8=str(E/'cloud/average_model_int8.pt'),local_average_int8_sha256=a['model_sha256'],model_bytes=a['model_bytes'],no_Test=True,no_final_calibration=True,no_selection_claim=True,remote_models_retained=True,owned_GPU_resources_released=True,at_unix=time.time())
(E/'COMPLETION_TERMINAL.json').write_text(json.dumps(record,indent=2));print(json.dumps(dict(status=record['status'],terminal_sha256=sha(E/'COMPLETION_TERMINAL.json'))))
