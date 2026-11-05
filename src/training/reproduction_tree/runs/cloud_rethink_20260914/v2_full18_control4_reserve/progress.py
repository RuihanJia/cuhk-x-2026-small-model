from pathlib import Path
import sys,json
E=Path(__file__).resolve().parent;C=E.parent;sys.path.insert(0,str(C));import post_aux_cloud_access as a
remote=a.REMOTE+'/cloud_rethink_20260914/'+E.name
code='from pathlib import Path\nimport json\np=Path('+repr(remote)+')\nnames=["parent/model_fp16.pt.uploading","parent/model_fp16.pt","protocol.cloud.json","output/cloud/QUEUED.json","output/cloud/CONTROLLER_DONE.json","output/cloud/CONTROLLER_ERROR.json","output/cloud/TELEMETRY.json"]\nprint(json.dumps({n:dict(bytes=(p/n).stat().st_size,json=json.loads((p/n).read_text())if n.endswith("json")and n!="protocol.cloud.json"else None)for n in names if(p/n).exists()}))'
r=a.run(code,timeout=30);print(json.dumps(r));(E/'REMOTE_PROGRESS_SNAPSHOT.json').write_text(json.dumps(r,indent=2))
