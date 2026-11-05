from pathlib import Path
import sys,json,hashlib
E=Path(__file__).resolve().parent;C=E.parent;sys.path.insert(0,str(C));import post_aux_cloud_access as a
remote=a.REMOTE+'/cloud_rethink_20260914/'+E.name
source=(E/'remote_storage_audit.py').read_text(encoding='utf-8-sig')
(E/'INDEPENDENT_AUDIT_REGISTRATION.json').write_text(json.dumps(dict(code_sha256=hashlib.sha256(source.encode()).hexdigest(),purpose='CPU exact raw/control/average/INT8 direct replay after successful terminal; no fit or GPU',remote_directory=remote),indent=2))
r=a.run(source+'\nprint(json.dumps(audit(Path('+repr(remote)+'))))',timeout=60)
(E/'INDEPENDENT_STORAGE_AUDIT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
