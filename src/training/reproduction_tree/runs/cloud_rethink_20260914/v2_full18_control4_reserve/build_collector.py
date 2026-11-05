from pathlib import Path
E=Path(__file__).resolve().parent;C=E.parent
s=(C/'v2_true_fullframe_readout/collect_cloud.py').read_text()
s=s.replace('COMPLETE_TRUE_FULLFRAME_READOUT','COMPLETE_FULL18_CONTROL4_AND_FIXED_AVERAGE_RESERVE').replace('COMPLETE_CLOUD_FULLFRAME_ARTIFACT_COLLECTION','COMPLETE_CLOUD_FULL18_RESERVE_COLLECTION')
s=s.replace("ordered=sorted(snap['files'],key=lambda n:n=='CONTROLLER_DONE.json')","ordered=sorted([n for n in snap['files'] if not n.endswith('.pt') or n=='average_model_int8.pt'],key=lambda n:n=='CONTROLLER_DONE.json')")
s=s.replace("atomic(w/'DONE.json',dict(status=", "atomic(local/'REMOTE_MODEL_BINDINGS.json',dict(models={n:dict(**v,remote=remote+'/'+n) for n,v in snap['files'].items() if n.endswith('.pt') and n!='average_model_int8.pt'},checkpoint_downloaded='average_model_int8.pt',remote_owned_absent=True,protocol_sha256=ph));atomic(w/'DONE.json',dict(status=")
(E/'collect_reserve.py').write_text(s)
print('Reserved-artifact collector prepared')
