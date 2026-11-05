"""Complete portable bindings and old fixed three-branch comparison; no fit."""
from pathlib import Path
import ast,hashlib,json
E=Path(__file__).resolve().parent;C=E.parent;O=C.parent/'error_resolution_20260913';H=C/'v2_temporal_halves_fusion'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 s=read(E/'protocol_candidate.json');assert s['status']=='PREPARED_NOT_EXECUTABLE'
 depth=C/'v2_depth_residual_pilot/protocol.json';ds=read(depth)
 assert sha(depth)=='1b6dbde0e054b97824d88621a2a14a51d467840b841cb48b3b32dda962905502'
 for host in ('local','cloud'):
  s['hosts'][host]['predecessor']=dict(directory=ds['hosts'][host]['output_directory'],protocol_sha256=sha(depth),success_only=True)
  s['hosts'][host]['workers']=2 if host=='local' else 12
  s['hosts'][host]['wall_cap_seconds']=3300
 for name in ('controller.py','resilient_io.py','watch.py'):
  s['code_sha256']['v2_context_readout/'+name]=sha(E/name)
 s['controller_sha256']=sha(E/'controller.py');s['wall_cap_seconds']=3600
 inp=s['calibration_inputs'];hs=read(H/'protocol.json')
 inp.update(context_outputs=dict(local=s['hosts']['local']['output_directory'],cloud=str(E/'cloud')),source_oof=hs['source_oof'],source_oof_sha256=hs['source_oof_sha256'],old_calibrator_barrier=str(H/'output/CALIBRATORS_FROZEN.json'))
 fs=read(C/'trajectory_fixed_fusion_protocol.json')
 def local_source(p):
  remote='/home/featurize/cuhkx_motion_transfer_20260911_v1/'
  return C.parent/p[len(remote):] if p.startswith(remote) else Path(p)
 inp['outer_scores']={str(f):dict(path=str(local_source(fs['inputs'][str(f)]['scores'])),sha256=fs['inputs'][str(f)]['scores_sha256']) for f in (0,3)}
 bound={inp[k]:inp[k+'_sha256'] for k in ('source_fitted','old_paired','old_protocol','source_oof','original_whole3_paired')}
 for v in inp['old_calibrators'].values():
  for item in v.values():bound[item['path']]=item['sha256']
 bound[inp['old_calibrator_barrier']]=sha(inp['old_calibrator_barrier'])
 for item in inp['outer_scores'].values():bound[item['path']]=item['sha256']
 for p,h in bound.items():assert sha(p)==h,p
 inp['bound_files']=bound
 s['calibration_code_sha256']={p:sha(C/p) for p in ['v2_context_readout/calibrate.py','v2_context_readout/classwise4.py','v2_temporal_halves_fusion/classwise3.py','frozen_fusion_math.py']}
 s['assets_restore_manifest']=dict(path=str(C/'cloud_access_20260915_52910/ASSET_MANIFEST.json'),sha256=sha(C/'cloud_access_20260915_52910/ASSET_MANIFEST.json'))
 manifest=read(s['assets_restore_manifest']['path']);assets={v['remote']:v for v in manifest['files']}
 for job in s['jobs']:
  if job['host']=='cloud':
   key=job['parent_directory']+'/model_fp16.pt';assert assets[key]['sha256']==job['parent_fp16_sha256'],key
 for p in E.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
 (E/'protocol_candidate.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(dict(status='PREPARED_NOT_EXECUTABLE',candidate_sha256=sha(E/'protocol_candidate.json'),all_cloud6_parents_match_restored_asset_manifest=True,calibration_bound_files=len(bound))))
if __name__=='__main__':main()
