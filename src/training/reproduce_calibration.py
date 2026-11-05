"""Refit selected calibrator using ONLY archived training-user OOF predictions."""
from pathlib import Path
import argparse,importlib.util,json,sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;TREE=HERE/'reproduction_tree';C=TREE/'runs/cloud_rethink_20260914'
def main():
 p=argparse.ArgumentParser();p.add_argument('--solution',choices=['dual_mirror','weight_average'],required=True);p.add_argument('--output',required=True);p.add_argument('--source-npz',type=Path);p.add_argument('--base-source-npz',type=Path);a=p.parse_args();torch.set_num_threads(1)
 sys.path.insert(0,str(C));import frozen_fusion_math as fm
 folder='v2_mirror_dual_production' if a.solution=='dual_mirror' else 'v2_weight_average_production';s=json.loads((C/folder/'protocol.json').read_text(encoding='utf-8'))
 def mapped(old):
  text=old.replace('\\','/');return TREE/('runs/'+text.split('/runs/',1)[1])
 v=dict(np.load(a.source_npz or mapped(s['source_npz']),allow_pickle=False));src=dict(np.load(a.base_source_npz or mapped(s['base_source_npz']),allow_pickle=False))
 assert len(v['targets'])==4768 and len(set(v['clip_ids']))==2931 and set(v['users'])==set(s['training_users'])
 index={(int(f),str(cid)):i for i,(f,cid)in enumerate(zip(src['source_fold'],src['original_clip_ids']))};ix=np.asarray([index[(int(f),str(cid))]for f,cid in zip(v['source_fold'],v['clip_ids'])]);assert np.array_equal(src['targets'][ix],v['targets']) and np.array_equal(src['users'][ix],v['users'])
 mirror=a.solution=='dual_mirror';logits=np.concatenate([v['TI'],v['base_mixed'][:,None],v['small_mixed'][:,None]],1) if mirror else np.concatenate([v['TI'],v['video'][:,None]],1)
 name='convex_BASE6_SMALL8_TI4' if mirror else 'convex_T_I_AVERAGE3';fm.STRATEGIES[name]=dict(branches=list(range(logits.shape[1])),mode='convex',availability=False)
 class Clock:
  def check(self,**kw):pass
 clock=Clock();d=dict(logits=logits,available=v['available'],targets=v['targets'],users=v['users'],depth=src['depth'][ix],ir=src['ir'][ix]);g=fm.fit(d,name,clock,500)
 file=C/('v2_context_readout/classwise4.py' if mirror else 'videomaev2_base_fusion/classwise3.py');spec=importlib.util.spec_from_file_location('selected_classwise',file);cw=importlib.util.module_from_spec(spec);spec.loader.exec_module(cw)
 cal=cw.fit(logits,v['available'],v['targets'],v['users'],g,clock);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(cal,separators=(',',':'))+'\n',encoding='utf-8');print(out)
if __name__=='__main__':main()
