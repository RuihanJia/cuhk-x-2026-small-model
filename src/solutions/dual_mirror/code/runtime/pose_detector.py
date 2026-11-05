"""Offline COCO detector reconstructed from safe tensors and YAML only."""
from pathlib import Path
import torch

def load_detector(state_path,yaml_path):
 import yaml
 from ultralytics import YOLO
 from ultralytics.nn.tasks import PoseModel
 from ultralytics.utils import SETTINGS
 SETTINGS.update({'sync':False})
 cp=torch.load(state_path,map_location='cpu',weights_only=True)
 net=PoseModel(cp['yaml'],ch=3,nc=1,data_kpt_shape=(17,3),verbose=False)
 net.load_state_dict(cp['state_dict'],strict=True);net.kpt_shape=[17,3];net.names={0:'person'};net.eval().requires_grad_(False)
 path=Path(yaml_path)
 if not path.exists():path.write_text(yaml.safe_dump(cp['yaml']),encoding='utf-8')
 else:assert yaml.safe_load(path.read_text(encoding='utf-8'))==cp['yaml']
 model=YOLO(str(path),task='pose');model.model=net
 return model
