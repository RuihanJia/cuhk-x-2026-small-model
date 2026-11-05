from pathlib import Path
import importlib.util,sys
HERE=Path(__file__).resolve().parent
if sys.platform=='win32':HERE=Path('\\\\?\\'+str(HERE))
C=HERE/'reproduction_tree/runs/cloud_rethink_20260914'
P=C/'aux_public_regression_20260914/dynamic_interaction/photometric_candidate';O=C.parent/'error_resolution_20260913'
sys.path[:0]=[str(P),str(O),str(O/'bundle')]
spec=importlib.util.spec_from_file_location('control_window_data',C/'v2_window_cloud52910/data.py');wd=importlib.util.module_from_spec(spec);sys.modules[spec.name]=wd;spec.loader.exec_module(wd)
class WarmClips(wd.WindowClips):
 def __init__(self,*args,arm='RAW_CONTROL',**kwargs):super().__init__(*args,arm='CONTROL',**kwargs)
