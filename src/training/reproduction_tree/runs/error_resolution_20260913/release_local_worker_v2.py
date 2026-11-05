from pathlib import Path
import sys,json,time,traceback
R=Path(__file__).resolve().parent
D=R.parents[1]/'deliverables/pose_dual_context_candidate_20260914'
sys.path.insert(0,str(D));sys.path.insert(1,str(R/'bundle'))
sys.path.append(str(R.parents[1]/'assets/pose/site'))
from dual_host_runtime.health import Guard
from dual_host_runtime.io import atomic
import infer

def main():
    out=R/'release_local_output_v2';out.mkdir(exist_ok=True)
    try:
        q=json.loads((R/'release_qa_output/DONE.json').read_text());assert q['status']=='PASS'
        guard=Guard(out,1789323192.0883653,out/'TELEMETRY.json')
        a=infer.infer(r'D:\Datasets\CUHK-X_Small_Model_Track\Small-Model-Track',D/'submission.csv',guard,workers=4,video_batch=4)
        atomic(out/'DONE.json',dict(status='COMPLETE',at_unix=time.time(),audit=a))
    except BaseException as e:
        atomic(out/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at_unix=time.time()));raise
if __name__=='__main__':main()
