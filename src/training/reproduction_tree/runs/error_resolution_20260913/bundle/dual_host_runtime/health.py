"""Native two-second local thermal control; only the registered job tree is affected."""
import ctypes, os, time
from pathlib import Path
import psutil
from .io import read,atomic
class NVML:
    def __init__(self):
        self.dll=ctypes.CDLL('nvml.dll' if os.name=='nt' else 'libnvidia-ml.so.1')
        self.ok(self.dll.nvmlInit_v2());self.handle=ctypes.c_void_p()
        self.ok(self.dll.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle)))
    @staticmethod
    def ok(code):
        if code:raise RuntimeError('NVML_ERROR:'+str(code))
    def sample(self):
        class Memory(ctypes.Structure):_fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
        class Util(ctypes.Structure):_fields_=[('gpu',ctypes.c_uint),('memory',ctypes.c_uint)]
        t=ctypes.c_uint();m=Memory();u=Util()
        self.ok(self.dll.nvmlDeviceGetTemperature(self.handle,0,ctypes.byref(t)))
        self.ok(self.dll.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(m)))
        self.ok(self.dll.nvmlDeviceGetUtilizationRates(self.handle,ctypes.byref(u)))
        battery=psutil.sensors_battery() if os.name=='nt' else None
        return dict(at_unix=time.time(),temperature_c=t.value,memory_used_bytes=m.used,memory_total_bytes=m.total,gpu_utilization_percent=u.gpu,AC_power=(battery.power_plugged if battery else os.name!='nt'))
def decision(sample,cooling,local=True):
    if sample.get('error') or time.time()-sample.get('at_unix',0)>20:return 'WAIT_TELEMETRY'
    if local and not sample.get('AC_power'):return 'WAIT_AC'
    t=sample['temperature_c']
    if local and (t>=90 or cooling and t>85):return 'COOLING'
    if not local and t>=88:return 'CLOUD_TEMPERATURE_STOP'
    return 'WARNING' if local and t>=87 else 'RUN'
def identity(pid):return dict(pid=pid,create_time=psutil.Process(pid).create_time())
def alive(v):
    try:
        p=psutil.Process(v['pid']);return p.is_running() and p.status()!=psutil.STATUS_ZOMBIE and abs(p.create_time()-v['create_time'])<.01
    except psutil.NoSuchProcess:return False
def owned_tree(owner):
    if not alive(owner):return []
    p=psutil.Process(owner['pid']);return [identity(p.pid)]+[identity(v.pid) for v in p.children(recursive=True)]
def resume_all(values):
    for v in values:
        if alive(v):psutil.Process(v['pid']).resume()
def stop_owned(owner):
    values=owned_tree(owner)
    resume_all(values)
    for v in reversed(values):
        if alive(v):psutil.Process(v['pid']).terminate()
    _,remaining=psutil.wait_procs([psutil.Process(v['pid']) for v in values if alive(v)],timeout=5)
    for p in remaining:
        v=next(v for v in values if v['pid']==p.pid)
        if alive(v):p.kill()
class Pause:
    def __init__(self,owner):self.owner=owner;self.paused=[];self.cooling=False;self.since=None
    def tick(self,sample,local):
        state=decision(sample,self.cooling,local);pause=state.startswith('WAIT_') or state=='COOLING'
        if state=='COOLING':self.cooling=True
        elif state in ['RUN','WARNING']:self.cooling=False
        if pause:
            self.since=self.since or time.time()
            if not self.paused:
                self.paused=owned_tree(self.owner)
                for v in self.paused:
                    if alive(v):psutil.Process(v['pid']).suspend()
        else:resume_all(self.paused);self.paused=[];self.since=None
        return state
    def close(self):resume_all(self.paused);self.paused=[]
class Guard:
    def __init__(self,out,deadline,telemetry):self.out=Path(out);self.deadline=deadline;self.telemetry=Path(telemetry);self.last=0
    def bounds(self):
        if time.time()>=self.deadline:raise TimeoutError('FROZEN_WALL_CAP_INCOMPLETE')
        if any((p/'STOP_REQUESTED').exists() for p in [self.out,self.out.parent]):raise InterruptedError('STOP_REQUESTED')
    def check(self,force=False,**progress):
        self.bounds();now=time.time()
        sample=read(self.telemetry)
        if now-sample.get('at_unix',0)>20:raise InterruptedError('TELEMETRY_STALE')
        if sample.get('control')=='CLOUD_TEMPERATURE_STOP':raise InterruptedError('CLOUD_TEMPERATURE_STOP')
        if not force and now-self.last<15:return
        self.last=now;atomic(self.out/'STATUS.json',dict(status='RUNNING',pid=os.getpid(),at_unix=now,deadline_unix=self.deadline,**progress,telemetry=sample))
