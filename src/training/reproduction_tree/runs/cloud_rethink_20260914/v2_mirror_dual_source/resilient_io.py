"""Process-local8second sharing retry; original runtime files remain immutable."""
import time
def retry(fn):
 deadline=time.monotonic()+8
 while True:
  try:return fn()
  except PermissionError:
   if time.monotonic()>=deadline:raise
   time.sleep(.05)
def install():
 import dual_host_runtime.io as io
 io.retry=retry
