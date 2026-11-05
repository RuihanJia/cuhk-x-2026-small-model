"""Optional local wall cap; inherited telemetry and stop handling remain active."""
from dual_host_runtime.health import Guard as TimedGuard

class Guard(TimedGuard):
    def bounds(self):
        if self.deadline > 0:
            return super().bounds()
        if any((p/'STOP_REQUESTED').exists() for p in [self.out,self.out.parent]):
            raise InterruptedError('STOP_REQUESTED')
