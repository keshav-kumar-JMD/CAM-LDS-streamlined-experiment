from __future__ import annotations
import importlib.metadata
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from .common import write_json

class Meter:
    def __init__(self,root,run_id):
        self.root=Path(root);self.run_id=run_id;self.start=time.perf_counter()
        self.samples=[];self.stop=threading.Event();self.nv=None;self.gpu=None
        try:
            import pynvml
            pynvml.nvmlInit();self.nv=pynvml;self.gpu=pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:pass
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
        try:
            import torch
            if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        except ImportError:pass
    def loop(self):
        while not self.stop.wait(2):
            if self.nv:
                try:
                    utilization=self.nv.nvmlDeviceGetUtilizationRates(self.gpu)
                    self.samples.append({'elapsed':time.perf_counter()-self.start,
                        'gpu_utilization_pct':utilization.gpu,
                        'memory_bytes':self.nv.nvmlDeviceGetMemoryInfo(self.gpu).used,
                        'power_watts':self.nv.nvmlDeviceGetPowerUsage(self.gpu)/1000})
                except Exception:pass
    def snapshot(self):
        report={'wall_seconds':time.perf_counter()-self.start,'python':platform.python_version(),
            'platform':platform.platform(),'packages':{},'samples':list(self.samples),
            'gpu_metrics_scope':'PyTorch peaks: this process; NVML: whole device, sampled every 2 seconds',
            'gpu_usage_note':'Utilization and integrated power are estimates, not Colab billing units'}
        for name in ('torch','transformers','peft','accelerate','anthropic'):
            try:report['packages'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:pass
        if self.samples:
            report['mean_gpu_utilization_pct']=sum(s['gpu_utilization_pct'] for s in self.samples)/len(self.samples)
            report['sampled_peak_gpu_memory_bytes']=max(s['memory_bytes'] for s in self.samples)
            report['estimated_energy_wh']=sum((b['elapsed']-a['elapsed'])*b['power_watts']/3600
                                               for a,b in zip(self.samples,self.samples[1:]))
        try:
            import torch
            if torch.cuda.is_available():report.update(gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        except ImportError:pass
        try:report['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
        except (OSError,subprocess.SubprocessError):pass
        return report
    def close(self):self.stop.set();self.thread.join(timeout=3);return self.snapshot()

def cost(usage,batch=False):
    inp=usage.get('input_tokens',0);out=usage.get('output_tokens',0)
    read=usage.get('cache_read_input_tokens',0);write=usage.get('cache_creation_input_tokens',0)
    mult=5 if inp+read+write>100000 else 1
    # This workflow does not request caching. Preserve fields if API returns them.
    return mult*(inp*.10+out*.50+read*.01+write*.125)/1e6*(.5 if batch else 1)
