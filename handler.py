import os
import sys
import time
import runpod
import torch


print("=== WAN GPU WORKER STARTED ===", flush=True)

print("Python:", sys.version, flush=True)
print("Torch version:", torch.__version__, flush=True)

print("CUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0), flush=True)
    print("GPU count:", torch.cuda.device_count(), flush=True)


def handler(job):

    print("=== JOB RECEIVED ===", flush=True)
    print(job, flush=True)

    gpu_available = torch.cuda.is_available()

    gpu_name = None

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "ok": True,
        "cuda_available": gpu_available,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "job_input": job.get("input", {})
    }


print("=== STARTING SERVERLESS ===", flush=True)

runpod.serverless.start({
    "handler": handler
})