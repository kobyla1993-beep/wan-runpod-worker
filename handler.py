import os
import sys
import runpod

print("=== HANDLER.PY STARTED ===", flush=True)
print("Python:", sys.version, flush=True)
print("RunPod SDK:", getattr(runpod, "__version__", "unknown"), flush=True)

print("RUNPOD_ENDPOINT_ID:", os.getenv("RUNPOD_ENDPOINT_ID"), flush=True)
print("RUNPOD_POD_ID:", os.getenv("RUNPOD_POD_ID"), flush=True)

def handler(job):
    print("=== JOB RECEIVED ===", flush=True)
    print(job, flush=True)

    job_input = job.get("input", {})

    return {
        "ok": True,
        "message": "worker funguje",
        "input": job_input
    }

print("=== STARTING SERVERLESS ===", flush=True)

runpod.serverless.start({
    "handler": handler
})