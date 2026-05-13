import os
import sys
import time
import runpod


print("=== WAN RUNPOD WORKER STARTED ===", flush=True)
print("Python:", sys.version, flush=True)
print("RunPod SDK:", getattr(runpod, "__version__", "unknown"), flush=True)
print("RUNPOD_ENDPOINT_ID:", os.getenv("RUNPOD_ENDPOINT_ID"), flush=True)
print("RUNPOD_POD_ID:", os.getenv("RUNPOD_POD_ID"), flush=True)


def handler(job):
    print("=== JOB RECEIVED ===", flush=True)
    print(job, flush=True)

    job_input = job.get("input", {})

    prompt = job_input.get("prompt")
    seconds = job_input.get("seconds", 5)
    width = job_input.get("width", 512)
    height = job_input.get("height", 512)

    if not prompt:
        return {
            "ok": False,
            "error": "Missing required field: prompt"
        }

    try:
        seconds = int(seconds)
        width = int(width)
        height = int(height)
    except Exception:
        return {
            "ok": False,
            "error": "seconds, width and height must be numbers"
        }

    if seconds < 1 or seconds > 10:
        return {
            "ok": False,
            "error": "seconds must be between 1 and 10"
        }

    if width < 256 or height < 256:
        return {
            "ok": False,
            "error": "width and height must be at least 256"
        }

    print("=== VALID REQUEST ===", flush=True)
    print("Prompt:", prompt, flush=True)
    print("Seconds:", seconds, flush=True)
    print("Width:", width, flush=True)
    print("Height:", height, flush=True)

    time.sleep(1)

    return {
        "ok": True,
        "status": "validated",
        "message": "WAN request accepted. Model generation is next step.",
        "request": {
            "prompt": prompt,
            "seconds": seconds,
            "width": width,
            "height": height
        }
    }


print("=== STARTING RUNPOD SERVERLESS ===", flush=True)

runpod.serverless.start({
    "handler": handler
})