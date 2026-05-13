import runpod

def handler(job):
    job_input = job.get("input", {})

    return {
        "ok": True,
        "message": "RunPod worker běží 😄",
        "input": job_input
    }

runpod.serverless.start({"handler": handler})