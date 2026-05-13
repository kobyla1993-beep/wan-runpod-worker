import os
import sys
import time
import uuid
import traceback

os.environ["DIFFUSERS_NO_ADDITIONAL_IMPORTS"] = "1"

import torch
import runpod

from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video


MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

OUTPUT_DIR = "/tmp/wan_outputs"

pipe = None


print("=== WAN VIDEO WORKER STARTED ===", flush=True)
print("Python:", sys.version, flush=True)
print("Torch:", torch.__version__, flush=True)
print("CUDA:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_pipeline():
    global pipe

    if pipe is not None:
        return pipe

    print("=== LOADING WAN MODEL ===", flush=True)

    start = time.time()

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16
    )

    pipe.to("cuda")

    print("=== MODEL LOADED ===", flush=True)
    print("Load time:", round(time.time() - start, 2), flush=True)

    return pipe


def handler(job):

    try:

        print("=== JOB RECEIVED ===", flush=True)
        print(job, flush=True)

        if not torch.cuda.is_available():
            return {
                "ok": False,
                "error": "CUDA unavailable"
            }

        job_input = job.get("input", {})

        prompt = job_input.get("prompt")

        if not prompt:
            return {
                "ok": False,
                "error": "Missing prompt"
            }

        pipeline = load_pipeline()

        print("=== GENERATING VIDEO ===", flush=True)

        start = time.time()

        result = pipeline(
            prompt=prompt,
            num_frames=17,
            num_inference_steps=10,
            guidance_scale=5.0
        )

        frames = result.frames[0]

        output_name = f"{uuid.uuid4().hex}.mp4"

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        export_to_video(
            frames,
            output_path,
            fps=8
        )

        generation_time = round(
            time.time() - start,
            2
        )

        print("=== VIDEO COMPLETE ===", flush=True)

        return {
            "ok": True,
            "video_path": output_path,
            "generation_time": generation_time
        }

    except Exception as e:

        traceback.print_exc()

        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


print("=== STARTING SERVERLESS ===", flush=True)

runpod.serverless.start({
    "handler": handler
})