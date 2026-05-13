import os
import sys
import time
import uuid
import traceback

import torch
import runpod

from diffusers import AutoencoderKLWan, WanPipeline
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

    print("=== LOADING WAN VAE ===", flush=True)

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32
    )

    print("=== LOADING WAN PIPELINE ===", flush=True)

    start = time.time()

    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        torch_dtype=torch.bfloat16
    )

    pipe.to("cuda")

    print("=== WAN PIPELINE LOADED ===", flush=True)
    print("LOAD TIME:", round(time.time() - start, 2), flush=True)

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
                "error": "missing prompt"
            }

        pipeline = load_pipeline()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{uuid.uuid4().hex}.mp4"
        )

        print("=== START GENERATION ===", flush=True)

        start = time.time()

        result = pipeline(
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted, watermark, text",
            height=480,
            width=832,
            num_frames=17,
            num_inference_steps=10,
            guidance_scale=5.0
        )

        frames = result.frames[0]

        export_to_video(
            frames,
            output_path,
            fps=8
        )

        total = round(time.time() - start, 2)

        print("=== VIDEO GENERATED ===", flush=True)
        print(output_path, flush=True)

        return {
            "ok": True,
            "video_path": output_path,
            "generation_time": total
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