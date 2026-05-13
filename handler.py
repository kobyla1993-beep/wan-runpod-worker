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

        prompt = job_input.get(
            "prompt",
            "a cute robot walking in snow"
        )

        negative_prompt = job_input.get(
            "negative_prompt",
            "low quality, blurry, distorted, watermark, text"
        )

        fps = int(job_input.get("fps", 8))
        guidance_scale = float(job_input.get("guidance_scale", 5.0))

        # TESTOVACÍ RYCHLÉ NASTAVENÍ
        num_frames = int(job_input.get("num_frames", 9))
        steps = int(job_input.get("steps", 5))

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{uuid.uuid4().hex}.mp4"
        )

        pipeline = load_pipeline()

        print("=== START GENERATION ===", flush=True)
        print("Prompt:", prompt, flush=True)
        print("Frames:", num_frames, flush=True)
        print("Steps:", steps, flush=True)

        start = time.time()

        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        )

        frames = result.frames[0]

        export_to_video(
            frames,
            output_path,
            fps=fps
        )

        total = round(time.time() - start, 2)

        file_size = os.path.getsize(output_path)

        print("=== VIDEO GENERATED ===", flush=True)
        print("Output:", output_path, flush=True)
        print("Size:", file_size, flush=True)
        print("Generation time:", total, flush=True)

        return {
            "ok": True,
            "message": "Video generated",
            "video_path": output_path,
            "file_size": file_size,
            "generation_time": total,
            "settings": {
                "prompt": prompt,
                "num_frames": num_frames,
                "steps": steps,
                "fps": fps,
                "guidance_scale": guidance_scale
            }
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