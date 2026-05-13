import os
import sys
import time
import uuid
import traceback

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
print("CUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_pipeline():
    global pipe

    if pipe is not None:
        return pipe

    print("=== LOADING WAN PIPELINE ===", flush=True)
    start = time.time()

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16
    )

    pipe.to("cuda")

    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()

    print("=== WAN PIPELINE LOADED ===", flush=True)
    print("Load seconds:", round(time.time() - start, 2), flush=True)

    return pipe


def handler(job):
    try:
        print("=== JOB RECEIVED ===", flush=True)
        print(job, flush=True)

        if not torch.cuda.is_available():
            return {
                "ok": False,
                "error": "CUDA is not available"
            }

        job_input = job.get("input", {})

        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get(
            "negative_prompt",
            "blur, low quality, distorted, ugly, bad anatomy, watermark, text"
        )

        num_frames = int(job_input.get("num_frames", 33))
        steps = int(job_input.get("steps", 20))
        guidance_scale = float(job_input.get("guidance_scale", 5.0))
        fps = int(job_input.get("fps", 16))

        if not prompt:
            return {
                "ok": False,
                "error": "Missing input.prompt"
            }

        if num_frames < 9:
            num_frames = 9

        if num_frames > 49:
            num_frames = 49

        if steps < 5:
            steps = 5

        if steps > 30:
            steps = 30

        pipeline = load_pipeline()

        output_name = f"wan_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        print("=== GENERATING VIDEO ===", flush=True)
        print("Prompt:", prompt, flush=True)
        print("Frames:", num_frames, flush=True)
        print("Steps:", steps, flush=True)

        start = time.time()

        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
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

        file_size = os.path.getsize(output_path)

        print("=== VIDEO DONE ===", flush=True)
        print("Output:", output_path, flush=True)
        print("Size:", file_size, flush=True)
        print("Seconds:", round(time.time() - start, 2), flush=True)

        return {
            "ok": True,
            "message": "Video generated successfully",
            "model": MODEL_ID,
            "output_path": output_path,
            "file_size": file_size,
            "settings": {
                "prompt": prompt,
                "num_frames": num_frames,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "fps": fps
            }
        }

    except Exception as e:
        print("=== ERROR ===", flush=True)
        traceback.print_exc()

        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


print("=== STARTING RUNPOD SERVERLESS ===", flush=True)

runpod.serverless.start({
    "handler": handler
})