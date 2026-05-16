import os
import sys
import time
import uuid
import base64
import traceback

import torch
import runpod
import imageio

from diffusers import AutoencoderKLWan, WanPipeline


print("=== WAN VIDEO WORKER STARTED ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Torch: {torch.__version__}", flush=True)
print(f"CUDA: {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OUTPUT_DIR = "/tmp/wan_outputs"

pipe = None


def load_model():
    """
    Načte WAN model do paměti.

    Tohle se udělá jen jednou při prvním jobu.
    Pak worker zůstane warm a další joby jsou rychlejší.
    """
    global pipe

    if pipe is not None:
        print("=== MODEL ALREADY LOADED ===", flush=True)
        return pipe

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== LOADING WAN VAE ===", flush=True)

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    print("=== LOADING WAN PIPELINE ===", flush=True)

    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    pipe.to("cuda")

    print("=== MODEL LOADED ===", flush=True)

    return pipe


def video_to_base64(video_path):
    """
    Vezme MP4 soubor a převede ho na base64 string.

    Proč:
    RunPod serverless nemůže uživateli jen tak dát soubor z /tmp.
    /tmp je uvnitř containeru.
    Base64 umožní nacpat video přímo do JSON response.

    Nevýhoda:
    Base64 zvětší data asi o třetinu.
    Pro velká videa je to prasárna.
    Pro testy je to ale ideální.
    """
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    return base64.b64encode(video_bytes).decode("utf-8")


def save_video(frames, fps):
    """
    Uloží vygenerované framy jako MP4.

    frames = seznam obrázků
    fps = počet snímků za sekundu
    """
    filename = f"{uuid.uuid4().hex}.mp4"
    video_path = os.path.join(OUTPUT_DIR, filename)

    print(f"=== SAVING VIDEO TO {video_path} ===", flush=True)

    imageio.mimsave(
        video_path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
    )

    return video_path


def handler(job):
    """
    Hlavní funkce workeru.

    RunPod sem pošle job.
    My z něj vytáhneme input.
    Spustíme WAN.
    Uložíme MP4.
    Vrátíme JSON s base64 videem.
    """
    try:
        print("=== JOB RECEIVED ===", flush=True)
        print(job, flush=True)

        job_input = job.get("input", {})

        prompt = job_input.get("prompt", "a cute robot walking in snow")
        num_frames = int(job_input.get("num_frames", 9))
        steps = int(job_input.get("steps", 5))
        fps = int(job_input.get("fps", 8))
        guidance_scale = float(job_input.get("guidance_scale", 5.0))

        return_base64 = bool(job_input.get("return_base64", True))

        print("=== SETTINGS ===", flush=True)
        print(f"Prompt: {prompt}", flush=True)
        print(f"Frames: {num_frames}", flush=True)
        print(f"Steps: {steps}", flush=True)
        print(f"FPS: {fps}", flush=True)
        print(f"Guidance scale: {guidance_scale}", flush=True)
        print(f"Return base64: {return_base64}", flush=True)

        pipeline = load_model()

        print("=== GENERATING VIDEO ===", flush=True)

        start_time = time.time()

        result = pipeline(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        )

        generation_time = round(time.time() - start_time, 2)

        print("=== VIDEO GENERATED ===", flush=True)

        frames = result.frames[0]

        video_path = save_video(frames, fps)

        file_size = os.path.getsize(video_path)

        output = {
            "ok": True,
            "message": "Video generated",
            "video_path": video_path,
            "file_size": file_size,
            "generation_time": generation_time,
            "settings": {
                "prompt": prompt,
                "num_frames": num_frames,
                "steps": steps,
                "fps": fps,
                "guidance_scale": guidance_scale,
            },
        }

        if return_base64:
            print("=== ENCODING VIDEO TO BASE64 ===", flush=True)

            output["video_base64"] = video_to_base64(video_path)
            output["video_mime_type"] = "video/mp4"
            output["video_filename"] = os.path.basename(video_path)

            print("=== BASE64 READY ===", flush=True)

        print("=== JOB COMPLETE ===", flush=True)

        return output

    except Exception as e:
        print("=== ERROR ===", flush=True)
        print(str(e), flush=True)
        traceback.print_exc()

        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


print("=== STARTING SERVERLESS ===", flush=True)

runpod.serverless.start({
    "handler": handler
})