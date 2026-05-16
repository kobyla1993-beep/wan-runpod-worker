import os
import sys
import time
import uuid
import base64
import traceback

import torch
import runpod
import imageio
import boto3

from botocore.client import Config
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


def env(name, default=None, required=False):
    """
    Bezpečně načte environment proměnnou.

    Environment proměnná = hodnota nastavená v RunPod endpointu.
    Například R2_BUCKET_NAME nebo R2_SECRET_ACCESS_KEY.

    required=True znamená:
    když proměnná chybí, worker hodí chybu.
    """
    value = os.getenv(name, default)

    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


def load_model():
    """
    Načte WAN model.

    Děláme lazy loading:
    model se nenačítá při startu containeru,
    ale až při prvním jobu.

    Výhoda:
    worker rychleji naběhne.

    Nevýhoda:
    první request je pomalejší.
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


def save_video(frames, fps):
    """
    Uloží framy do MP4 souboru.

    frames:
    seznam obrázků, které WAN vygeneroval

    fps:
    kolik snímků za sekundu video má mít
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


def video_to_base64(video_path):
    """
    Převede MP4 na base64.

    Necháváme to tu jako debug možnost.
    Normálně už budeme používat R2 upload.
    """
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    return base64.b64encode(video_bytes).decode("utf-8")


def get_r2_client():
    """
    Vytvoří klienta pro Cloudflare R2.

    R2 používá S3-compatible API.
    Proto endpoint vypadá takhle:

    https://ACCOUNT_ID.r2.cloudflarestorage.com
    """
    account_id = env("R2_ACCOUNT_ID", required=True)
    access_key_id = env("R2_ACCESS_KEY_ID", required=True)
    secret_access_key = env("R2_SECRET_ACCESS_KEY", required=True)

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_video_to_r2(video_path):
    """
    Nahraje MP4 do Cloudflare R2.

    Výsledek:
    vrací veřejnou URL, kterou může otevřít browser/frontend.

    Bucket musí být buď public,
    nebo musíš mít nastavenou custom public domain.
    """
    bucket_name = env("R2_BUCKET_NAME", required=True)
    public_base_url = env("R2_PUBLIC_BASE_URL", required=True).rstrip("/")

    filename = os.path.basename(video_path)
    object_key = f"videos/{filename}"

    print(f"=== UPLOADING TO R2: {object_key} ===", flush=True)

    s3 = get_r2_client()

    s3.upload_file(
        video_path,
        bucket_name,
        object_key,
        ExtraArgs={
            "ContentType": "video/mp4",
        },
    )

    video_url = f"{public_base_url}/{object_key}"

    print(f"=== R2 UPLOAD COMPLETE: {video_url} ===", flush=True)

    return video_url, object_key


def handler(job):
    """
    Hlavní RunPod handler.

    Přijme job:
    - načte input
    - vygeneruje video
    - uloží MP4
    - volitelně nahraje do R2
    - volitelně vrátí base64
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

        upload_to_r2 = bool(job_input.get("upload_to_r2", True))
        return_base64 = bool(job_input.get("return_base64", False))

        print("=== SETTINGS ===", flush=True)
        print(f"Prompt: {prompt}", flush=True)
        print(f"Frames: {num_frames}", flush=True)
        print(f"Steps: {steps}", flush=True)
        print(f"FPS: {fps}", flush=True)
        print(f"Guidance scale: {guidance_scale}", flush=True)
        print(f"Upload to R2: {upload_to_r2}", flush=True)
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
                "upload_to_r2": upload_to_r2,
                "return_base64": return_base64,
            },
        }

        if upload_to_r2:
            video_url, object_key = upload_video_to_r2(video_path)

            output["video_url"] = video_url
            output["r2_object_key"] = object_key

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