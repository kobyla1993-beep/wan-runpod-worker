import os
import sys
import time
import uuid
import base64
import traceback
from pathlib import Path

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
OUTPUT_DIR = Path("/tmp/wan_outputs")

MAX_FRAMES = 49
MAX_STEPS = 30
MAX_PROMPT_LENGTH = 800
MAX_FPS = 24

pipe = None


def env(name, default=None, required=False):
    value = os.getenv(name, default)

    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


def clamp_int(value, default, minimum, maximum):
    """
    Převede hodnotu na int a omezí ji do bezpečného rozsahu.

    Proč:
    Když někdo pošle num_frames=9999,
    worker by se mohl vysrat na paměti nebo běžet půl dne.
    """
    try:
        value = int(value)
    except Exception:
        value = default

    return max(minimum, min(value, maximum))


def clamp_float(value, default, minimum, maximum):
    """
    Stejné jako clamp_int, ale pro desetinná čísla.
    """
    try:
        value = float(value)
    except Exception:
        value = default

    return max(minimum, min(value, maximum))


def cleanup_tmp_outputs(max_age_seconds=3600):
    """
    Smaže stará MP4 videa z /tmp.

    /tmp je uvnitř containeru.
    Když worker běží dlouho, soubory by se tam hromadily jak bordel pod postelí.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    now = time.time()

    for file_path in OUTPUT_DIR.glob("*.mp4"):
        try:
            age = now - file_path.stat().st_mtime

            if age > max_age_seconds:
                file_path.unlink()
                print(f"Deleted old temp file: {file_path}", flush=True)

        except Exception as e:
            print(f"Cleanup warning: {file_path} - {e}", flush=True)


def load_model():
    """
    Lazy-load WAN model.

    První job:
    - stáhne/načte model
    - dá ho na GPU

    Další job:
    - použije už načtený model
    - mnohem rychlejší start
    """
    global pipe

    if pipe is not None:
        print("=== MODEL ALREADY LOADED ===", flush=True)
        return pipe

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    Uloží framy jako MP4.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.mp4"
    video_path = OUTPUT_DIR / filename

    print(f"=== SAVING VIDEO TO {video_path} ===", flush=True)

    imageio.mimsave(
        str(video_path),
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
    )

    return video_path


def video_to_base64(video_path):
    """
    Debug mód.

    Pro produkci radši R2 URL.
    Base64 je velká textová nudle a žere response size.
    """
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_r2_client():
    """
    Vytvoří S3 klienta pro Cloudflare R2.
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
    Nahraje MP4 do R2 a vrátí veřejnou URL.
    """
    bucket_name = env("R2_BUCKET_NAME", required=True)
    public_base_url = env("R2_PUBLIC_BASE_URL", required=True).rstrip("/")

    object_key = f"videos/{video_path.name}"

    print(f"=== UPLOADING TO R2: {object_key} ===", flush=True)

    s3 = get_r2_client()

    s3.upload_file(
        str(video_path),
        bucket_name,
        object_key,
        ExtraArgs={
            "ContentType": "video/mp4",
            "CacheControl": "public, max-age=31536000",
        },
    )

    video_url = f"{public_base_url}/{object_key}"

    print(f"=== R2 UPLOAD COMPLETE ===", flush=True)
    print(video_url, flush=True)

    return video_url, object_key


def parse_input(job_input):
    """
    Zpracuje input z requestu.

    Tady jsou bezpečnostní limity.
    Když někdo pošle šílenost, ořízneme ji.
    """
    prompt = str(job_input.get("prompt", "a cute robot walking in snow")).strip()

    if not prompt:
        prompt = "a cute robot walking in snow"

    if len(prompt) > MAX_PROMPT_LENGTH:
        prompt = prompt[:MAX_PROMPT_LENGTH]

    num_frames = clamp_int(
        job_input.get("num_frames", 9),
        default=9,
        minimum=1,
        maximum=MAX_FRAMES,
    )

    steps = clamp_int(
        job_input.get("steps", 5),
        default=5,
        minimum=1,
        maximum=MAX_STEPS,
    )

    fps = clamp_int(
        job_input.get("fps", 8),
        default=8,
        minimum=1,
        maximum=MAX_FPS,
    )

    guidance_scale = clamp_float(
        job_input.get("guidance_scale", 5.0),
        default=5.0,
        minimum=1.0,
        maximum=15.0,
    )

    upload_to_r2 = bool(job_input.get("upload_to_r2", True))
    return_base64 = bool(job_input.get("return_base64", False))
    delete_local_after_upload = bool(job_input.get("delete_local_after_upload", True))

    return {
        "prompt": prompt,
        "num_frames": num_frames,
        "steps": steps,
        "fps": fps,
        "guidance_scale": guidance_scale,
        "upload_to_r2": upload_to_r2,
        "return_base64": return_base64,
        "delete_local_after_upload": delete_local_after_upload,
    }


def handler(job):
    """
    Hlavní RunPod handler.
    """
    video_path = None

    try:
        cleanup_tmp_outputs()

        print("=== JOB RECEIVED ===", flush=True)
        print(job, flush=True)

        job_input = job.get("input", {})
        settings = parse_input(job_input)

        print("=== SETTINGS ===", flush=True)
        print(settings, flush=True)

        pipeline = load_model()

        print("=== GENERATING VIDEO ===", flush=True)

        start_time = time.time()

        result = pipeline(
            prompt=settings["prompt"],
            num_frames=settings["num_frames"],
            num_inference_steps=settings["steps"],
            guidance_scale=settings["guidance_scale"],
        )

        generation_time = round(time.time() - start_time, 2)

        print("=== VIDEO GENERATED ===", flush=True)

        frames = result.frames[0]

        video_path = save_video(frames, settings["fps"])
        file_size = video_path.stat().st_size

        output = {
            "ok": True,
            "message": "Video generated",
            "file_size": file_size,
            "generation_time": generation_time,
            "settings": settings,
        }

        if settings["upload_to_r2"]:
            video_url, object_key = upload_video_to_r2(video_path)

            output["video_url"] = video_url
            output["r2_object_key"] = object_key

        else:
            output["video_path"] = str(video_path)

        if settings["return_base64"]:
            print("=== ENCODING VIDEO TO BASE64 ===", flush=True)

            output["video_base64"] = video_to_base64(video_path)
            output["video_mime_type"] = "video/mp4"
            output["video_filename"] = video_path.name

            print("=== BASE64 READY ===", flush=True)

        if settings["delete_local_after_upload"] and settings["upload_to_r2"]:
            try:
                video_path.unlink()
                print(f"=== LOCAL VIDEO DELETED: {video_path} ===", flush=True)
            except Exception as e:
                print(f"Local delete warning: {e}", flush=True)

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