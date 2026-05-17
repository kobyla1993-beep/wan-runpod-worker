import os
import uuid
import time
import base64
import traceback

import boto3
import torch

from diffusers import WanPipeline
from diffusers.utils import export_to_video

import runpod


MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

OUTPUT_DIR = "/tmp/wan_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pipe = None


def load_model():
    global pipe

    if pipe is not None:
        print("=== MODEL ALREADY LOADED ===")
        return pipe

    print("=== LOADING WAN PIPELINE ===")

    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    )

    pipe.to("cuda")

    print("=== MODEL LOADED ===")

    return pipe


def get_r2_client():
    account_id = os.environ["R2_ACCOUNT_ID"]

    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto"
    )


def upload_video_to_r2(video_path):
    bucket_name = os.environ["R2_BUCKET_NAME"]
    public_base = os.environ["R2_PUBLIC_BASE_URL"]

    object_name = f"videos/{uuid.uuid4().hex}.mp4"

    print(f"=== UPLOADING TO R2: {object_name} ===")

    s3 = get_r2_client()

    s3.upload_file(
        video_path,
        bucket_name,
        object_name,
        ExtraArgs={
            "ContentType": "video/mp4"
        }
    )

    video_url = f"{public_base}/{object_name}"

    print("=== R2 UPLOAD COMPLETE ===")
    print(video_url)

    return video_url


def generate_video(
    prompt,
    num_frames,
    steps,
    guidance_scale
):
    pipe = load_model()

    print("=== GENERATING VIDEO ===")

    result = pipe(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale
    )

    frames = result.frames[0]

    print("=== VIDEO GENERATED ===")

    return frames


def save_video(frames, fps):
    filename = f"{uuid.uuid4().hex}.mp4"

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    print(f"=== SAVING VIDEO TO {output_path} ===")

    export_to_video(
        frames,
        output_path,
        fps=fps
    )

    return output_path


def video_to_base64(video_path):
    print("=== ENCODING VIDEO TO BASE64 ===")

    with open(video_path, "rb") as f:
        encoded = base64.b64encode(
            f.read()
        ).decode("utf-8")

    print("=== BASE64 READY ===")

    return encoded


def cleanup_file(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"=== LOCAL VIDEO DELETED: {path} ===")


def handler(job):
    try:
        print("=== JOB RECEIVED ===")

        job_input = job["input"]

        prompt = job_input.get(
            "prompt",
            "a cinematic robot walking through snow at night"
        )

        num_frames = int(
            job_input.get("num_frames", 9)
        )

        steps = int(
            job_input.get("steps", 5)
        )

        fps = int(
            job_input.get("fps", 8)
        )

        guidance_scale = float(
            job_input.get("guidance_scale", 5)
        )

        upload_to_r2 = bool(
            job_input.get("upload_to_r2", True)
        )

        return_base64 = bool(
            job_input.get("return_base64", False)
        )

        delete_local_after_upload = bool(
            job_input.get(
                "delete_local_after_upload",
                True
            )
        )

        print("=== SETTINGS ===")
        print(job_input)

        start_time = time.time()

        frames = generate_video(
            prompt=prompt,
            num_frames=num_frames,
            steps=steps,
            guidance_scale=guidance_scale
        )

        video_path = save_video(
            frames,
            fps
        )

        response = {
            "ok": True
        }

        if upload_to_r2:
            video_url = upload_video_to_r2(
                video_path
            )

            response["video_url"] = video_url

        if return_base64:
            response["base64"] = video_to_base64(
                video_path
            )

        response["file_size"] = os.path.getsize(
            video_path
        )

        response["generation_time"] = round(
            time.time() - start_time,
            2
        )

        if delete_local_after_upload:
            cleanup_file(video_path)

        print("=== JOB COMPLETE ===")

        return response

    except Exception as e:
        print("=== ERROR ===")

        traceback.print_exc()

        return {
            "ok": False,
            "error": str(e)
        }


print("=== WAN VIDEO WORKER STARTED ===")
print(f"Python: {os.sys.version}")
print(f"Torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

print("=== PRELOADING MODEL ===")

load_model()

print("=== MODEL PRELOAD COMPLETE ===")

runpod.serverless.start(
    {
        "handler": handler
    }
)