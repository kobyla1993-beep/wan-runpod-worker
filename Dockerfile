FROM runpod/pytorch:2.6.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV HUGGINGFACE_HUB_CACHE=/tmp/huggingface/hub

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]