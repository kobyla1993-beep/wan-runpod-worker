FROM runpod/pytorch:1.0.3-cu1281-torch260-ubuntu2204

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV HUGGINGFACE_HUB_CACHE=/tmp/huggingface/hub
ENV TOKENIZERS_PARALLELISM=false

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]