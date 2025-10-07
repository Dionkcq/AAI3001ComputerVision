FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data/hf-cache /data/hf-home /data/uploads && chmod -R 777 /data
ENV HF_HOME=/data/hf-home
ENV HUGGINGFACE_HUB_CACHE=/data/hf-cache
ENV HF_HUB_CACHE=/data/hf-cache
ENV HF_CACHE_DIR=/data/hf-cache
ENV XDG_CACHE_HOME=/data/hf-cache

ENV HF_SPACE=1
ENV PORT=7860

# Serve Flask via Gunicorn (module:variable => app:app)
CMD ["gunicorn","app:app","--bind=0.0.0.0:7860","--workers=1","--threads=8","--timeout=120"]
