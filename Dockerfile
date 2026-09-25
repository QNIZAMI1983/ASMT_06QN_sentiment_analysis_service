# Not python:3.8 - torch 2.4 wheels start at Python 3.9
FROM python:3.11-slim

WORKDIR /app

# Where Hugging Face caches model weights. Set before the download step so the
# weights land inside the image.
ENV HF_HOME=/app/model_cache \
    TRANSFORMERS_OFFLINE=0 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Install the CPU-only build of torch first. The default PyPI wheel drags in
# several hundred MB of CUDA libraries that are useless on ECS Fargate, which
# has no GPU.
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch==2.4.1 \
      --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Download the model at build time rather than on first request.
RUN python -c "import model; model.load_model(); print('Model cached at build time')"

# The model is now local, so refuse any runtime network calls to the Hub.
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8080

# Run as a non-root user
RUN useradd --create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app
USER appuser

# One worker: each worker holds its own copy of the model in memory, so more
# workers means proportionally more RAM. Scale by running more ECS tasks instead.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "120", "--preload", "app:app"]
