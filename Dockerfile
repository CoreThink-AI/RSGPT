FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system deps needed by rdkit
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy source (model loaded at runtime from GCS mount)
COPY app.py infer.py vocab.json ./
COPY configs/ configs/
COPY models/__init__.py models/rxngpt.py models/ema.py models/
COPY tokenizer/tokenization.py tokenizer/__init__.py tokenizer/
COPY utils/ utils/

ENV WANDB_MODE=disabled
ENV MODEL_PATH=/models/finetune_full.pth
ENV TOKENIZER_PATH=/app/vocab.json

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
