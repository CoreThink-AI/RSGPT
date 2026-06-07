FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# ── Build llama-server (CPU-only) ─────────────────────────────────────────────
RUN git clone --depth=1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp && \
    cmake -B /tmp/llama.cpp/build /tmp/llama.cpp \
          -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/llama.cpp/build --config Release \
          --target llama-server -j$(nproc) && \
    cp /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server && \
    rm -rf /tmp/llama.cpp

# ── Python API layer ──────────────────────────────────────────────────────────
WORKDIR /app

RUN python3 -m venv /app/.venv && \
    /app/.venv/bin/pip install --no-cache-dir \
        fastapi "uvicorn[standard]" httpx tokenizers rdkit-pypi

COPY vocab.json      /app/vocab.json
COPY app.py          /app/app.py
COPY deploy/start.sh /start.sh
RUN chmod +x /start.sh

# ── Environment ───────────────────────────────────────────────────────────────
# MODEL_PATH: GCS volume mount path set in Cloud Run YAML
# PORT:       Cloud Run injects this; uvicorn listens here
# LLAMA_PORT: internal llama-server port (not exposed externally)
ENV MODEL_PATH=/models/rsgpt.gguf \
    VOCAB_PATH=/app/vocab.json \
    LLAMA_PORT=8181 \
    PORT=8080 \
    N_CTX=512 \
    N_THREADS=4

EXPOSE 8080
CMD ["/start.sh"]
