#!/bin/bash
set -e

MODEL_PATH="${MODEL_PATH:-/models/rsgpt.gguf}"
LLAMA_PORT="${LLAMA_PORT:-8181}"
PORT="${PORT:-8080}"
N_CTX="${N_CTX:-512}"
N_THREADS="${N_THREADS:-4}"

echo "[start] Waiting for model file at ${MODEL_PATH} ..."
until [ -f "${MODEL_PATH}" ]; do sleep 2; done
echo "[start] Model found ($(du -h "${MODEL_PATH}" | cut -f1))"

echo "[start] Starting llama-server on port ${LLAMA_PORT} ..."
llama-server \
    --model        "${MODEL_PATH}" \
    --ctx-size     "${N_CTX}" \
    --threads      "${N_THREADS}" \
    --port         "${LLAMA_PORT}" \
    --host         127.0.0.1 \
    --no-mmap \
    --log-disable \
    &
LLAMA_PID=$!

# Wait until llama-server is ready
echo "[start] Waiting for llama-server to be ready ..."
until curl -sf "http://127.0.0.1:${LLAMA_PORT}/health" > /dev/null 2>&1; do
    sleep 2
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "[start] llama-server crashed, exiting"
        exit 1
    fi
done
echo "[start] llama-server ready."

echo "[start] Starting FastAPI on port ${PORT} ..."
exec /app/.venv/bin/uvicorn app:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers 1 \
    --timeout-keep-alive 300
