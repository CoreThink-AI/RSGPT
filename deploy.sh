#!/usr/bin/env bash
set -euo pipefail

PROJECT=biochem-db-by-hobs
REGION=us-central1
SERVICE=rsgpt
IMAGE=gcr.io/$PROJECT/$SERVICE
BUCKET=$PROJECT-rsgpt-models

# ── 1. Create model bucket and upload weights ──────────────────────────────
echo "==> Creating GCS bucket $BUCKET (if not exists)"
gcloud storage buckets create gs://$BUCKET \
    --project=$PROJECT \
    --location=$REGION \
    --uniform-bucket-level-access 2>/dev/null || echo "Bucket already exists"

echo "==> Uploading model (6.1 GB — this takes a few minutes)"
gcloud storage cp models/finetune_full.pth gs://$BUCKET/finetune_full.pth

# ── 2. Build and push container ────────────────────────────────────────────
echo "==> Building container image"
gcloud builds submit \
    --project=$PROJECT \
    --tag=$IMAGE \
    --machine-type=e2-highcpu-32 \
    --timeout=30m \
    .

# ── 3. Deploy Cloud Run service with GPU + GCS volume mount ───────────────
echo "==> Deploying Cloud Run service"
gcloud beta run deploy $SERVICE \
    --project=$PROJECT \
    --region=$REGION \
    --image=$IMAGE \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --memory=16Gi \
    --cpu=4 \
    --no-cpu-throttling \
    --timeout=300 \
    --concurrency=4 \
    --min-instances=1 \
    --max-instances=3 \
    --add-volume=name=models,type=cloud-storage,bucket=$BUCKET \
    --add-volume-mount=volume=models,mount-path=/models \
    --set-env-vars=MODEL_PATH=/models/finetune_full.pth,WANDB_MODE=disabled \
    --allow-unauthenticated

echo "==> Done. Service URL:"
gcloud run services describe $SERVICE --region=$REGION --format="value(status.url)"
