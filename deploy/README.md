# RSGPT — Cloud Run Deployment

## Prerequisites
```bash
gcloud auth login
gcloud auth configure-docker
export PROJECT_ID=$(gcloud config get-value project)
```

## 1. Build and push container
```bash
docker build -t gcr.io/$PROJECT_ID/rsgpt:latest .
docker push gcr.io/$PROJECT_ID/rsgpt:latest
```

Or use Cloud Build (no local Docker needed):
```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/rsgpt:latest .
```

## 2. Enable required APIs
```bash
gcloud services enable \
  run.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com
```

## 3. Grant Cloud Run access to the GCS bucket
```bash
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gcloud storage buckets add-iam-policy-binding gs://biochem-db-by-hobs-rsgpt-models \
  --member="serviceAccount:service-${PROJECT_NUMBER}@serverless-robot-prod.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

## 4. Deploy
```bash
# Replace PROJECT_ID in the YAML, then deploy
sed "s/PROJECT_ID/$PROJECT_ID/" deploy/cloudrun.yaml | \
  gcloud run services replace - --region us-central1
```

## 5. Make public (optional)
```bash
gcloud run services add-iam-policy-binding rsgpt \
  --region us-central1 \
  --member="allUsers" \
  --role="roles/run.invoker"
```

## 6. Test
```bash
SERVICE_URL=$(gcloud run services describe rsgpt \
  --region us-central1 --format='value(status.url)')

curl -X POST "$SERVICE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "beam_size": 3}'
```

## Architecture
```
Cloud Run container
├── llama-server (port 8181, internal)  ← rsgpt.gguf via GCS volume mount
└── uvicorn/FastAPI (port 8080, external)
        tokenizes SMILES → token IDs
        POST /completion to llama-server
        decodes + parses reaction SMILES
```

## Resource sizing
| Field | Value | Notes |
|---|---|---|
| CPU | 4 vCPU | llama-server uses N_THREADS=4 |
| Memory | 8 GiB | F16 model ~3.2 GB + headroom |
| Min instances | 1 | Avoids ~60s cold-start model load |
| Max instances | 3 | Scale for concurrent requests |
| Timeout | 300s | CPU inference can be slow |
