import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from infer import RSGPT

model: RSGPT = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = RSGPT()
    yield


app = FastAPI(title="RSGPT Retrosynthesis", lifespan=lifespan)


class PredictRequest(BaseModel):
    smiles: str
    beam_size: int = 3


class PredictResponse(BaseModel):
    smiles: str
    reactions: list[str]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        results = model.predict(req.smiles, beam_size=req.beam_size)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictResponse(smiles=req.smiles, reactions=results)


@app.get("/health")
def health():
    return {"status": "ok"}
