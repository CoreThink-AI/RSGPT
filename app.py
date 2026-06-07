"""
RSGPT retrosynthesis API.
Tokenizes SMILES with vocab.json, sends token IDs to llama-server's /completion
endpoint, decodes the output, and parses into reaction SMILES.
"""
import json
import os
import re
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem
from tokenizers import Tokenizer as HFTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
VOCAB_PATH  = os.getenv("VOCAB_PATH",  "vocab.json")
LLAMA_PORT  = os.getenv("LLAMA_PORT",  "8181")
LLAMA_URL   = f"http://127.0.0.1:{LLAMA_PORT}"
EOS_ID      = 2   # </s>
MAX_TIMEOUT = 300  # seconds per request

# ── Tokenizer (loaded once at startup) ───────────────────────────────────────
_tokenizer: HFTokenizer = None
_id2tok: dict[int, str] = {}


def _load_tokenizer():
    global _tokenizer, _id2tok
    _tokenizer = HFTokenizer.from_file(VOCAB_PATH)
    vocab_data = json.loads(Path(VOCAB_PATH).read_text())
    vocab = vocab_data["model"]["vocab"]
    _id2tok = {v: k for k, v in vocab.items()}
    for entry in vocab_data.get("added_tokens", []):
        _id2tok[entry["id"]] = entry["content"]


def encode(text: str) -> list[int]:
    return _tokenizer.encode(text, add_special_tokens=False).ids


def decode_ids(ids: list[int]) -> str:
    return "".join(_id2tok.get(i, "") for i in ids)


# ── llama-server client ────────────────────────────────────────────────────────

async def _complete(
    client: httpx.AsyncClient,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 1,
) -> list[int]:
    """
    Call llama-server /completion with token IDs as the prompt.
    Returns list of generated token IDs (not including the prompt).
    """
    payload = {
        "prompt":      prompt_ids,   # llama-server accepts int[] directly
        "n_predict":   max_new_tokens,
        "temperature": temperature,
        "top_k":       top_k,
        "stop":        [],
        "stream":      False,
    }
    resp = await client.post(
        f"{LLAMA_URL}/completion",
        json=payload,
        timeout=MAX_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    # Response contains generated text; we need token IDs.
    # llama-server returns token IDs in "tokens" when "tokenize": true is set,
    # or we re-tokenize the generated text. The simpler path: re-encode.
    generated_text = data.get("content", "")
    return encode(generated_text) if generated_text else []


async def beam_search(
    prompt_ids: list[int],
    beam_size: int,
    max_new_tokens: int,
) -> list[str]:
    """
    Run beam_size independent completions and deduplicate.
    First pass is greedy (temp=0); subsequent passes use temp=0.7 for diversity.
    """
    results: list[str] = []
    seen: set[str] = set()

    async with httpx.AsyncClient() as client:
        for i in range(beam_size):
            temperature = 0.0 if i == 0 else 0.7
            top_k = 1 if i == 0 else 20
            out_ids = await _complete(client, prompt_ids, max_new_tokens, temperature, top_k)
            text = decode_ids(out_ids).rstrip("</s>").strip()
            if text and text not in seen:
                seen.add(text)
                results.append(text)

    return results


# ── Output parsing ─────────────────────────────────────────────────────────────

def parse_outputs(raw_outputs: list[str], product_smiles: str) -> list[str]:
    """Convert raw model outputs into reaction SMILES: reactants>>product."""
    reactions = []
    for text in raw_outputs:
        fragments = re.findall(r"<F\d+>(.*?)(?=<F|$)", text)
        reactants = ".".join(f for f in fragments if f)
        if reactants:
            reactions.append(f"{reactants}>>{product_smiles}")
    return reactions


# ── FastAPI ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_tokenizer()
    # Verify llama-server is reachable before accepting traffic
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{LLAMA_URL}/health", timeout=10)
        resp.raise_for_status()
    yield


app = FastAPI(title="RSGPT Retrosynthesis", version="3.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    smiles: str
    beam_size: int = 5
    max_new_tokens: int = 80


class PredictResponse(BaseModel):
    smiles: str
    canonical_smiles: str
    reactions: list[str]


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {req.smiles!r}")
    canonical = Chem.MolToSmiles(mol)

    prompt = f"<s><Isyn><O>{canonical}<F1>"
    prompt_ids = encode(prompt)

    try:
        raw_outputs = await beam_search(prompt_ids, req.beam_size, req.max_new_tokens)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"llama-server error: {e}")

    reactions = parse_outputs(raw_outputs, canonical)
    return PredictResponse(smiles=req.smiles, canonical_smiles=canonical, reactions=reactions)


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{LLAMA_URL}/health", timeout=5)
            llama_ok = resp.status_code == 200
    except Exception:
        llama_ok = False
    return {"status": "ok" if llama_ok else "degraded", "llama_server": llama_ok}


@app.get("/")
def root():
    return {"message": "POST /predict with {smiles, beam_size, max_new_tokens}"}
