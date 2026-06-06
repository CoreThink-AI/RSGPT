import json
import re
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem
from llama_cpp import Llama

# ── Paths ─────────────────────────────────────────────────────────────────────
GGUF_PATH  = os.getenv("GGUF_PATH",  "rsgpt.gguf")
VOCAB_PATH = os.getenv("VOCAB_PATH", "vocab.json")
N_THREADS  = int(os.getenv("N_THREADS", os.cpu_count() or 4))
N_CTX      = 128   # max context length; SMILES prompts are short

# ── Globals set at startup ────────────────────────────────────────────────────
llm:    Optional[Llama] = None
id2tok: dict[int, str]  = {}
tok2id: dict[str, int]  = {}
EOS_ID = 2   # </s>

# ── Tokenizer helpers ─────────────────────────────────────────────────────────

def load_vocab(path: str):
    global id2tok, tok2id
    data = json.loads(Path(path).read_text())
    vocab = data["model"]["vocab"]
    tok2id.update(vocab)
    id2tok.update({v: k for k, v in vocab.items()})
    # also index added_tokens so special tokens round-trip cleanly
    for entry in data.get("added_tokens", []):
        tok2id[entry["content"]] = entry["id"]
        id2tok[entry["id"]] = entry["content"]


def encode(text: str) -> list[int]:
    """Tokenize a pre-formatted prompt string using the BPE vocab.
    Falls back to character-level for unknown tokens."""
    from tokenizers import Tokenizer as HFTokenizer
    _tok = HFTokenizer.from_file(VOCAB_PATH)
    return _tok.encode(text, add_special_tokens=False).ids


def decode_ids(ids: list[int]) -> str:
    return "".join(id2tok.get(i, "") for i in ids)

# ── Beam search over llama-cpp eval/scores ────────────────────────────────────

def beam_search(prompt_ids: list[int], beam_size: int, max_new: int) -> list[str]:
    """
    Returns up to beam_size decoded strings (prompt stripped).
    Uses llama-cpp-python's low-level eval() + scores to implement beam search.
    Each beam re-evaluates from scratch (no KV-cache sharing), which is fine
    for the short SMILES sequences this model generates.
    """
    from copy import deepcopy

    # Each beam: (token_ids, cumulative_log_prob, finished)
    beams: list[tuple[list[int], float, bool]] = [(list(prompt_ids), 0.0, False)]
    completed: list[tuple[list[int], float]] = []

    for _ in range(max_new):
        active = [(ids, sc) for ids, sc, done in beams if not done]
        if not active:
            break

        candidates: list[tuple[list[int], float]] = []
        for ids, score in active:
            # Evaluate full sequence (llama-cpp resets KV cache each call)
            llm.reset()
            llm.eval(ids)

            # Raw logits for the last position → log-softmax
            logits = np.array(llm.scores[len(ids) - 1], dtype=np.float64)
            logits -= logits.max()
            log_probs = logits - np.log(np.exp(logits).sum())

            top_ids = np.argsort(log_probs)[::-1][: beam_size + 5]
            for tid in top_ids:
                candidates.append((ids + [int(tid)], score + log_probs[tid]))

        # Keep best beam_size, separate finished from active
        candidates.sort(key=lambda x: -x[1])
        beams = []
        for ids, sc in candidates[: beam_size * 2]:
            if ids[-1] == EOS_ID:
                completed.append((ids, sc))
            else:
                if len(beams) < beam_size:
                    beams.append((ids, sc, False))

        if len(completed) >= beam_size:
            break

    # Collect any unfinished beams too
    for ids, sc, _ in beams:
        completed.append((ids, sc))

    completed.sort(key=lambda x: -x[1])

    # Decode and strip prompt prefix
    prompt_len = len(prompt_ids)
    seen: set[str] = set()
    results: list[str] = []
    for ids, _ in completed[:beam_size]:
        text = decode_ids(ids[prompt_len:]).rstrip("</s>")
        if text and text not in seen:
            seen.add(text)
            results.append(text)

    return results

# ── Output parsing (mirrors infer.py jiexi) ───────────────────────────────────

def parse_output(raw_outputs: list[str], product_smiles: str) -> list[str]:
    """Convert raw model outputs into reaction SMILES: reactants>>product."""
    results = []
    for text in raw_outputs:
        f_matches = re.findall(r"<F\d+>(.*?)(?=<F|$)", text)
        if not f_matches:
            continue
        reactants = ".".join(m for m in f_matches if m)
        if reactants:
            results.append(f"{reactants}>>{product_smiles}")
    return results

# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    load_vocab(VOCAB_PATH)
    llm = Llama(
        model_path=GGUF_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=0,   # CPU only; set >0 if ROCm/CUDA available
        verbose=False,
        logits_all=True,  # required to read scores at every position
    )
    yield
    del llm


app = FastAPI(title="RSGPT Retrosynthesis (llama.cpp)", version="2.0")


class PredictRequest(BaseModel):
    smiles: str
    beam_size: int = 5
    max_new_tokens: int = 80


class PredictResponse(BaseModel):
    smiles: str
    canonical_smiles: str
    reactions: list[str]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Validate and canonicalize SMILES
    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {req.smiles!r}")
    canonical = Chem.MolToSmiles(mol)

    # Build prompt: <s><Isyn><O>{smiles}<F1>
    prompt = f"<s><Isyn><O>{canonical}<F1>"
    try:
        prompt_ids = encode(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenization error: {e}")

    # Run beam search
    try:
        raw_outputs = beam_search(prompt_ids, req.beam_size, req.max_new_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    reactions = parse_output(raw_outputs, canonical)
    return PredictResponse(smiles=req.smiles, canonical_smiles=canonical, reactions=reactions)


@app.get("/health")
def health():
    return {"status": "ok", "model": GGUF_PATH}


@app.get("/")
def root():
    return {"message": "RSGPT retrosynthesis API. POST /predict with {smiles, beam_size}"}
