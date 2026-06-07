"""
Run retrosynthesis inference on all test_molecules.yml entries.
Saves results to data/model_comparison.yml.
"""
import json
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

import yaml
from tokenizers import Tokenizer

VOCAB_PATH   = "vocab.json"
MODELS = {
    "q4_k_m": {"port": 8181, "file": "rsgpt-q4_k_m.gguf"},
}
MAX_TOKENS   = 150
BEAM_SIZE    = 10  # 1 greedy + 9 diverse
TIMEOUT      = 120 # seconds per request

tok = Tokenizer.from_file(VOCAB_PATH)
vocab_data = json.loads(Path(VOCAB_PATH).read_text())
vocab      = vocab_data["model"]["vocab"]
id2tok     = {v: k for k, v in vocab.items()}
for entry in vocab_data.get("added_tokens", []):
    id2tok[entry["id"]] = entry["content"]


def encode(text: str) -> list:
    return tok.encode(text, add_special_tokens=False).ids


def decode_ids(ids: list) -> str:
    return "".join(id2tok.get(i, "") for i in ids)


def wait_ready(port: int, timeout: int = 120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def complete(port: int, prompt_ids: list, temperature: float = 0.0, top_k: int = 1) -> dict:
    payload = json.dumps({
        "prompt":      prompt_ids,
        "n_predict":   MAX_TOKENS,
        "temperature": temperature,
        "top_k":       top_k,
        "stop":        [],
        "stream":      False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read())
    stop_type = data.get("stop_type", "")
    return {
        "content":          data.get("content", ""),
        "tokens_predicted": data.get("tokens_predicted", 0),
        "stopped_eos":      stop_type == "eos",
        "stopped_limit":    stop_type == "limit",
        "stop_type":        stop_type,
    }


def parse_fragments(raw: str) -> list:
    """Split on <Fn> tags, strip EOS, return non-empty fragments."""
    parts = re.split(r"<F\d+>", raw)
    return [p.rstrip("</s>").strip() for p in parts if p.strip().rstrip("</s>").strip()]


def run_beam(port: int, prompt_ids: list) -> dict:
    results   = []
    reactions = []
    seen      = set()
    for i in range(BEAM_SIZE):
        temperature = 0.0 if i == 0 else 0.7
        top_k       = 1   if i == 0 else 20
        try:
            r = complete(port, prompt_ids, temperature, top_k)
        except Exception as e:
            results.append({"error": str(e), "beam": i})
            continue
        frags = parse_fragments(r["content"])
        rxn   = ".".join(frags) if frags else ""
        results.append({
            "beam":             i,
            "raw":              r["content"],
            "fragments":        frags,
            "tokens_predicted": r["tokens_predicted"],
            "stopped_eos":      r["stopped_eos"],
            "stopped_limit":    r["stopped_limit"],
            "stop_type":        r["stop_type"],
        })
        if rxn and rxn not in seen:
            seen.add(rxn)
            reactions.append(rxn)
    return {"beams": results, "unique_precursor_sets": reactions}


def main():
    # Load test molecules
    with open("data/test_molecules.yml") as f:
        data = yaml.safe_load(f)

    molecules = [m for m in data["molecules"] if m.get("canonical_smiles")]
    print(f"Loaded {len(molecules)} molecules with valid canonical SMILES")

    # Wait for both servers
    for name, cfg in MODELS.items():
        print(f"Waiting for {name} server on port {cfg['port']} ...", flush=True)
        if not wait_ready(cfg["port"]):
            print(f"ERROR: {name} server not ready, aborting")
            sys.exit(1)
        print(f"  {name} ready")

    results = []
    total   = len(molecules)
    for idx, mol in enumerate(molecules, 1):
        name   = mol["query_name"]
        smiles = mol["canonical_smiles"]
        prompt = f"<s><Isyn><O>{smiles}<F1>"
        ids    = encode(prompt)
        print(f"\n[{idx}/{total}] {name}  ({len(ids)} prompt tokens)", flush=True)

        entry = {
            "query_name":      name,
            "pubchem_cid":     mol.get("pubchem_cid"),
            "canonical_smiles": smiles,
            "prompt_tokens":   len(ids),
            "models":          {},
        }

        for model_name, cfg in MODELS.items():
            print(f"  {model_name} ...", end=" ", flush=True)
            t0  = time.time()
            out = run_beam(cfg["port"], ids)
            elapsed = round(time.time() - t0, 1)
            out["elapsed_s"] = elapsed
            out["model_file"] = cfg["file"]
            entry["models"][model_name] = out
            # Show first reaction if any
            first = out["unique_precursor_sets"][0] if out["unique_precursor_sets"] else "(none)"
            print(f"{elapsed}s  →  {first[:80]}", flush=True)

        results.append(entry)

    output = {
        "metadata": {
            "generated":  datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "beam_size":  BEAM_SIZE,
            "max_tokens": MAX_TOKENS,
            "models":     {k: v["file"] for k, v in MODELS.items()},
        },
        "results": results,
    }

    out_path = Path("data/model_comparison_q4_10beams.yml")
    with open(out_path, "w") as f:
        yaml.dump(output, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"\nSaved {len(results)} entries to {out_path}")


if __name__ == "__main__":
    main()
