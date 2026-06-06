"""
Export finetune_full.pth → HuggingFace-compatible directory for GGUF conversion.

Output layout:
  rsgpt_hf/
    config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    model.safetensors
"""
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

CHECKPOINT = "models/finetune_full.pth"
OUT_DIR = Path("rsgpt_hf")
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Load checkpoint ────────────────────────────────────────────────────────
print("Loading checkpoint …")
raw = torch.load(CHECKPOINT, map_location="cpu")

# Strip DDP prefix
raw = {(k[7:] if k.startswith("module.") else k): v for k, v in raw.items()}

# ── 2. Extract model.* weights (the LlamaForCausalLM inside RxnGPT) ──────────
# Keys look like:  model.model.layers.0.self_attn.q_proj.weight
#                  model.lm_head.weight
# Strip the leading "model." so HF sees:
#   model.layers.0.self_attn.q_proj.weight  (body)
#   lm_head.weight
print("Extracting model weights …")
hf = {}
skipped = []
for k, v in raw.items():
    if k.startswith("model."):
        new_key = k[6:]          # drop "model."
        hf[new_key] = v.to(torch.float16)
    else:
        skipped.append(k)

print(f"  Kept   : {len(hf)} tensors")
print(f"  Skipped: {len(skipped)} tensors (outer LlamaModel, not needed)")

# Sanity-check expected keys
assert "lm_head.weight" in hf, "lm_head.weight missing"
assert any("layers.0" in k for k in hf), "layer 0 missing"

# ── 3. Save safetensors ───────────────────────────────────────────────────────
out_weights = OUT_DIR / "model.safetensors"
print(f"Saving {out_weights} …")
save_file(hf, str(out_weights))
print(f"  {out_weights.stat().st_size / 1e9:.2f} GB")

# ── 4. config.json ────────────────────────────────────────────────────────────
config = {
    "_name_or_path": "rsgpt",
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 24,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-5,
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "use_cache": True,
    "vocab_size": 1000,
    "transformers_version": "4.31.0",
}
(OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))
print("Wrote config.json")

# ── 5. tokenizer.json (the correct 1000-token BPE from vocab.json) ────────────
shutil.copy("vocab.json", OUT_DIR / "tokenizer.json")
print("Copied vocab.json → tokenizer.json")

# ── 6. tokenizer_config.json ──────────────────────────────────────────────────
tok_config = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "model_max_length": 512,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "clean_up_tokenization_spaces": True,
}
(OUT_DIR / "tokenizer_config.json").write_text(json.dumps(tok_config, indent=2))
print("Wrote tokenizer_config.json")

# ── 7. special_tokens_map.json ────────────────────────────────────────────────
special_tokens = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
}
(OUT_DIR / "special_tokens_map.json").write_text(json.dumps(special_tokens, indent=2))
print("Wrote special_tokens_map.json")

# ── 8. generation_config.json (optional but helpful) ─────────────────────────
gen_config = {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
}
(OUT_DIR / "generation_config.json").write_text(json.dumps(gen_config, indent=2))
print("Wrote generation_config.json")

print(f"\nDone. HF model exported to: {OUT_DIR.resolve()}/")
print("Files:")
for f in sorted(OUT_DIR.iterdir()):
    size = f.stat().st_size
    print(f"  {f.name:40s} {size/1e6:8.1f} MB")
