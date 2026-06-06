"""
Write rsgpt.gguf directly using the gguf Python library.
Bypasses convert_hf_to_gguf.py entirely to avoid tokenizer compatibility issues.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import gguf
from safetensors import safe_open

MODEL_DIR = Path("rsgpt_hf")
OUT_FILE  = "rsgpt.gguf"

# ── Load config ───────────────────────────────────────────────────────────────
cfg = json.loads((MODEL_DIR / "config.json").read_text())
n_layers    = cfg["num_hidden_layers"]       # 24
n_heads     = cfg["num_attention_heads"]     # 32
n_kv_heads  = cfg["num_key_value_heads"]     # 32
hidden      = cfg["hidden_size"]             # 2048
ffn         = cfg["intermediate_size"]       # 8192
vocab_size  = cfg["vocab_size"]              # 1000
ctx_len     = cfg["max_position_embeddings"] # 512
rms_eps     = cfg["rms_norm_eps"]            # 1e-5

# ── Load tokenizer vocab ──────────────────────────────────────────────────────
tok = json.loads((MODEL_DIR / "tokenizer.json").read_text())
bpe_vocab   = tok["model"]["vocab"]          # token -> id
bpe_merges  = tok["model"]["merges"]         # list of "A B" strings
added_toks  = {x["content"]: x["id"] for x in tok.get("added_tokens", [])}

# Build id -> token list (size = vocab_size)
id2tok = [""] * vocab_size
for token, tid in bpe_vocab.items():
    if tid < vocab_size:
        id2tok[tid] = token
# Fill any gaps with <unk>
for i, t in enumerate(id2tok):
    if t == "":
        id2tok[i] = f"<unused{i}>"

scores = [0.0] * vocab_size  # BPE doesn't use scores; set to 0
toktypes = [gguf.TokenType.NORMAL] * vocab_size
# Mark specials
for name, tid in [("<pad>", 0), ("<s>", 1), ("</s>", 2), ("<unk>", 3)]:
    if tid < vocab_size:
        toktypes[tid] = gguf.TokenType.CONTROL

# ── Create writer ─────────────────────────────────────────────────────────────
print(f"Creating {OUT_FILE} …")
writer = gguf.GGUFWriter(OUT_FILE, arch="llama")

# ── Model metadata ────────────────────────────────────────────────────────────
writer.add_name("RSGPT")
writer.add_description("RSGPT retrosynthesis model (Llama-2 architecture, 1.6B)")
writer.add_context_length(ctx_len)
writer.add_embedding_length(hidden)
writer.add_block_count(n_layers)
writer.add_feed_forward_length(ffn)
writer.add_head_count(n_heads)
writer.add_head_count_kv(n_kv_heads)
writer.add_rope_dimension_count(hidden // n_heads)  # 64
writer.add_rope_freq_base(10000.0)
writer.add_layer_norm_rms_eps(rms_eps)
writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

# ── Tokenizer metadata ────────────────────────────────────────────────────────
writer.add_tokenizer_model("gpt2")
writer.add_tokenizer_pre("gpt-2")
writer.add_token_list(id2tok)
writer.add_token_scores(scores)
writer.add_token_types(toktypes)
writer.add_bos_token_id(1)
writer.add_eos_token_id(2)
writer.add_pad_token_id(0)
writer.add_unk_token_id(3)
writer.add_add_bos_token(False)  # prompt already contains <s>
writer.add_add_eos_token(False)

# Add BPE merges
merges_flat = [m.replace(" ", "") for m in bpe_merges]  # "A B" -> "AB"
# gguf expects merges as list of strings in "A B" format
writer.add_token_merges(bpe_merges)

# ── Helper: tensor name mapping (HF -> GGUF/llama.cpp convention) ─────────────
def hf_to_gguf_name(hf_key: str):
    k = hf_key
    # lm_head
    if k == "lm_head.weight":               return "output.weight"
    # final norm
    if k == "model.norm.weight":            return "output_norm.weight"
    # embed
    if k == "model.embed_tokens.weight":    return "token_embd.weight"
    # per-block
    import re
    m = re.match(r"model\.layers\.(\d+)\.(.*)", k)
    if not m:
        return None
    i, rest = m.group(1), m.group(2)
    mapping = {
        "input_layernorm.weight":        f"blk.{i}.attn_norm.weight",
        "post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
        "self_attn.q_proj.weight":       f"blk.{i}.attn_q.weight",
        "self_attn.k_proj.weight":       f"blk.{i}.attn_k.weight",
        "self_attn.v_proj.weight":       f"blk.{i}.attn_v.weight",
        "self_attn.o_proj.weight":       f"blk.{i}.attn_output.weight",
        "mlp.gate_proj.weight":          f"blk.{i}.ffn_gate.weight",
        "mlp.up_proj.weight":            f"blk.{i}.ffn_up.weight",
        "mlp.down_proj.weight":          f"blk.{i}.ffn_down.weight",
    }
    return mapping.get(rest)

# ── Stream tensors from safetensors ──────────────────────────────────────────
print("Writing tensors …")
sf_path = str(MODEL_DIR / "model.safetensors")
n_written = 0
n_skipped = 0

with safe_open(sf_path, framework="numpy") as f:
    for hf_key in sorted(f.keys()):
        gguf_name = hf_to_gguf_name(hf_key)
        if gguf_name is None:
            print(f"  skip: {hf_key}")
            n_skipped += 1
            continue

        tensor = f.get_tensor(hf_key)  # numpy array, float16

        # Norm layers → float32 (required by llama.cpp)
        if "norm" in gguf_name:
            tensor = tensor.astype(np.float32)
            dtype = gguf.GGMLQuantizationType.F32
        else:
            tensor = tensor.astype(np.float16)
            dtype = gguf.GGMLQuantizationType.F16

        writer.add_tensor(gguf_name, tensor, raw_dtype=dtype)
        n_written += 1
        if n_written % 50 == 0:
            print(f"  {n_written} tensors written …")

print(f"  Done: {n_written} tensors written, {n_skipped} skipped")

# ── Finalise ──────────────────────────────────────────────────────────────────
print("Writing header and finalizing …")
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

size_gb = Path(OUT_FILE).stat().st_size / 1e9
print(f"\nDone → {OUT_FILE}  ({size_gb:.2f} GB)")
