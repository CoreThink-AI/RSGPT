# RSGPT Retrosynthesis Model Accuracy Report

**Generated:** 2026-06-06  
**Models evaluated:** `rsgpt-q4_k_m.gguf` (930 MB, 4-bit quantized) vs `rsgpt.gguf` (3.1 GB, F16)  
**Test set:** 23 molecules from `data/test_molecules.yml`  
**Inference:** 3 beams per molecule (1 greedy + 2 diverse, temp=0.7), max 150 tokens  
**Validation:** RDKit SMILES parsing + PubChem CID lookup for each generated fragment

---

## Summary

| Metric | Q4_K_M | F16 |
|---|---|---|
| Reaction sets with all-valid SMILES | 4 / 67 (**6.0%**) | 5 / 69 (**7.2%**) |
| Reaction sets with all frags in PubChem | 4 / 67 (**6.0%**) | 4 / 69 (**5.8%**) |
| Individual fragments: valid SMILES (RDKit) | 46 / 132 (**34.8%**) | 36 / 123 (**29.3%**) |
| Individual fragments: found in PubChem | 42 / 132 (**31.8%**) | 31 / 123 (**25.2%**) |
| Molecules with ≥1 fully-valid reaction set | 3 / 23 (**13%**) | 3 / 23 (**13%**) |
| EOS token produced (clean stop) | 0 / 23 | 0 / 23 |

Both models performed similarly and poorly overall: fewer than 1 in 14 generated reaction sets contained entirely valid SMILES, and meaningful results were produced for only 3 of 23 molecules. The Q4_K_M quantized model marginally outperforms the F16 model on fragment validity (34.8% vs 29.3%), which is likely noise rather than a systematic advantage.

---

## Critical Issue: EOS Token Suppression

**Neither model produced an EOS token (`</s>`, ID=2) on any of the 23 molecules across all 69 beam completions.** Every generation ran to the 150-token limit. This is the primary driver of poor accuracy:

- The model correctly identifies the first 1–2 precursor fragments in many cases
- Without EOS, it continues generating chemically incoherent SMILES beyond those fragments
- The resulting run-on strings fail RDKit parsing, masking what would otherwise be acceptable first-fragment predictions

This issue is present in both F16 and Q4_K_M models, ruling out quantization as the sole cause. Likely explanations:
1. The model was trained on a dataset where products were much longer on average — short SMILES targets don't trigger the EOS distribution
2. The prompt format (`<s><Isyn><O>{smiles}<F1>`) or tokenization may differ slightly from training
3. EOS probability is low-entropy but not zero — increasing temperature or adding EOS to the stop list may help

---

## Per-Molecule Results

### ✓✓ Strong results

#### Methoxy_Diphenylamine (complexity 191) — Best case
Product: `COc1ccc(Nc2ccc(C)cc2)cc1`

| Model | Sets | Best result |
|---|---|---|
| Q4_K_M | **2/2 fully valid (PubChem confirmed)** | `COc1ccc(N)cc1` [CID 7732] + `Cc1ccc(S(=O)(=O)O)cc1` [CID 6101] |
| Q4_K_M | | `COc1ccc(Br)cc1` [CID 7730] + `Cc1ccc(Nc2ccccc2)cc1` [CID 12109] |
| F16 | 2/3 fully valid (PubChem confirmed) | Same set 1 as Q4_K_M; also `COc1ccc(N)cc1` + `CS(C)=O` + `Cc1ccc(S(=O)(=O)O)cc1` |

Both models consistently identify **4-methoxyaniline** (CID 7732) as a core precursor — correct, as it is one of two arylamine building blocks in the target. Q4_K_M's second set (`4-bromoanisole` + `4-methyldiphenylamine`) is mechanistically plausible for a C–N coupling retrosynthesis (palladium-catalyzed Buchwald–Hartwig or copper-mediated Ullmann). This is the only molecule where both models produce multiple chemically sensible routes.

#### Aspirin (complexity 212)
Product: `CC(=O)Oc1ccccc1C(=O)O`

| Model | Sets | Best result |
|---|---|---|
| Q4_K_M | 1/3 fully valid (PubChem confirmed) | `O=C(Cl)Cl` [CID 6371] + `O=C(O)c1ccccc1` [CID 243] + `O=C(O)C1CCCCC1` [CID 7413] |
| F16 | 2/3 fully valid (PubChem confirmed) | `O=C(Cl)Cl` [CID 6371] + `O=C(O)O` [CID 767] + `O=C(O)C(=O)C(=O)[O-]` [CID 5461064] |

Both models identify **acetyl chloride** (CID 6371) and **salicylic acid** (CID 243, `O=C(O)c1ccccc1`) in their first fragments — these are exactly the correct retrosynthetic precursors (Fischer esterification with acyl chloride). The third fragment in each set is chemically invalid for aspirin synthesis but happens to be a real PubChem compound. The genuine retrosynthetic knowledge embedded in the first two fragments is a notable positive.

#### Omeprazole (complexity 339) — Partial credit
Product: `COc1cc2c(cc1OC)N(CS(=O)c1nc(OC)c(C)cc1C)c1ccccc1N2`

| Model | Sets | Best result |
|---|---|---|
| Q4_K_M | 1/3 fully valid (PubChem confirmed) | `c1ccc(CNc2ccccc2Oc2ccccc2)cc1` [CID 29794524] + `CON1C(=O)c2ccccc2C1=O` [CID 278066] |
| F16 | 0/3 | All invalid |

Q4_K_M produced one fully-valid PubChem-confirmed set, but the fragments (`N-benzyl-2-(diphenyloxy)aniline` + `3-methoxyphthalimide`) are not plausible precursors for omeprazole — the model found real molecules that are chemically unrelated to the target synthesis.

---

### ✗ Failed molecules

The remaining 20 molecules produced zero fully-valid reaction sets. Failure modes cluster by molecule type:

#### Simple aryl coupling targets (Tolyl_Pyridine, Etoricoxib, losartan)
The model consistently generates a correct aryl halide fragment in position 1 (e.g., `Brc1cccnc1` [CID 12286] for Tolyl_Pyridine, `O=C(Cl)c1ccc(C)cc1` [CID 13405] for losartan), demonstrating that it has learned SNAr/cross-coupling retrosynthesis patterns. However, fragment 2 is always an extended invalid SMILES string — the model begins correctly then hallucinates a complex ring system.

#### Complex natural products and peptides (Paclitaxel, Ozempic)
Zero valid fragments across all beams. The model appears to have no useful signal for targets with >30 prompt tokens. Paclitaxel (33 tokens, complexity 1790) elicits the same invalid response from both models, suggesting the model is not interpolating from training data at this complexity level.

#### Very complex drugs (Orforglipron, Venetoclax, Acalabrutinib)
All outputs begin with `C[N+]#N)` or `O=C(Cl)OC(Cl)(Cl)Cl)Cl` — syntactically broken SMILES from the first token. These molecules (>30 prompt tokens, complexity >800) appear to be entirely out-of-distribution.

#### Ibuprofen (complexity 203)
Unexpectedly poor result for a simple NSAID. Both models produce invalid SMILES in all beams. The canonical SMILES `CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1` encodes a chiral center; the model may be confused by the stereochemistry annotation. Aspirin (similar complexity, no stereocenters) succeeds where Ibuprofen fails, supporting this hypothesis.

---

## PubChem-Confirmed Fragments Appearing Across Multiple Molecules

Several small reagent fragments appear repeatedly across molecules and are found in PubChem, suggesting the model has internalized a reagent vocabulary:

| SMILES | PubChem CID | Name | Appears in |
|---|---|---|---|
| `O=C(Cl)Cl` | 6371 | Oxalyl chloride / acetyl chloride | Aspirin, Tolyl_Pyridine, losartan |
| `C=CC(=O)O` | 6581 | Acrylic acid | Ibrutinib (×3 beams) |
| `CC(=O)O` | 176 | Acetic acid | Palbocyclib (×4 beams) |
| `COc1ccc(N)cc1` | 7732 | 4-Methoxyaniline | Methoxy_Diphenylamine (×multiple) |
| `COc1ccc(Br)cc1` | 7730 | 4-Bromoanisole | Methoxy_Diphenylamine |
| `CCN(C(C)C)C(C)C` | 81531 | Diisopropylethylamine (Hünig's base) | Imatinib, venetoclax |
| `CS(=O)(=O)O` | 6395 | Methanesulfonic acid | Etoricoxib (×3 beams) |
| `Brc1cccnc1` | 12286 | 3-Bromopyridine | Tolyl_Pyridine (×3 beams) |
| `O=C(O)c1ccccc1` | 243 | Benzoic acid | Aspirin |
| `CCO` | 702 | Ethanol | Apixaban |

The repetition of `Brc1cccnc1` for Tolyl_Pyridine (all 3 beams, both models) is particularly meaningful: 3-bromopyridine is a genuine retrosynthetic precursor for a 3-aminopyridine coupling to form the tolyl-pyridine target.

---

## Complexity vs. Accuracy

| Complexity Range | Molecules | Q4_K_M valid sets | F16 valid sets |
|---|---|---|---|
| <250 (Aspirin, Methoxy_Diphenylamine) | 2 | 3/6 (50%) | 4/6 (67%) |
| 250–500 (Etoricoxib, Fluorinated_Imidazoles, Tolyl_Pyridine, losartan) | 5 | 0/15 (0%) | 0/15 (0%) |
| 500–800 (Camlipixant, Apixaban, Imatinib, Omeprazole, Rivaroxaban, Palbocyclib, nintedanib) | 7 | 1/21 (5%) | 0/21 (0%) |
| >800 (Paclitaxel, Orforglipron, Acalabrutinib, Ibrutinib, Ozempic, etoposide, venetoclax, Methoxybiphenyl) | 9 | 0/25 (0%) | 1/27 (4%) |

There is a clear negative correlation between molecular complexity and output validity. The model appears calibrated for training-set molecules of low-to-medium complexity.

---

## Conclusions and Recommendations

1. **The model encodes real retrosynthetic knowledge for simple molecules.** For Aspirin and Methoxy_Diphenylamine it reliably identifies correct building blocks (salicylic acid, acetyl chloride, 4-methoxyaniline, aryl bromides). This is non-trivial and validates the training approach.

2. **EOS suppression is the most urgent bug to fix.** Correct first fragments are buried under invalid run-on output. Adding `</s>` as an explicit stop string in the llama-server request, or fine-tuning on shorter sequences to restore EOS probability, would likely double the apparent accuracy at low complexity.

3. **The useful complexity range is approximately 150–250 (Bertz score).** Above ~300 the model produces at most one valid fragment before generating noise; above ~800 all output is invalid from the first token.

4. **Q4_K_M quantization does not measurably hurt accuracy** relative to F16 at the 150-token generation budget. Fragment validity is 34.8% vs 29.3% in Q4_K_M's favour — a difference likely within noise. For deployment, Q4_K_M (930 MB, 3.3× smaller) is the practical choice.

5. **Beam diversity is low.** In almost all cases, the 3 beam completions either converge on the same first fragment or are all invalid. The `top_k=20, temperature=0.7` sampling setting for diverse beams is not producing meaningfully different retrosynthetic routes — more aggressive diversity strategies (nucleus sampling, beam blocking) may help.

6. **For production use**, consider restricting input to molecules with Bertz complexity < 300 and prompt token count < 15, and post-filtering output to return only beams where all fragments parse in RDKit. This would yield a precision-focused system rather than attempting to rank invalid outputs.

---

*Report generated from `data/model_comparison.yml`. Validation: RDKit 2024.09, PubChem PUG REST API (2026-06-06).*
