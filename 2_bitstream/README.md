[← Back to the overview](../README.md) · [Part 1: Strings](../1_strings/README.md) · Part 2 of 3 · [Part 3: Audio tokeniser](../3_audio/README.md)

# 2 · Bitstream: can a diffusion model read a spectrum written as fixed-width bit fields?

The same task as the strings (predict the molecule's SMILES from its IR and Raman spectra), with the same QM9S molecules, splits and metrics. But instead of text for a pretrained LLM, every field of the `components_v3` string gets a fixed group of bits. A bit-level diffusion model, built on [CoBit](https://github.com/GBATZOLIS/BitstreamDiffusion) and trained from scratch, reads the spectrum bits as a clean prefix and generates the SMILES bits.

![Pipeline: the spectrum is written as fixed-width bit fields; the model denoises the SMILES bits](figures/pipeline.svg)

## Result

Validation set: 500 molecules, each tested with IR, Raman and IR+Raman. The bitstream model is the checkpoint at step 40,000 of 60,000, the best of six checkpoints on validation. The string arms are the Qwen3-8B models from [Part 1](../1_strings/README.md), scored on the same validation prompts. The test set has not been used for the bitstream yet.

![Validity and Tanimoto on IR+Raman for the bitstream and the three string arms](figures/bits_vs_strings.png)

1. **On the benchmark metric, the bits lose.** With invalid outputs scored as 0, IR+Raman Tanimoto is 0.148 for the bits, against 0.230 for SpectraLLM's format and 0.357 for our string with mode labels. The main reason is validity: only 44% of the bitstream outputs are valid SMILES, against 97% to 99% for the strings.
2. **When the output is valid, it is close to the best string.** On valid outputs only, the bits reach 0.335 on IR+Raman, against 0.368 for the string with mode labels, and above both string arms without mode labels (0.232 and 0.243).
3. **On the same molecules, the gap is small.** Where the bits give a valid SMILES, the paired difference to the string with mode labels is −0.038 on IR+Raman (95% interval −0.073 to −0.004). For IR alone and Raman alone, the interval includes zero.
4. **The failures are grammar, not chemistry.** 823 of 1,500 outputs are invalid. 75% have an unclosed ring or unmatched brackets; only 14% are well formed but chemically impossible.
5. **The model can memorise but does not yet generalise.** On 300 of its own training molecules it gives the exact molecule 97.7% of the time; on validation, 2.4%.

The bitstream model has 41M parameters and starts from zero. The string arms fine-tune an 8B model pretrained on a large text corpus. That difference matters most for validity.

### All conditions

| Spectra | Arm | Valid SMILES | Exact match | Tanimoto (invalid = 0) | Tanimoto (valid only) |
|---|---|---|---|---|---|
| IR | **Bitstream, step 40k** | 43.6% | 1.2% | 0.125 | 0.287 |
| IR | String A (SpectraLLM's format) | 99.0% | 1.0% | 0.194 | 0.196 |
| IR | String B, no mode labels | 99.4% | 0.2% | 0.194 | 0.195 |
| IR | String B + mode labels | 98.0% | 5.0% | **0.293** | **0.299** |
| Raman | **Bitstream, step 40k** | 47.6% | 2.0% | 0.139 | 0.292 |
| Raman | String A (SpectraLLM's format) | 99.2% | 0.6% | 0.194 | 0.195 |
| Raman | String B, no mode labels | 99.0% | 1.0% | 0.202 | 0.204 |
| Raman | String B + mode labels | 98.2% | 3.4% | **0.268** | **0.273** |
| IR+Raman | **Bitstream, step 40k** | 44.2% | 2.4% | 0.148 | 0.335 |
| IR+Raman | String A (SpectraLLM's format) | 99.0% | 1.0% | 0.230 | 0.232 |
| IR+Raman | String B, no mode labels | 99.0% | 1.0% | 0.241 | 0.243 |
| IR+Raman | String B + mode labels | 97.0% | 10.2% | **0.357** | **0.368** |

n = 500 per row. A random training molecule scores Tanimoto 0.081. The bitstream and `B_modes_on` both carry mode labels; `A` and `B_modes_off` do not.

### Same molecules: only where the bits give a valid SMILES

Each row keeps the molecules for which the bitstream output is valid, and scores every arm on exactly those molecules. This asks whether the bits fail on hard molecules or fail at random.

| Spectra | Molecules | Bitstream | String A | String B, no labels | String B + labels | Bitstream − (B + labels) | 95% interval |
|---|---|---|---|---|---|---|---|
| IR | 218 | 0.287 | 0.196 | 0.195 | 0.305 | −0.018 | −0.051 to +0.014 |
| Raman | 238 | 0.292 | 0.196 | 0.214 | 0.275 | +0.017 | −0.010 to +0.045 |
| IR+Raman | 221 | 0.335 | 0.223 | 0.244 | 0.374 | −0.038 | −0.073 to −0.004 |

**What it shows.** The string arms score only slightly higher on this subset than on all molecules (by 0.007 to 0.017 for the string with labels), so the valid bitstream outputs are not just the easy molecules. On these molecules the bits are within 0.04 of the best string, and on Raman they are slightly ahead (not significant). Paired bootstrap, 10,000 resamples.

### Why the outputs are invalid

| Reason | Share of invalid outputs | Example output | True molecule |
|---|---|---|---|
| Parentheses or brackets do not match | 30.1% | `C#CC(CC#CC)CN)=O` | `C#CCC(N)=O` |
| Ring is opened but not closed | 44.8% | `CC1(O)CCCC2CCC1` | `CC1(O)CCCC1` |
| Well formed, but chemically impossible (e.g. valence) | 14.3% | `C#CC(C#CCN)(N)=O` | `C#CCC(N)=O` |
| Other syntax (e.g. a symbol in the wrong place) | 10.7% | `CC1CC(=O)CCC1?O` | `CC1CC(=O)CC1=O` |

823 of 1,500 outputs at step 40k. The checks are applied in the order of the table. The SMILES grammar needs matching pairs far apart in the string. A model that sets all bits in parallel finds this hard; the string LLM, which writes one symbol at a time, rarely gets it wrong (97% to 99% valid).

### Choosing the checkpoint

| Checkpoint | Valid SMILES | Exact match | Tanimoto (invalid = 0) | Tanimoto (valid only) |
|---|---|---|---|---|
| step 10,000 | 0.6% | 0.0% | 0.000 | 0.000 |
| step 20,000 | 0.6% | 0.0% | 0.000 | 0.065 |
| step 30,000 | 33.0% | 2.6% | 0.116 | 0.353 |
| **step 40,000** | **44.2%** | **2.4%** | **0.148** | 0.335 |
| step 50,000 | 40.6% | 1.8% | 0.135 | 0.331 |
| step 60,000 (last) | 35.2% | 1.8% | 0.120 | 0.340 |

IR+Raman, validation set. Step 40,000 has the highest score on the benchmark metric, so it is the one reported above.

### Memorisation check

| Run | Molecules | Valid SMILES | Exact match | Tanimoto |
|---|---|---|---|---|
| Full run (10,000 training molecules, last checkpoint), 300 of its training molecules | 300 | 98.7% | 97.7% | 0.982 |
| Small run (1,000 training molecules, trained to memorise), 300 of its training molecules | 300 | 93.7% | 90.0% | 0.917 |

The model can learn the mapping from spectrum bits to SMILES bits, and it writes valid SMILES for molecules it has seen. On new molecules it does not yet generalise. With 10,000 molecules and no pretraining, it overfits.

## Design

- **Data.** QM9S, the same molecules and splits as the strings: 10,000 training molecules (each with IR, Raman and IR+Raman, so 30,000 examples), 500 validation molecules. Mode labels come from the same DFT Hessians and ORCA PED analysis.
- **Bit fields.** Every field of the `components_v3` string with mode labels gets a fixed number of bits (see the schematic). Up to 35 IR peaks and 17 Raman peaks; empty slots are zero. A mode label keeps up to 3 components.
- **Bit budget.** 5,570 bits carry data: headers 128 (2%), peak numbers 2,392 (43%), mode labels 2,912 (52%), SMILES 138 (3%). They are packed into 627 tokens of 12 bits: 604 tokens for the spectrum, 23 for the SMILES. The strings need about 1,500 tokens for the same content.
- **Model.** CoBit's transformer: 8 blocks, width 512, 8 heads, one 12-bit token per position, self-conditioning. 41M parameters, trained from scratch.
- **Training.** Noise is added to the SMILES bits only (σ from 0.002 to 80); the spectrum stays clean. Loss: binary cross-entropy on the SMILES bits, with EDM weighting. 60,000 steps, batch 256, AdamW, learning rate 3e-4 with cosine decay.
- **Sampling.** DDIM, 128 steps from σ = 80 to σ = 0.08 on an entropy-based schedule, EMA weights, float32, no guidance, one sample per molecule. Each bit is then set to 1 where P > 0.5, and the 23 symbols are decoded to a SMILES.
- **Metrics.** The definitions of Part 1: Morgan radius 2, 2,048 bits, binary Tanimoto; targets are stereo-free canonical SMILES.

## Notes

- **Validation only.** The test set is kept for the final comparison, once the sampler and the representation are settled.
- **One training run.** The intervals cover variation between molecules, not between runs.
- **Float32 sampling.** Sampling with bfloat16 gave wrong results for the same checkpoint (12% exact match on training molecules, against 97.7% in float32). All results here use float32.
- **The first full run failed.** It scored near zero. The causes, fixed before this run: the input scaling made the clean spectrum prefix almost invisible at high noise, the model had no absolute position for the fixed slot layout, and training and sampling used different values of σ_data.

## Next

- Tune the sampler (temperature, guidance, number of steps) before changing the model.
- Teach the model SMILES first (pretraining on SMILES), or generate a representation that is always valid, such as SELFIES.
- Train on more molecules (50,000) and see how validity and accuracy scale.
- Extend the bit layout to EC-SERS, where each sample is a sequence of spectra over time.

## Files in this folder

Results and outputs only; no code.

| File | What it holds |
|---|---|
| `results/bits_vs_strings_val.csv` | The table "All conditions" |
| `results/same_molecules_where_bits_valid.csv` | The same-molecules table, with paired gaps on all molecules and on the valid subset |
| `results/invalid_reasons.csv` | Counts behind "Why the outputs are invalid" |
| `results/checkpoint_sweep_IR+Raman.csv` | The checkpoint table |
| `results/memorisation_summary.csv` | The memorisation table |
| `results/scores_val_vs_strings/` | Full metrics (all seven, per molecule and summary) for the bits and the two B strings on validation |
| `results/scores_val_sweep/` | Metrics for every checkpoint |
| `results/predictions/val__sweep_step40000.jsonl` | Every generated SMILES at step 40k, with its sampling manifest |
| `results/memorisation/` | Generations on training molecules for the two runs above |
| `results/run/` | Run configuration, the bit schema and the cache build report |
| `figures/` | The figures on this page |
