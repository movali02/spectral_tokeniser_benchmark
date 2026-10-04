# spectral tokenisers

**Does a richer text format help a language model read a spectrum?**
A controlled comparison of two ways of writing IR and Raman spectra as text, for predicting molecular structure with a fine-tuned LLM, on the QM9S benchmark used by [SpectraLLM](https://arxiv.org/abs/2508.08441).

## Result

Four models were trained that differ only in how the spectrum is written as text. Scores are mean ECFP4 Tanimoto similarity between the predicted and the true molecule on 1,000 held-out molecules (0.081 for a random training molecule, 1.0 for a perfect prediction).

![Mean ECFP4 Tanimoto for four arms across IR, Raman and IR+Raman](assets/tanimoto_by_arm.png)

1. **The format alone slightly helps.** Our structured format without mode labels beats SpectraLLM's layout by +0.012 on IR+Raman (95% interval +0.0004 to +0.023). Exact match is unchanged at 1.8%.
2. **Vibrational mode labels help a lot.** Adding a mode label to each peak gives +0.102 (nearly 50% increase) and raises exact match from 1.8% to 8.1% (nearly 7x). About 90% of the total gain comes from the labels.
3. **The labels carry information a spectrum alone does not.** They are derived from each molecule's DFT Hessian. So result 2 measures what correct peak assignments are worth to the model, not how much more the model extracts from the spectrum, rather how priors can help generate more useful representations.
4. **Reading the IR axis in reverse costs nothing.** SpectraLLM's released preprocessing assigns the IR axis in the opposite direction to the QM9S file. Reproducing that exactly changes the score by +0.002 (95% interval −0.009 to +0.012). A fine-tuned model learns whatever consistent mapping it is given so all good.

### Decomposition of the gain (IR+Raman, primary condition)

| Step | Comparison | Gain | 95% interval |
|---|---|---|---|
| Change the format only | `B_modes_off` − `A` | +0.012 | +0.0004 to +0.023 |
| Add mode labels | `B_modes_on` − `B_modes_off` | +0.102 | +0.087 to +0.118 |
| Total | `B_modes_on` − `A` | +0.114 | +0.098 to +0.130 |

### Full string against SpectraLLM's format

| Spectra | `A` | `B_modes_on` | Difference | 95% interval |
|---|---|---|---|---|
| IR | 0.194 | 0.293 | +0.099 | +0.086 to +0.112 |
| Raman | 0.193 | 0.264 | +0.071 | +0.060 to +0.082 |
| IR+Raman (primary) | 0.235 | 0.349 | +0.114 | +0.098 to +0.130 |

| Metric (IR+Raman) | `A` | `B_modes_on` |
|---|---|---|
| Exact match | 1.8% | 8.1% |
| Tanimoto ≥ 0.4 | 7.4% | 25.2% |
| Functional-group overlap | 0.811 | 0.893 |
| MACCS Tanimoto | 0.523 | 0.686 |
| Cosine | 0.364 | 0.484 |
| Fraggle | 0.583 | 0.657 |
| MCES distance (lower is better) | 5.71 | 3.92 |
| Validity | 99.1% | 97.3% |

Validity is the one metric that got worse, by 1.8 percentage points.

## Design

Comparing a small run against SpectraLLM's published numbers would mostly measure scale: they trained a 32B model on about 105,000 molecules. Here every arm uses the same base model, molecules, splits, training settings, batch order, decoding and metrics. Only the text changes.

| Arm | What the model is given |
|---|---|
| `A_native` | SpectraLLM's released pipeline reproduced exactly, including its IR axis |
| `A` | SpectraLLM's text layout on a shared peak list, IR axis as in the QM9S file |
| `B_modes_off` | Our format (`components_v3`) on the same peaks |
| `B_modes_on` | `components_v3` with a vibrational mode label on each peak |

- **Clean test of the format:** `A` against `B_modes_off`.
- **Value of mode labels:** `B_modes_on` against `B_modes_off`.
- **Effect of the released pipeline's axis and peak settings:** `A_native` against `A`.

`components_v3` writes each peak as a structured record and adds derived features: peak counts in 250 cm⁻¹ windows, a relative intensity band per peak, and the start and end of each peak. It has 23 fields, of which 18 are not in SpectraLLM's format.

### Data

- **Source.** [QM9S](https://www.nature.com/articles/s43588-023-00550-y): 129,817 small organic molecules with DFT-computed IR and Raman spectra. 122,633 pass checks on chemistry, geometry and duplicates.
- **Splits.** 1,000 test, 500 validation and 10,000 training molecules, stratified by heavy-atom count and 17 functional groups. Split by molecule.
- **Aligned with SpectraLLM's split.** Test and validation molecules come from their test set and training molecules from their training set.
- **Targets.** Stereo-free canonical SMILES, matching SpectraLLM's convention.
- **Prompts.** 188,400 across 4 arms, 3 conditions (IR, Raman, IR+Raman) and all splits.

### Mode labels without new DFT

QM9S ships the Hessian for every molecule. Each Hessian is written as an ORCA Hessian file and passed to the ORCA PED Analyzer, and each mode is matched to a peak.

- The analyzer succeeded for 11,349 of 11,700 molecules.
- 97% of peaks received a label (160,330 IR peaks, 92,728 Raman peaks).
- Its frequencies agree with an independent diagonalisation to 0.000 cm⁻¹ over 552,773 modes.
- Spectra rebuilt from the DFT modes reproduce the QM9S files: IR median correlation 0.9999, Raman 0.949 (frequency scale 0.965).

### Training and evaluation

- **Model.** Qwen3-8B with LoRA (rank 16, alpha 32, all attention and MLP projections).
- **Schedule.** Learning rate 1e-4, 5% warm-up then cosine decay, 3 epochs, effective batch 16. Loss on the answer tokens only.
- **Identical batches.** Batch *k* of each epoch contains the same molecules and conditions in every arm.
- **Decoding.** Temperature 0.6, top-k 20, top-p 0.95, one sample per molecule. Greedy decoding gives the same picture (`B_modes_on` on IR+Raman: 0.356 against 0.349).
- **Metrics.** SpectraLLM's seven metrics, reimplemented. Primary: ECFP4 Tanimoto (Morgan radius 2, 2,048 bits) on IR+Raman. Invalid outputs count as 0.
- **Statistics.** Per-molecule paired differences, paired bootstrap with 10,000 resamples, Wilcoxon signed-rank tests.

### Cost

`components_v3` prompts are 2.2 to 3.2 times longer than SpectraLLM's.

| Arm | IR | Raman | IR+Raman |
|---|---|---|---|
| `A` | 333 | 263 | 479 |
| `A_native` | 414 | 301 | 599 |
| `B_modes_off` | 757 | 585 | 1,157 |
| `B_modes_on` | 1,018 | 703 | 1,518 |

Mean tokens per example on the test set (Qwen3 tokenizer; system prompt, prompt and answer).

## Limitations

- **One training run per arm.** The intervals cover variation between molecules, not between seeds. That hardly matters for the 0.10 gain from mode labels. It does matter for the 0.012 format-only gain, which should not be claimed without more seeds.
- **Training was not finished.** All four arms were still improving at epoch 3, so absolute scores are probably underestimates.
- **Absolute accuracy is low.** The best arm gets the exact molecule 8% of the time.
- **Mode labels need a source.** On experimental spectra they would have to come from somewhere other than the molecule's own Hessian, and would be less complete and less accurate.
- **No formal pre-registration.** The primary metric was named in the project roadmap before test predictions were generated.

## Against SpectraLLM's published numbers

For orientation only. The test molecules, model size and training data differ, and at least one metric is implemented differently (our Fraggle scores are 0.53 to 0.66 where theirs are 0.25 to 0.48). ECFP4 Tanimoto averaged over valid outputs, the paper's convention:

| Spectra | Our `A` | Our `B_modes_on` | Published, one model (Table 4) | Published, retrained per combination (Table 19) |
|---|---|---|---|---|
| IR | 0.195 | 0.299 | 0.192 | 0.147 |
| Raman | 0.196 | 0.270 | 0.250 | 0.325 |
| IR+Raman | 0.237 | 0.359 | 0.336 (also includes UV) | 0.532 |

## Notes on reproducing SpectraLLM's pipeline

- **IR axis.** The QM9S IR file runs 500 to 4000 cm⁻¹; the released notebook assigns 4000 to 400. Spectra rebuilt from the DFT modes fitted the reversed axis better for 0 of 60 molecules. This does not affect accuracy (result 4).
- **Peak threshold.** The released code drops peaks below 10% of the maximum; the paper states 1%.
- **Row order.** The broadened spectrum files are ordered as `qm9s.pt`, not by the molecule number in the per-molecule files.

## Next

- Seeds two and three for every arm.
- Learning curve at 1,000, 3,000 and 10,000 training molecules (subsets already built).
- Noise robustness using SpectraLLM's noise settings.
- A windowed variant (400–1800 cm⁻¹) to measure what a SERS-compatible window costs.
- SpectraLLM's released 32B adapter on these test molecules, as a reference for scale.

## Repository layout

> **TODO (Mo):** replace with the real tree (`tree -L 2 -I 'outputs|data|wandb'`).

```
configs/      one YAML per arm
data/         split files and peak lists (QM9S itself is not redistributed)
prompts/      serialisers for A, A_native and components_v3
train/        LoRA training script
eval/         inference, metrics and paired statistics
results/      per-arm metric tables and figures
```

## Reproduce

> **TODO (Mo):** fill in the real commands.

```bash
conda env create -f environment.yml && conda activate spectral-strings
python data/build_splits.py       # needs QM9S downloaded separately
python train/train.py --config configs/B_modes_on.yaml
python eval/predict.py --arm B_modes_on && python eval/score.py
```

Four training runs take about a day on one L40S (48 GB).

## References

- SpectraLLM: [arXiv:2508.08441](https://arxiv.org/abs/2508.08441)
- QM9S: Zou et al., [Nature Computational Science (2023)](https://www.nature.com/articles/s43588-023-00550-y)

## Licence

MIT. See `LICENSE`.
