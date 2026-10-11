# Responsible medical LLMs for mental healthcare: a pre-registered benchmark

LCS2 (IIT Delhi) coding round. Two fixed instruction-tuned models, `meta-llama/Llama-3.2-3B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`, answer one mental-health multiple-choice task (MedMisBench-MentalHealth). They are evaluated on four dimensions (hallucination and uncertainty, interpretability, bias and fairness, privacy and security), on one cross-dimension test and on one intervention. The hypotheses, metrics and verdict rules were fixed in `HYPOTHESES.md` and pushed before any generation on a reported row. The report is submitted separately as a PDF; its LaTeX source is in `report/` if present.

## Contents

| Path | What it is |
| --- | --- |
| `HYPOTHESES.md` | Pre-registration, v3.3: hypotheses, primary metric per section, thresholds, the changes made before the final run with their reasons, and the pilot record |
| `run.ipynb` | Every model run (Kaggle GPU): smoke, pilot and final modes, every prompt string and setting, and the checks that run before and during generation. Writes `outputs/<mode>_<model>/` |
| `analysis.ipynb` | Every pre-registered table, figure and verdict, computed from `outputs/` (CPU) |
| `posthoc.ipynb` | Three post-hoc descriptions used in the report, added after the final run (CPU) |
| `outputs/final_llama/`, `outputs/final_qwen/` | The raw final outputs, unedited: `settings.json`, `records.jsonl` (one record per condition and row, including fallback parses and outputs cut at the token limit), `interp.jsonl` (deletion test), `run_meta.json`, `report.txt` |
| `results/` | `tables/*.csv`, `figures/*.png`, `summary.json`, and the executed notebooks `analysis_executed.ipynb` and `posthoc_executed.ipynb` |
| `dev_runs/` | Development runs, never reported: the smoke (rows 447–448) and pilot (rows 200–237) output folders for both models as `run.ipynb` wrote them, and `qwen_checks.py` with the code and output of the float32 and reasoning-first checks that `HYPOTHESES.md` §9 cites |
| `AI_USAGE.md` | AI tools used, what for, and how their output was checked |

## Fixed setup

| Item | Value |
| --- | --- |
| Models | `meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4f764b7a12671c53f0838cd831a0843b95`; `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Dataset | `Johndfm/MedMisBench-MentalHealth` @ `9757095598b6555c3e935679fdb7384aed4eb651`, `medmisbench_mentalhealth.jsonl`, split `train`, 449 rows, SHA-256 `0c7db2295ca5bf220933bc95d8f613c11e3f8c0d3d5f0235dbf56595d5434c21` |
| Subsets | Core = rows 0–199, Stress = rows 0–59, Interpretability = rows 0–39, in stored order; no row is edited or removed |
| Inference | float16, greedy decoding, batch size 1, eager attention, at most 256 new tokens, assistant prefill `{"answer": "`, one T4 GPU. Every generated token is checked against the argmax of the raw logits |
| Seed | 42 for all sampling code (bootstraps and random deletion sets) |
| Settings hash | `c3bca64c0891632ae757c85d0f6ebea016fad3a097dadd99139819dc7d5d0e8c`: SHA-256 of the fixed settings, stored in every record; `analysis.ipynb` stops unless every record carries it |
| Prompts and conditions | `run.ipynb` §1 (the Settings cell); each record also stores its full prompt (`prompt_text`) |
| Pre-registration | Tag `prereg-v3.3`, commit `f15ad05`, 11 Oct 2026 02:29 IST; the final runs started at 02:34 IST |

## Reproduce from a fresh Kaggle notebook

### 1. Model runs (GPU, about three hours of compute)

1. On Kaggle, create a new notebook and import `run.ipynb` (*File → Import Notebook*).
2. *Settings*: accelerator **GPU T4 x2** (the notebook uses one GPU); internet **on**.
3. *Add-ons → Secrets*: add **`HF_TOKEN`**, a Hugging Face token from an account that has accepted the Llama 3.2 licence, and enable it for the notebook. The notebook never prints or saves it.
4. In the Settings cell, the only cell to edit, set `MODE = "final"` and `MODEL = "llama"`, then *Save Version → Save & Run All (Commit)*. Set `MODEL = "qwen"` and commit again. Each version loads one model in its own session; the Llama run took 109 minutes and the Qwen run 76.
5. From each version's *Output* tab, download `outputs/final_llama/` and `outputs/final_qwen/`.

The tagged `run.ipynb` still has `MODE = "pilot"`, left from the pilot run, so set it to `"final"`. `MODE` and `MODEL` are not part of the settings hash, so the hash stays `c3bca64c…`; each output's `settings.json` records the mode it ran in. Smoke and pilot modes (development rows only) are described at the top of `run.ipynb`.

Package versions of the final runs, also stored in every record: Python 3.13.15, torch 2.11.0+cu128 (CUDA 12.8), transformers 5.16.1, tokenizers 0.23.1, huggingface_hub 1.29.0, numpy 2.1.3, accelerate 1.14.0, on Kaggle's GPU image of 10–11 Oct 2026 (Tesla T4, driver 580.178.04). If the image has changed since, install these versions first: `pip install transformers==5.16.1 tokenizers==0.23.1 huggingface_hub==1.29.0 accelerate==1.14.0`.

### 2. Analysis (CPU, about a minute)

1. Put `final_llama/` and `final_qwen/` in one folder (in this repository, `outputs/`).
2. Open `analysis.ipynb` in a Kaggle CPU notebook or local Jupyter (needs numpy, pandas, scipy and matplotlib), set `OUTPUTS_DIR` in the Settings cell to that folder, and *Run All*. Section 1 stops the notebook if any record is missing, duplicated, made with other settings, or failed the greedy audit. Tables go to `results/tables/`, figures to `results/figures/`, verdicts to `results/summary.json`.
3. Run `posthoc.ipynb` the same way for the three post-hoc tables.

`analysis.ipynb` and `posthoc.ipynb` were both run on Kaggle's CPU image, unchanged except that `OUTPUTS_DIR` points at a Kaggle input dataset. `results/analysis_executed.ipynb` and `results/posthoc_executed.ipynb` are those two runs. A second run of both notebooks outside Kaggle (Python 3.13.16, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, matplotlib 3.11.2) gave identical tables and `summary.json`.

## Where the report's numbers come from

| Report element | Source in `results/` |
| --- | --- |
| Pre-registered verdicts and primary metrics | `tables/summary.csv`, `summary.json` |
| Part A: accuracy and calibration | `tables/A_accuracy_calibration.csv`, `tables/A_output_format.csv`, `figures/A_reliability.png` |
| Part B1 and H2: misleading context | `tables/B1_misleading_coreports.csv`, `tables/B1_H2_auroc.csv` |
| Part B2: deletion test | `tables/B2_deletion_test.csv`, `figures/B2_deletion_curves.png`, the cases printed in `analysis_executed.ipynb` §5, `tables/posthoc_deletion_robustness.csv` |
| Part B3: metadata pairs | `tables/B3_metadata_pairs.csv`, `figures/B3_flip_rates.png` |
| Part B4: injection and canary | `tables/B4_injection.csv`, `tables/B4_canary.csv`, `tables/posthoc_canary_leak_location.csv` |
| Part C and H1 | `tables/C_H1.csv`, `figures/C_adoption.png`, exploratory `tables/X1_*.csv` to `tables/X5_*.csv` |
| Part D: defence and H1 corollary | `tables/D_defence.csv`, `tables/D_evidence_quality.csv`, `figures/D_defence.png`, `tables/posthoc_canary_leak_location.csv` |
| Range of stated confidence | `tables/posthoc_stated_confidence.csv` |
| Qualitative cases | `outputs/final_<model>/records.jsonl` and `interp.jsonl`, by the row numbers given in the report |

## Pre-registration and deviations

`HYPOTHESES.md` lists every change made before the final run, with dates and reasons (v2 to v3.3); the last version followed the pilot and changed only `HYPOTHESES.md` and `analysis.ipynb`. Nothing pre-registered changed after the final run. `posthoc.ipynb` is the only later addition, and the report labels its numbers post-hoc.

The same repo can be accessed on https://github.com/AnandAyush11/Mental_Health_Evals
