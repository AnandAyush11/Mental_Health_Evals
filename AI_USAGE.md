# AI usage

The brief allows AI tools and asks, for each one, for a shareable link, what it was used for, and how its output was checked. I remain responsible for every claim, result, figure, table and line of code submitted.

## Tool and link

| Tool | Link | Used for |
| --- | --- | --- |
| Claude (Anthropic), one long working chat | https://claude.ai/share/10dace5f-2edd-494d-b84b-046823344815 | Generating hypotheses; understanding the problem and planning the approach; writing the code (`run.ipynb`, `analysis.ipynb`, `posthoc.ipynb` and a CPU test harness); checking the smoke, pilot and final outputs; writing `HYPOTHESES.md`, `README.md`, this file and the report |

## Human verification

I verified every step before moving on: each hypothesis and design decision, every pre-registration version, every notebook before it ran, every output, and every number in the report. I ran every notebook on Kaggle (the smoke and pilot runs, the two final runs, `analysis.ipynb` and `posthoc.ipynb`), ran the float32 and reasoning-first checks on Qwen (`dev_runs/qwen_checks.py`), and decided each change after the smoke run and the pilot. Each change is recorded with its reason in `HYPOTHESES.md` §7 and §9.

## How the AI output was checked

- **Checks inside `run.ipynb`, before any generation:** the dataset's SHA-256; every prompt string compared with a separately typed copy of the brief's wording; fixed test cases for the parser, the leak detector and the quote matcher; pinned model-file hashes; the rendered chat template; the deletion-test scorer reproducing the generation probabilities.
- **During generation:** every generated token is checked against the argmax of the raw logits (0 mismatches in 2,320 final generations).
- **Smoke and pilot runs** on development rows, with pass or fail gates fixed in advance. I reviewed their raw outputs and asked for the changes listed in `HYPOTHESES.md`.
- **After the final runs:** `analysis.ipynb` and `posthoc.ipynb` were re-run on Kaggle and gave identical tables, a separately written script recomputed the headline numbers from the raw records, and I matched every number in the report against the result files.
