# LEWS 1.0: Litigation Early Warning Benchmark

**Can you forecast which drugs and medical devices will become the next mass torts?**

LEWS 1.0 is a temporally strict benchmark for forecasting whether a petition to the U.S. Judicial Panel on Multidistrict Litigation (JPML) will be **granted** (creating an MDL) or **denied**. It contains **167 real petitions** (78 granted, 89 denied) spanning 1970–2026. Each task is an evidence dossier reconstructed **strictly as of the petition date**: regulatory signals, scientific literature, court dockets, plus leading indicators such as plaintiff-firm intake advertising and litigation-funding activity. No information dated on or after the petition date appears in any dossier.

It accompanies the paper *Signals Beat Scale: Evidence Acquisition Dominates Model Choice in Forecasting Mass-Tort Consolidation* (Lee & Tandon, Decover AI; arXiv link forthcoming) and is maintained by [Decover AI](https://decover.ai), which operates the surrounding production system (LEWS).

## Leaderboard (paper v1)

5-fold out-of-fold cross-validation, n=167. Unanswered cases are excluded (coverage), never imputed.

| Model | AUROC | 95% CI | Coverage | Private* |
|---|---|---|---|---|
| Claude Sonnet 4.6 (API) | **0.903** | [0.854, 0.945] | 167/167 | ✗ |
| OpenLEWS-14B (3 runs) | 0.869 ± 0.016 | — | 167/167 | ✓ |
| HistGBM (21 features) | 0.871 | [0.813, 0.923] | 167/167 | ✓ |
| OpenLEWS-7B (4 runs) | 0.856 ± 0.011 | — | 166/167 | ✓ |
| Logistic regression (21 features) | 0.829 | [0.760, 0.890] | 167/167 | ✓ |
| Untrained heuristic | 0.691 | [0.609, 0.769] | 167/167 | ✓ |

Strict temporal holdout (train on earliest 75%, test on latest 25%, n=41, cutoffs 2022–2026): heuristic 0.672, logistic regression 0.870, OpenLEWS-7B 0.852, HistGBM 0.889, Claude Sonnet 4.6 0.909.

*Private = no investigation data leaves the operator's infrastructure at inference time.

## Quickstart

```bash
pip install -r requirements.txt

# score the paper's reference predictions (reproduces the leaderboard above)
python3 evaluation/evaluate.py results/paper_v1/predictions_*.json \
    --compare results/paper_v1/predictions_claude-sonnet-4.6.json

# temporal protocol (paper's temporal-train predictions)
python3 evaluation/evaluate.py results/paper_v1/temporal/predictions_*.json --temporal

# run an OpenLEWS model (or any causal LM) over the benchmark
python3 harness/run_openlews.py --base Qwen/Qwen2.5-7B \
    --adapter decoverai/OpenLEWS-7B-v1 --four-bit --out my_predictions.json
python3 evaluation/evaluate.py my_predictions.json \
    --compare results/paper_v1/predictions_claude-sonnet-4.6.json
```

## Task format

`tasks/lews_v1.jsonl`, one JSON object per petition:

| Field | Description |
|---|---|
| `id` | stable task id (`lews-0000` …) |
| `subject` | the substance / device / product at issue |
| `cutoff` | the JPML petition date; every dossier line predates this |
| `dossier` | the as-of-date evidence dossier (mean ~910 tokens) |
| `features` | 21 engineered signal counts (inputs to the classical baselines) |
| `label` | 1 = petition granted (MDL created), 0 = denied |
| `outcome` | `grant` / `deny` (readable form of `label`) |

**Prediction file format**: `{"model": "name", "predictions": {"lews-0000": 0.87, ...}}`. Use `null` for a case your model did not answer; the evaluator reports it as coverage and excludes it. **Do not impute 0.5 for a non-answer** (see the paper's Limitations section for why).

## Evaluation protocols

1. **Full (cross-validation)**: if you train on LEWS 1.0 dossiers, every prediction must come from a model whose training folds excluded that petition. Zero-shot / API models can simply predict all 167.
2. **Temporal** (`--temporal`): train only on the earliest 75% of petitions, predict the latest 25% (n=41). This is the stricter, headline protocol.

Metrics: AUROC with 10,000-sample bootstrap 95% CI, DeLong and paired-bootstrap tests against a reference model, Brier score, 10-bin ECE, and generation coverage.

## What is and is not released

- **Released**: all 167 frozen as-of-date dossiers, labels, engineered features, the evaluation harness, and the paper's reference predictions.
- **Not released**: the live ~17M-item signal corpus, the acquisition agents, and the pipeline that produces current dossiers for new substances. These are the commercial Decover AI / LEWS product. The dossiers here are a static historical snapshot; they do not update.
- Model weights: [decoverai/OpenLEWS-7B-v1](https://huggingface.co/decoverai/OpenLEWS-7B-v1) and [decoverai/OpenLEWS-14B-v1](https://huggingface.co/decoverai/OpenLEWS-14B-v1) on Hugging Face.

## Licenses

- **Code** (`evaluation/`, `harness/`): MIT (see `LICENSE`).
- **Data** (`tasks/`, `results/`): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): research and evaluation use; commercial use of the dataset requires permission from Decover AI (see `DATA_LICENSE`).

This benchmark is decision-support research material, **not legal advice**; predictions concern JPML petition outcomes, not the merits of any claim.

## Citation

```bibtex
@article{lee2026signals,
  title  = {Signals Beat Scale: Evidence Acquisition Dominates Model Choice in Forecasting Mass-Tort Consolidation},
  author = {Lee, Jason and Tandon, Ravi},
  year   = {2026},
  note   = {arXiv preprint, forthcoming}
}
```
