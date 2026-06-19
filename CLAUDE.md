# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [DyGLib](https://github.com/yule-BUAA/DyGLib) (the NeurIPS 2023 dynamic-graph library) extended with a thesis project on the `HT-Transformer` branch. The original DyGLib pipeline (8 baselines + DyGFormer, bipartite/unipartite dynamic link prediction and node classification) is preserved in `train_link_prediction.py`, `train_node_classification.py`, `evaluate_*.py`. The new work is a **tripartite (User, Streamer, Item/Room) temporal hypergraph link-prediction** task with a custom model `HTTransformer` and a dedicated training/eval stack, evaluated on the `kuailive_tripartite` dataset.

When asked about "the model" or "training", assume the tripartite stack unless the user names a DyGLib baseline task.

## Environment & commands

Python venv lives in `venv/` (activate with `venv\Scripts\Activate.ps1` on Windows PowerShell). PyTorch >= 1.8.1. Tripartite baselines additionally need `torch-geometric`, `mlxtend`, `torch-kmeans` (see `requirements.txt`).

There is **no test suite, linter, or build step** — this is a research codebase driven by command-line training/eval scripts.

### Tripartite task (the thesis work — primary)
```bash
# (Optional but recommended) build the fixed, shared ranking-candidate set once before evaluating.
# All models/ablations then rank against byte-identical negatives. Auto-built on first train run
# if missing, so this is only needed to inspect/verify the set.
python build_eval_candidates.py --verify

# Train HT-Transformer (full config) on kuailive_tripartite
python train_tripartite_link_prediction.py --model_name HTTransformer --num_runs 5

# Train a DyGLib baseline under the same tripartite scoring head
python train_tripartite_link_prediction.py --model_name DyGFormer --num_runs 1

# Ablations toggle HT-Transformer components (each changes the checkpoint name suffix):
python train_tripartite_link_prediction.py --model_name HTTransformer --no-use_type_init
python train_tripartite_link_prediction.py --model_name HTTransformer --no-use_hetero_coocc
python train_tripartite_link_prediction.py --model_name HTTransformer --no-use_bias_gate
python train_tripartite_link_prediction.py --model_name HTTransformer --fusion_mode mean
python train_tripartite_link_prediction.py --model_name HTTransformer --loss_type bpr
python train_tripartite_link_prediction.py --model_name HTTransformer --train_neg_ratio 5
```
**How to run the thesis experiments.** The batch engine is **`run_experiments.py`** (`--mode overall | ablation | hyperparam`): it loops the right runs, calls `train_tripartite_link_prediction.py --metrics_out <tmp.json>` for each, and appends one row to `experiment_results/results_{mode}.csv` (test_/val_ ranking metrics: P@K, N@K, AUC, AP). It forwards any extra CLI tokens to the trainer, so alignment flags must be passed in. Two thin PowerShell wrappers supply those flags from a single shared config so you don't have to remember them:
```powershell
# Overall (main) comparison — HT-Transformer vs. all baselines, 5 seeds each → results_overall.csv
./run_main_comparison.ps1     # = python run_experiments.py --mode overall   --num_runs 5 $Common

# Ablation (2^3 of --no-use_* flags) + hyperparam (patch_size×hidden grid) → results_ablation.csv / results_hyperparam.csv
./run_experiments.ps1         # = python run_experiments.py --mode ablation/hyperparam --num_runs 1 $Common
```
All fairness-critical knobs live in **`common_config.ps1`** (`$Common`: `--train_neg_ratio`, `--train_neg_sampling`, epochs, patience, batch, gpu), dot-sourced by both wrappers, so the *baseline training config is identical across the main comparison and the ablations* — change the alignment in one place; `--num_runs` is appended per wrapper. The aligned default protocol: training uses `--train_neg_ratio 5 --train_neg_sampling popularity` and validation/early-stopping uses popularity negatives too (see "Training/eval protocol" below). To run a mode directly without the wrappers, pass the same flags by hand, e.g. `python run_experiments.py --mode overall --num_runs 5 --train_neg_ratio 5 --train_neg_sampling popularity` (use `--dry_run` to print commands without training). `run_experiments.py` uses `subprocess.run(check=True)`, so one model failing aborts the rest of that batch. Thesis tables/figures are produced by `eval_table10.py` (per-user-bias-group metrics), `eval_table13.py` (tripartite vs. masked-2D ablation), `eval_coldstart.py` (zero-history scenarios), and `analyze_user_bias.py` (KMeans user grouping → `experiment_results/user_bias_groups.csv` + plots). These eval scripts load saved checkpoints and import helpers from each other (`eval_table13`/`eval_coldstart` import from `eval_table10`).

### Original DyGLib tasks (upstream, still functional)
```bash
python train_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0
python evaluate_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --negative_sample_strategy random --load_best_configs --num_runs 5 --gpu 0
```
`--load_best_configs` pulls grid-searched hyperparameters from `utils/load_configs.py` (only wired up for the original tasks, not the tripartite one).

### Data preprocessing
```bash
cd preprocess_data/
python preprocess_data.py --dataset_name wikipedia        # original DyGLib datasets (need raw files in DG_data/)
python preprocess_kuailive_tripartite.py                  # builds the tripartite dataset
```
The tripartite loader (`get_tripartite_link_prediction_data`) auto-runs preprocessing if the standard `ml_*` files are missing but raw `ml_kuailive_*` files exist.

## Architecture

### Tripartite scoring contract
Every tripartite model's forward returns `(y_hat, h_u, h_v, h_w)` for a batch of hyperedges `(u, v, w, t)` — User, Streamer, Item/Room embeddings plus a score. This shared signature is what lets `train_tripartite_link_prediction.py` swap models freely:
- **`models/HTTransformer.py`** — the custom model. Builds per-node temporal neighbor sequences, applies a DyGFormer-style `TransformerEncoder` (reused from `models/DyGFormer.py`), and adds three configurable components: type-aware node init (`use_type_init`), 3D heterogeneous neighbor co-occurrence encoding (`Tripartite3DCooccurrenceEncoder`, `use_hetero_coocc`), and a bias-aware merge head (`use_bias_gate`). `fusion_mode` selects concat vs. mean pooling of the three embeddings.
- **`models/BaselineTripartiteWrapper.py`** — wraps DyGLib pairwise encoders (`DyGFormer, TGAT, GraphMixer, CAWN, TCL, TGN`; see `TRIPARTITE_BASELINE_MODELS`) into the tripartite contract by making two pairwise calls: `(u,v)` for `h_u,h_v` and `(u,w)` for `h_w`, then a shared concat+MLP+sigmoid head. EdgeBank is excluded (non-parametric).

### Training/eval protocol (tripartite)
Defined in `train_tripartite_link_prediction.py`:
- **Training**: BCE (default) or BPR loss with `--train_neg_ratio` random negatives `(v', w')` per positive. `--train_neg_sampling` (`uniform`|`popularity`) sets the negative distribution. Per-epoch validation is a fast **1-negative BCE** on `val_data`; early stopping minimizes val loss (`utils/EarlyStopping.py`). The validation negative is drawn per `--val_neg_sampling` (`uniform`|`popularity`), **default `popularity`** (train-split frequencies, no leakage) so checkpoint selection aligns with the eval popularity component at the same cost as the old uniform 1-neg validation; pass `--val_neg_sampling uniform` for the legacy behaviour. `--early_stop_metric` can instead monitor a full ranking metric (`val_ndcg@5`, …) — wired but off by default (using it runs full ranking validation every epoch and means retraining).
- **Test**: after loading the best checkpoint, runs **1-positive + N-negative ranking** per query (`tripartite_ranking_metrics_per_query` in `utils/metrics.py`) → P@1, N@3/5/10, AUC, AP. Two negative-sampling protocols:
  - **Fixed shared candidates (default, `--use_fixed_eval_candidates`)** — negatives come from a pre-built, model-independent file (`utils/eval_candidates.py`), so every model/ablation ranks against byte-identical negatives. The mixture is harder than uniform: popularity-as-of-t (50%) + train-only user-similarity hard negatives (30%) + uniform (20%), with an all-time false-negative filter. See "Fixed evaluation candidates" below.
  - **Legacy (`--no-use_fixed_eval_candidates`)** — samples uniform `(v', w')` on the fly with `--eval_seed + 1`. Kept only for old-vs-new comparison.
- **Ranking metric tie handling**: `tripartite_ranking_metrics_per_query` computes HR/NDCG as the *expected value under a uniform random tie-break* (identical to integer-rank metrics when there are no ties). This fixed a real bug where a constant-score model (e.g. degenerate HAN) scored HR/NDCG = 1.0 while AUC = 0.5, because the old `argsort` rank always placed the positive (index 0) ahead of equal-scored candidates.
- `evaluate_tripartite_ranking(..., eval_candidates=...)` is the reusable test-ranking function imported by the `eval_*.py` scripts; pass a loaded candidate set (or `--eval_candidates_path` on those scripts) to use the shared protocol there too.

### Fixed evaluation candidates (`utils/eval_candidates.py`, `build_eval_candidates.py`)
`build_eval_candidates(eval_data, full_data, node_type_ids, config, train_data=...)` produces a per-query candidate set (`EvalCandidates`), saved as an `.npz` under `eval_candidates/` (gitignored; rebuilt from data + seed). Everything is parameterized off the data — pool sizes, the `(streamer, room)` combo universe, and `N` are read dynamically; no dataset scale is hard-coded. Determinism: one `RandomState(seed)` advanced in a fixed query order → same data + seed reproduces byte-identical negatives (verify with `build_eval_candidates.py --verify`). **No test-period leakage**: popularity is strictly history `< t`; hard-negative similarity/frequency use the **train split only**; only the deliberate false-negative filter (never propose a `(v,w)` the user truly interacts with at any time) uses the full timeline. On a small dev subset `N=99` may not fill (too few combos) — logged, not an error; full data fills it.

### Checkpoint & result naming
`save_model_name = f"{model_name}{suffix}_seed{seed}"`, where `suffix` encodes ablation flags (e.g. `_no_type_init`, `_mean`, `_bpr`, `_neg5`). Full-config HT-Transformer has empty suffix → `HTTransformer_seed0`. Note the `eval_table*.py` scripts default to a `HTTransformer_full_seed0` directory; verify the actual checkpoint dir name under `saved_models/HTTransformer/kuailive_tripartite/` before running them (rename or pass an explicit path if it mismatches). Outputs land in `saved_models/`, `saved_results/`, and `logs/`, each namespaced `<model_name>/<dataset_name>/<save_model_name>/`.

`TRIPARTITE_RANKING_KS` (in `utils/metrics.py`, mirrored in `run_experiments.py`) is `(1, 3, 5, 10, 20, 50, 100)` — `1` and `3` were added so the thesis tables' P@1 / N@3 are actually computed. Changing this set changes the `results_*.csv` columns, so start fresh CSVs rather than appending to old ones.

### Data layout
- `DG_data/` — raw upstream datasets (gitignored except Myket + `DATASETS_README.md`).
- `processed_data/<dataset>/` — `ml_<dataset>.csv` (tripartite cols: `u, streamer, room, ts, label, idx`), `ml_<dataset>_node.npy`, `ml_<dataset>_node_types.npy`, `ml_<dataset>.npy` (edge features), and optional `ml_<dataset>_temporal_edges.npz` (pairwise edges for the neighbor sampler). Node/edge features are zero-padded to dim 172.
- `reference_models/` — pristine upstream source for the baselines (HAN, HyperHawkes, LightGCN, tgn) kept for reference; the integrated versions live in `models/`.

### Shared utilities (`utils/`)
`DataLoader.py` (data loaders + tripartite split logic, holdout/dedup), `utils.py` (`NeighborSampler`, `get_tripartite_neighbor_sampler`, `convert_to_gpu`, `set_random_seed`, optimizer/param helpers), `metrics.py` (ranking metrics with tie-aware HR/NDCG), `eval_candidates.py` (fixed shared candidate sets), `EarlyStopping.py`, `load_configs.py`.
