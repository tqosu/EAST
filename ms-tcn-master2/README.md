# High-Frame-Rate Aggregation and Refinement (EAST Stage 2)

This folder implements the **second stage** of EAST's *segmentation-by-detection*
pipeline, corresponding to **Sec. 3.3–3.5** of the paper
([End-to-End Action Segmentation Transformer](https://arxiv.org/abs/2503.06316)).

The EAST transformer detector (Stage 1, run via `tools/train.py` / `tools/test.py`)
predicts action **proposals** over a coarsely down-sampled (low-frame-rate) video.
This stage takes those proposals and produces the final high-frame-rate, frame-wise
segmentation:

1. **Aggregation (Sec. 3.3).** Every frame of the full-resolution video aggregates the
   class distributions of all proposals whose temporal interval covers that frame; the
   summed scores are normalized into a per-frame distribution
   (`batch_gen.py:normalize_last_dim`).
2. **Refinement (Sec. 3.4).** The aggregated distributions are refined by a **3-stage
   temporal convolutional network** (`num_stages=3, num_layers=10, num_f_maps=64` in
   `main.py`) to produce the final labels.
3. **Proposal-dropout augmentation (Sec. 3.5).** During training, `K` of the top-`A`
   (`A=30`) most confident proposals are randomly removed before aggregation, forcing the
   remaining proposals to compete under higher uncertainty. `K` is controlled by
   `--lambda_val`.

## Acknowledgement

The TCN training scaffold (`model.py`, `batch_gen.py`, `main.py`) is derived from
**MS-TCN** — [yabufarha/ms-tcn](https://github.com/yabufarha/ms-tcn). The evaluation
metrics in `eval.py` are adapted from
[colincsl/TemporalConvolutionalNetworks](https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py).
Please credit and cite these original works if you use this code.

## Inputs

This stage consumes the `.npz` proposal files emitted by the Stage-1 detector. Generate
them by running `tools/test.py` with `evaluation.save_npz=True` (see each dataset's
README under `configs/adatad/<dataset>/`). Each `.npz` contains:

- `seg`  — proposals as `[label, start, end, prob]`
- `prob` — per-frame class-score buffer (target shape)
- `gt`   — ground-truth frame labels (`-100` marks ignored frames)

A `data/<dataset>/mapping.txt` file (action id ↔ name) is also required.

## Usage

### Train + evaluate

```shell
# from inside this folder
cd ms-tcn-master2

python main.py --action train --dataset gtea --split 1 \
    --directory_path     <path-to-train-proposals>/evaluation \
    --directory_path_eva <path-to-test-proposals>/evaluation \
    --sampler uniform --lambda_val 1 --bg_idx 10
```

Training logs per-epoch evaluation results and appends them to `results.csv`.

### Average the best results across splits

```shell
python max_FEA_avg_scores.py            # reads results.csv by default
python max_FEA_avg_scores.py --csv_file results.csv
```

This picks the best-**FEA** row per `(dataset, split)` and averages the metrics across
all splits of a dataset, printing the `F1@10 & F1@25 & F1@50 & Edit & Acc` summary.

## Arguments

| Argument         | Meaning |
| :--------------- | :------ |
| `--dataset`      | Dataset name (`gtea`, `50salads`, `breakfast`, `assembly101`). |
| `--split`        | Cross-validation split id. |
| `--directory_path`     | Folder of `.npz` **training** proposals. |
| `--directory_path_eva` | Folder of `.npz` **evaluation/test** proposals. |
| `--sampler`      | Proposal-dropout sampler: `uniform` or `poisson`. |
| `--lambda_val`   | Number of top-confidence proposals dropped during augmentation (Sec. 3.5). |
| `--bg_idx`       | Background class index, excluded from the F1 computation. |

## Metrics (`results.csv`)

Columns: `dataset, split, sampler, lambda, epoch, F1@10, F1@25, F1@50, Edit, f_Acc, o_Acc, FEA`.

- **F1@{10,25,50}** — segmental F1 at IoU overlap thresholds 0.10 / 0.25 / 0.50.
- **Edit** — segmental edit (Levenshtein) score between predicted and GT label sequences.
- **f_Acc / o_Acc** — frame-wise accuracy (filtered / overall).
- **FEA** — the combined selection score (mean of F1, Edit, Acc) used to pick the best
  checkpoint; `max_FEA_avg_scores.py` reports the row with the maximum FEA per split.
