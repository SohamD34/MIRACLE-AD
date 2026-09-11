# Experiment guide

## Primary architecture grid

The paper's eight primary configurations are the Cartesian product of:

- backbones: `gamma_gm_cnn`, `gamma_erb_cnn`
- pooling: `unilstm`, `bilstm`, `abmil`, `gated_abmil`

`configs/paper_grid.yaml` records the shared signal and optimization settings as
well as the Chinese, Spanish, English, Greek, and multilingual dataset groups.

Preview commands before allocating compute:

```bash
python scripts/run_paper_grid.py \
  --group chinese \
  --classes both \
  --language-awareness both \
  --data-root /path/to/Datasets
```

The script prints commands by default. `--execute` runs them serially. For a
cluster, redirect the preview to a job generator or select one command per job.

## Reproducibility controls

- Split membership is independent per corpus, so combining corpora does not
  reshuffle an individual dataset.
- The default seed is 42 for Python, NumPy, and PyTorch.
- Variable-length bags use recording batch size 1. Increase effective batch size
  with `--accumulation-steps`.
- Every run stores both resolved configuration and exact split membership.
- The test partition is passed only to final metric/plot generation, never early
  stopping.

GPU kernels can still introduce small nondeterministic differences. For strict
determinism, configure PyTorch's deterministic algorithms for the target GPU and
record the CUDA/cuDNN versions with the result.

## Language-aware objective

Enable the four-way auxiliary language head with `--language-aware`. The `auto`
loss then resolves to weighted disease cross-entropy plus weighted language
cross-entropy:

```text
L_total = L_disease + 0.5 * L_language
```

Change the coefficient with `--language-loss-weight`.

## SSL pretraining

`miracle-ad-ssl` trains any retained backbone with SimCLR-style NT-Xent loss.
It uses two independently augmented views, never reads the fixed test holdout,
and can cap source duration independently per language (4.25 training hours and
1.25 validation hours by default).

```bash
miracle-ad-ssl \
  --datasets pitt taukadial_e vas elu \
  --backbone gamma_erb_cnn \
  --batch-size 8 \
  --epochs 50 \
  --data-root /path/to/Datasets
```

Downstream modes:

- `--ssl-mode freeze`: load `_best.pt` and train only pooling/classification heads.
- `--ssl-mode finetune`: initialize from `_best.pt` and update the full model.
- `--ssl-mode none`: train the complete supervised model from scratch.

Use `--pretrained-backbone PATH` when the default
`outputs/ssl/<datasets>/<backbone>/_best.pt` location is not appropriate.

## Attention outputs

`abmil`, `gated_abmil`, and `transformer_abmil` can emit recording-level chunk
weights. Add `--save-attention-maps` during training or use
`miracle-ad-predict` for a machine-readable list of time ranges and weights.
Attention indicates model influence, not validated clinical localization.
