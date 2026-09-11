# Research-workspace migration

This repository contains the reusable MIRACLE-AD pipeline extracted from the
original research workspace. The migration favors a small, installable package
over preserving the workspace's historical directory layout.

## Retained components

| Research component | Clean location |
|---|---|
| Backbone and MIL model definitions | `src/miracle_ad/models.py` |
| Supervised training and evaluation | `src/miracle_ad/training.py` |
| Supervised entry point | `src/miracle_ad/cli/train.py` |
| SimCLR-style backbone pretraining | `src/miracle_ad/ssl_training.py` |
| SSL audio pipeline | `src/miracle_ad/ssl_data.py` |
| JSON-based dataset loading | `src/miracle_ad/data.py` and `split_registry.py` |
| Fixed split artifacts and TAUKADIAL metadata | `src/miracle_ad/splits/` |

All 13 backbone choices from the active workspace are exposed through the same
factory and both SSL and supervised training. The paper's four pooling choices,
plus the workspace's experimental Transformer variant, are also retained.

The split JSON files now expose explicit `train`, `val`, and `test` membership.
The former `val` holdout is unchanged under `test`, while a deterministic,
label-stratified 10% of the legacy training pool is committed under `val`.
Audio paths and labels are otherwise unchanged. Checkout-relative paths are
normalized only in memory when a dataset root is selected.

## Intentional exclusions

The following workspace material is not required to train, evaluate, or run
inference with MIRACLE-AD and was therefore not copied:

- licensed speech recordings and local dataset directories;
- checkpoints, TensorBoard logs, plots, caches, and generated result tables;
- exploratory notebooks, temporary analyses, and duplicate legacy loaders;
- one-off comparison-baseline reproductions with machine-specific paths.

Those exclusions keep generated and non-portable material out of the public
package. They do not remove any backbone, MIL pooling option, language-aware
training path, attention export, or SSL pretraining capability used by the
MIRACLE-AD method itself.

## Portability and correctness changes

The clean package also makes several research-workspace assumptions explicit:

- the original holdout remains isolated under `test`, while validation
  membership is fixed directly in each versioned JSON artifact;
- every run records the exact resolved split membership and configuration;
- language IDs are contiguous across the four-language auxiliary task, and the
  combined TAUKADIAL artifact uses its participant language map;
- Gammatone front ends use the paper's 64 filters and 50 Hz–8 kHz range;
- Wav2Vec2 is optional and internally resamples 24 kHz input to the checkpoint's
  expected sampling rate;
- checkpoint resume, gradient accumulation, OneCycle stepping, fixed-label
  metrics, and CPU/GPU-safe weighted losses are handled consistently.

The original research workspace is not modified by this migration.
