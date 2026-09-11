# Command-line reference

Install the package with `python -m pip install -e .` before using the four
`miracle-ad-*` commands. Run any command with `--help` to see the installed
version of its interface.

`MIRACLE_AD_DATA_ROOT` can supply the dataset root for commands that read audio.
An explicit `--data-root` takes precedence.

## `miracle-ad-train`

Train and evaluate a supervised recording-level MIL model. YAML values loaded
with `--config` become defaults; explicitly supplied CLI values override them.

### Data and model

| Argument | Description |
|---|---|
| `--config PATH` | YAML mapping of argument names to values. |
| `--datasets NAME [NAME ...]` | One or more bundled dataset keys; required unless supplied by the config. |
| `--data-root PATH` | Root beneath which split paths resolve. |
| `--split-dir PATH` | Optional directory of replacement `*_split.json` files. |
| `--backbone NAME` | Acoustic backbone. Default: `gamma_gm_cnn`. |
| `--network NAME` | MIL pooling head. Default: `abmil`. |
| `--sample-rate INT` | Input sampling rate in Hz. Default: `24000`. |
| `--chunk-duration FLOAT` | Chunk length in seconds. Default: `10`. |
| `--overlap-factor FLOAT` | Fractional chunk overlap in `[0, 1)`. Default: `0.5`. |
| `--combine-mci-ad` / `--no-combine-mci-ad` | Merge MCI into AD for two-class training. Default: disabled. |
| `--language-aware` / `--no-language-aware` | Enable the four-way auxiliary language head. Default: disabled. |

Backbone choices are `kurdish_cnn`, `gamma_erb_cnn`, `gamma_gm_cnn`,
`soundnet8`, `soundnet5`, `m3`, `m5`, `m11`, `m18`, `raw_audio_cnn`,
`wavenet`, `raw_audio_lstm`, and `wav2vec2`.

Pooling choices are `unilstm`, `bilstm`, `abmil`, `gated_abmil`, and
`transformer_abmil`.

### Optimization

| Argument | Description |
|---|---|
| `--epochs INT` | Total training epochs. Default: `100`. |
| `--batch-size INT` | Recording batch size. Variable-length bags require `1`. |
| `--accumulation-steps INT` | Gradient-accumulation steps. Default: `4`. |
| `--learning-rate FLOAT` | Initial learning rate. Default: `3e-2`. |
| `--final-learning-rate FLOAT` | Cosine scheduler minimum. Default: `3e-5`. |
| `--weight-decay FLOAT` | Optimizer weight decay. Default: `5e-5`. |
| `--patience INT` | Early-stopping patience measured in epochs. Default: `15`. |
| `--optimizer NAME` | `adam`, `adamw`, or `sgd`. Default: `adamw`. |
| `--scheduler NAME` | `plateau`, `onecycle`, `cosine`, or `step`. Default: `cosine`. |
| `--loss NAME` | `auto`, `ce`, `weighted_ce`, `focal`, or `combined`. `auto` selects weighted CE, or combined loss for language-aware runs. |
| `--class-weights auto\|W [W ...]` | Automatic inverse-frequency weights or one value per disease class. |
| `--focal-gamma FLOAT` | Focal-loss focusing parameter. Default: `2`. |
| `--language-loss-weight FLOAT` | Auxiliary language-loss coefficient. Default: `0.5`. |
| `--seed INT` | Python, NumPy, PyTorch, and loader seed. Default: `42`. |
| `--num-workers INT` | Audio-loading worker processes. Default: `0`. |
| `--device cpu\|cuda` | Device override; otherwise CUDA is selected when available. |

### SSL initialization, outputs, and resume

| Argument | Description |
|---|---|
| `--ssl-mode none\|freeze\|finetune` | Train from scratch, freeze an SSL backbone, or fine-tune it. |
| `--pretrained-backbone PATH` | Explicit SSL backbone state. |
| `--ssl-root PATH` | Root used for automatic SSL checkpoint discovery. Default: `outputs/ssl`. |
| `--wav2vec2-model NAME` | Hugging Face model identifier for the optional Wav2Vec2 backbone. |
| `--cache-dir PATH` | Optional Hugging Face cache directory. |
| `--output-root PATH` | Root for generated supervised runs. Default: `outputs/supervised`. |
| `--output-dir PATH` | Exact run directory, overriding generated hierarchy. |
| `--run-name NAME` | Final generated run-directory component. |
| `--continue-training` | Resume model, optimizer, scheduler, and history from the selected run directory. |
| `--save-attention-maps` | Export validation-set chunk-attention visualizations for attention-based pooling heads. |
| `--skip-file-check` | Skip the up-front existence check for every split path. |

New runs refuse to overwrite an existing run directory. Choose another
`--run-name`/`--output-dir`, or use `--continue-training`.

## `miracle-ad-ssl`

Pretrain any retained backbone with two augmented views and NT-Xent loss. Only
the committed `train` and `val` partitions are used; `test` is never opened.

| Argument | Description |
|---|---|
| `--backbone NAME` | Backbone to pretrain. Default: `gamma_gm_cnn`. |
| `--projection-hidden INT` | Projection-head hidden width. Default: `256`. |
| `--projection-dim INT` | Contrastive embedding width. Default: `128`. |
| `--datasets NAME [NAME ...]` | Source datasets. Required; `--dataset` is an alias. |
| `--data-root PATH` | Dataset root. |
| `--split-dir PATH` | Optional replacement split directory. |
| `--chunk-duration FLOAT` | Chunk length in seconds. Default: `10`. |
| `--chunk-overlap FLOAT` | Fractional overlap. Default: `0.5`. |
| `--sample-rate INT` | Input sampling rate. Default: `24000`. |
| `--epochs INT` | Pretraining epochs. Default: `50`. |
| `--batch-size INT` | Contrastive batch size; must be at least 2. Default: `8`. |
| `--lr FLOAT` | Initial learning rate. Default: `1e-3`. |
| `--final-lr FLOAT` | Cosine scheduler minimum. Default: `1e-7`. |
| `--weight-decay FLOAT` | AdamW weight decay. Default: `1e-5`. |
| `--temperature FLOAT` | NT-Xent temperature. Default: `0.07`. |
| `--num-workers INT` | Audio-loading workers. Default: `0`. |
| `--iterations-multiplier FLOAT` | Repeat-factor applied to the indexed training chunks. Default: `1`. |
| `--augment-both-probability FLOAT` | Probability that both views are augmented. Default: `1`. |
| `--max-train-hours-per-language FLOAT` | Per-language training-duration cap. Default: `4.25`; use `0` to disable. |
| `--max-validation-hours-per-language FLOAT` | Per-language validation-duration cap. Default: `1.25`; use `0` to disable. |
| `--seed INT` | Random seed. Default: `42`. |
| `--continue-training` | Resume from `checkpoint_last.pth`. |
| `--output-dir PATH` | Exact output directory; otherwise derived from datasets and backbone. |
| `--save-every INT` | Periodic backbone-save interval. Default: `10`. |
| `--wav2vec2-model NAME` | Hugging Face checkpoint for Wav2Vec2. |
| `--cache-dir PATH` | Optional Hugging Face cache directory. |
| `--device cpu\|cuda` | Device override. |

The best and final backbone states are `_best.pt` and `_final.pt`. The directory
also contains a resumable checkpoint, logs, `run_config.json`, and the exact
recording membership in `resolved_splits.json`.

## `miracle-ad-predict`

Run one trained checkpoint on one recording. By default, architecture and signal
settings are loaded from `run_config.json` beside the checkpoint.

| Argument | Description |
|---|---|
| `--input PATH` | Input audio recording; required. |
| `--checkpoint PATH` | Trained supervised model state; required. |
| `--config PATH` | Explicit run-configuration JSON. |
| `--backbone NAME` | Override the recorded backbone. |
| `--network NAME` | Override the recorded pooling network. |
| `--sample-rate INT` | Override the recorded sampling rate. |
| `--chunk-duration FLOAT` | Override the recorded chunk duration. |
| `--overlap-factor FLOAT` | Override the recorded chunk overlap. |
| `--combine-mci-ad` / `--no-combine-mci-ad` | Override two/three-class mode. |
| `--language-aware` / `--no-language-aware` | Override auxiliary language-head mode. |
| `--device cpu\|cuda` | Inference device override. |
| `--output-json PATH` | Also save the printed prediction report as JSON. |

Attention-capable pooling heads include each chunk's time range and normalized
weight in the report.

## `miracle-ad-splits`

Inspect split metadata without decoding audio, or validate every resolved path.

| Argument | Description |
|---|---|
| `--datasets NAME [NAME ...]` | Dataset subset; defaults to all bundled splits. |
| `--split-dir PATH` | Optional replacement split directory. |
| `--data-root PATH` | Resolve entries beneath this root. |
| `--check-files` | Check every resolved audio path; requires a data root. |
| `--json` | Emit a machine-readable report. |

## `scripts/run_paper_grid.py`

Preview or serially execute the paper architecture grid.

| Argument | Description |
|---|---|
| `--config PATH` | Grid YAML. Default: `configs/paper_grid.yaml`. |
| `--group NAME` | Dataset group from the YAML. Default: `multilingual`. |
| `--data-root PATH` | Dataset root; required. |
| `--output-root PATH` | Grid output root. Default: `outputs/paper-grid`. |
| `--classes 2\|3\|both` | Disease-label configurations. Default: `both`. |
| `--language-awareness standard\|aware\|both` | Auxiliary-head modes. Default: `both`. |
| `--ssl-mode none\|freeze\|finetune` | Backbone initialization mode. |
| `--execute` | Execute commands; without it, only print the generated commands. |
