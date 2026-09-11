# Datasets and fixed splits

MIRACLE-AD uses audio from DementiaBank-derived corpora and NCMMSC. The audio is
not distributed by this repository; obtain each corpus under its own terms.

## Labels

The committed split files use the same integer mapping across corpora:

| ID | Class |
|---:|---|
| 0 | Healthy control (HC) |
| 1 | Alzheimer's dementia (AD) |
| 2 | Mild cognitive impairment (MCI) |

`--combine-mci-ad` maps label 2 to label 1 at load time and leaves the JSON
artifacts unchanged.

The auxiliary language task uses contiguous IDs for English, Chinese, Spanish,
and Greek. The M. Chou corpus is assigned to the Chinese language group, matching
the paper's four-group evaluation.

## Partition semantics

Every committed split artifact has three explicit keys:

- `train`: recordings used to fit model parameters.
- `val`: recordings used for model selection and early stopping.
- `test`: the untouched legacy holdout, used only for final evaluation.

The former two-part artifacts used `val` for their test holdout. During the
three-way migration, that section was renamed to `test` without changing its
membership or labels. Within each dataset and label, paths from the legacy
training pool were sorted, deterministically shuffled with seed 42, and about
10% were moved to the new `val` section. Rounding preserves every label where
possible; across all artifacts, exactly 200 of the original 2,000 training-pool
entries moved to validation.

The machine-readable migration recipe and aggregate counts are stored in
`src/miracle_ad/splits/split_provenance.json`.

Membership is now fixed in JSON and is never resampled at runtime. The files
still contain legacy `../Datasets/...` prefixes and mixed slash styles; the
loader normalizes those paths beneath `--data-root` on Windows, macOS, and
Linux.

## Included artifacts

| Dataset key | Language | Train | Val | Test |
|---|---|---:|---:|---:|
| `elu` | English | 12 | 2 | 10 |
| `gds3` | Greek | 166 | 18 | 47 |
| `gpilot` | Greek | 130 | 14 | 36 |
| `ivanova` | Spanish | 256 | 29 | 72 |
| `mchou` | Chinese | 187 | 21 | 53 |
| `ncmmsc` | Chinese | 202 | 22 | 56 |
| `pitt` | English | 337 | 37 | 206 |
| `taukadial_c` | Chinese | 116 | 13 | 44 |
| `taukadial_e` | English | 115 | 12 | 34 |
| `taukadial` | Mixed TAUKADIAL provenance artifact | 230 | 26 | 78 |
| `vas` | English | 49 | 6 | 20 |
| **Total** |  | **1,800** | **200** | **656** |

Prefer `taukadial_c` and `taukadial_e` in experiments. The combined
`taukadial_split.json`, language map CSV, and statistics JSON are retained for
auditability.

## Expected relative roots

The exact nested paths are visible in the JSON files. Their leading corpus
locations are:

```text
Chinese/NCMMSC/
English/0extra/Greek/DS3/
English/0extra/TAUKADIAL/TAUKADIAL-24/train/
English/Lu/
English/Pitt/
English/VAS/
Greek/Dem@Care/pilot/
Mandarin/Chou/
Spanish/Ivanova/
```

Validate metadata only:

```bash
miracle-ad-splits
```

Validate every resolved audio path:

```bash
miracle-ad-splits --data-root /path/to/Datasets --check-files
```
