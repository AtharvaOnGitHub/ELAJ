# Corpus CSV Specification

The preprocessing pipeline must produce a CSV file at `data/processed/studies.csv` (path configured in `configs/data/corpus.yaml` under `studies_csv`). The ELAJ training code reads this file to discover which studies are available, where their `.h5ad` files live, and what biological metadata they carry.

## Required columns

| Column | Type | Description |
|---|---|---|
| `id` | integer | Study index used by DSBN and PerStudyBatchSampler. Must be 0-indexed and contiguous (0, 1, 2, …). The number of unique IDs determines `n_studies` in the model config. |
| `name` | string | Human-readable study name, e.g. `fonseca_2023`. Used for logging only; no spaces. |
| `path` | string | Path to the preprocessed `.h5ad` file. May be absolute or relative to the project root. |
| `tissue_types` | string | Pipe-separated (`\|`) list of tissue types present in this study. Valid values: `eutopic`, `ectopic`, `control`, `other`. |
| `has_cycle_phase` | boolean | Whether the study has clinical menstrual cycle phase labels. Accepted values: `true`/`false`, `1`/`0`, `yes`/`no` (case-insensitive). |

## Example

```csv
id,name,path,tissue_types,has_cycle_phase
0,fonseca_2023,data/processed/fonseca_2023.h5ad,eutopic|ectopic,false
1,xiao_2024,data/processed/xiao_2024.h5ad,eutopic|ectopic|control,true
2,chen_2022,data/processed/chen_2022.h5ad,eutopic,false
```

## Constraints

- `id` values must be contiguous starting from 0. Gaps will cause `nn.Embedding` index-out-of-range errors at runtime.
- Rows are sorted by `id` when loaded. Row order in the CSV does not matter.
- The `path` column is not validated at config-load time. A bad path will raise a `FileNotFoundError` when the `EndometriosisDataset` first tries to open the file.
- Each study must have at least one tissue type. An empty `tissue_types` field will produce an empty list, which is valid but should be flagged during data QC.

## How the preprocessing pipeline should write this file

```python
import csv
from pathlib import Path

studies = [
    {"id": 0, "name": "fonseca_2023", "path": "data/processed/fonseca_2023.h5ad",
     "tissue_types": ["eutopic", "ectopic"], "has_cycle_phase": False},
    ...
]

out_path = Path("data/processed/studies.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["id", "name", "path", "tissue_types", "has_cycle_phase"]
    )
    writer.writeheader()
    for s in sorted(studies, key=lambda x: x["id"]):
        writer.writerow({
            **s,
            "tissue_types": "|".join(s["tissue_types"]),
            "has_cycle_phase": str(s["has_cycle_phase"]).lower(),
        })
```

## Where this is used in the codebase

`DataConfig.resolve_studies(root)` in `endo_model/configs/data_config.py` calls `load_studies_csv(csv_path)` which reads this file and returns a list of `StudyConfig` objects. The `EndometriosisDataset` (Phase 3) calls `resolve_studies` at initialisation time.
