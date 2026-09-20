# Data Directory — Review 1

*Only the two supervised datasets are needed here. BankChurners belongs to the
Review 2 (clustering) repository.*

```
data/
├── raw/         source CSVs - IMMUTABLE, git-ignored, never written to by any script
├── interim/     scratch space
└── processed/   derived artifacts (cluster labels etc.) - regenerable
```

## Required files

Place these two files in `data/raw/`:

| File | Expected shape | Source |
|---|---|---|
| `Student_Performance.csv` | 10,000 × 6 | https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression |
| `predictive_maintenance.csv` | 10,000 × 10 | https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification |

## How to get them

```bash
python scripts/download_data.py
```

It checks `data/raw/` first, then tries the Kaggle API, and otherwise prints
exact manual instructions. It also records SHA-256 checksums into
`results/tables/dataset_provenance.json`.

### Kaggle API (optional)
```bash
pip install kaggle
# Kaggle -> Account -> "Create New API Token" -> downloads kaggle.json
# put it at ~/.kaggle/kaggle.json   (Windows: %USERPROFILE%\.kaggle\kaggle.json)
chmod 600 ~/.kaggle/kaggle.json
```

> **Never put a Kaggle token in source code and never commit `kaggle.json`.**
> `.gitignore` already excludes it.

## Why raw data is not committed

The CSVs total about 0.7 MB and are redistributed under the licences shown on
their Kaggle pages. `.gitignore` excludes `data/raw/*` so the repository stays
clean and no licensing question arises. Checksums in
`results/tables/dataset_provenance.json` let anyone confirm they have the same
files we used:

| File | SHA-256 |
|---|---|
| `Student_Performance.csv` | `93793b00d9026d0b4907df0ca9f88b3696747c7496679d35833bb0fbf9fb57cf` |
| `predictive_maintenance.csv` | `9f0ede0b6fc33edacccfa1924e6430f92bd33ecce6e895543cd228c3a55ff10b` |

## Version warning

For predictive maintenance, use **this Kaggle release**. The UCI *AI4I 2020*
release has a different column structure (separate `TWF`/`HDF`/`PWF`/`OSF`/`RNF`
flags instead of `Target` + `Failure Type`) and is **not** a drop-in substitute.
`src/data_loading.load_raw` reports a shape mismatch loudly rather than adapting
silently.
