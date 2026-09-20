# Dataset Sources, Provenance and Licensing — Review 1

*The clustering dataset (BankChurners) belongs to Review 2 and is documented in
that repository.*

Both files were supplied by the team from Kaggle downloads and copied
unmodified into `data/raw/`. Raw files are treated as immutable — no script in
this repository writes to `data/raw/`.

Checksums are recorded automatically on every load into
`results/tables/dataset_provenance.json`.

---

## 1. Regression — Student Performance (Multiple Linear Regression)

| | |
|---|---|
| **Source** | https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression |
| **Author** | Nikhil Narayan (Kaggle user `nikhil7280`) |
| **File** | `Student_Performance.csv` |
| **SHA-256** | `93793b00d9026d0b4907df0ca9f88b3696747c7496679d35833bb0fbf9fb57cf` |
| **Size** | 175,071 bytes |
| **Raw shape** | 10,000 rows × 6 columns |
| **Prepared shape** | 10,000 rows × 6 columns (5 inputs + target) |
| **Target** | `Performance Index` (continuous, 10–100, integer-rounded) |
| **Synthetic?** | **Yes — explicitly synthetic.** No real student was measured. |
| **Licence** | Stated on the Kaggle dataset page; verify the current licence there before redistributing. Not reproduced here. |

**Inputs:** Hours Studied · Previous Scores · Extracurricular Activities (Yes/No)
· Sleep Hours · Sample Question Papers Practiced

**Quality notes:** no missing values; **127 exact duplicate rows** (see Q-RG1).

---

## 2. Classification — Machine Predictive Maintenance

| | |
|---|---|
| **Source** | https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification |
| **Author** | Shivam Bansal (Kaggle user `shivamb`) |
| **File** | `predictive_maintenance.csv` |
| **SHA-256** | `9f0ede0b6fc33edacccfa1924e6430f92bd33ecce6e895543cd228c3a55ff10b` |
| **Size** | 531,014 bytes |
| **Raw shape** | 10,000 rows × 10 columns |
| **Prepared shape** | 10,000 rows × 7 columns (6 inputs + target) |
| **Target** | `Target` (binary: 0 = no failure, 1 = failure) |
| **Class balance** | 9,661 / 339 → **3.39 % positive** |
| **Synthetic?** | **Yes.** Modelled on a milling machine; no real machine was measured. |
| **Licence** | Stated on the Kaggle dataset page. Derived from the UCI *AI4I 2020 Predictive Maintenance Dataset* lineage. |

**Excluded columns and why:**

| Column | Reason |
|---|---|
| `UDI` | row identifier — carries no signal, invites index leakage |
| `Product ID` | identifier, 10,000 unique values |
| `Failure Type` | **target leakage** — a second recording of the same event as `Target`. Including it would produce near-perfect scores that mean nothing. Evidenced by the crosstab in `notebooks/classification.ipynb` § 1. |

> **Version warning.** Use this Kaggle release. The UCI AI4I 2020 release has a
> different column structure (separate `TWF`/`HDF`/`PWF`/`OSF`/`RNF` failure
> flags) and is **not** a drop-in substitute.

---

## External code sources cited

| Where | What | Source |
|---|---|---|
| `src/clustering_models.elbow_knee` | largest-perpendicular-distance knee detection | idea from Satopaa et al. (2011), *"Finding a Kneedle in a Haystack: Detecting Knee Points in System Behavior"*, ICDCS Workshops. Implemented from the description, not copied. |
| everywhere | estimators, metrics, splitters, transformers | scikit-learn (BSD-3-Clause) |
| `src/plotting.dendrogram_plot` | `scipy.cluster.hierarchy.linkage` / `dendrogram` | SciPy (BSD-3-Clause) |

No StackOverflow or blog code was pasted into this repository.
