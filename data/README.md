# Probability-distortion datasets

Raw replication data for comparing theories of how probabilities are distorted in belief
updating (decision-time bias vs. noisy sampling, etc.). Everything under `raw/` is
unmodified source material; derived/harmonized files should go in `processed/`.

| Dir | Paper | Source | Status | Key data files |
|---|---|---|---|---|
| `raw/hill2017/` | Hill (2017), *Learning Together Slowly*, JOP | Dataverse doi:10.7910/DVN/LGDIZD | ✅ | `AnalysisData-long.tab`, `AnalysisData-wide.tab`, `Exper2-DataForStata.tab`, `PewWeights.tab` |
| `raw/tappin2020/` | Tappin, Pennycook & Rand (2020), *Bayesian or Biased?* | OSF zwjkd (components xqzu4, v65js, 7mh3y) | ✅ | `studies1_2/S1_data.csv`, `S2_data.csv` (+ codebooks), `control_study/Control-Study_data.csv` |
| `raw/mobius2022/` | Möbius, Niederle, Niehaus & Rosenblat (2022), *Managing Self-Confidence*, Mgmt Sci | INFORMS mnsc.2021.4294 | ❌ **manual** | — |
| `raw/buser2018/` | Buser, Gerhards & van der Weele (2018), *Responsiveness to Feedback as a Personal Trait* | OSF g96xz | ✅ | `Data.dta`, `Codes.do` |
| `raw/augenblick2025/` | Augenblick, Lazarus & Thaler (2025), *Overinference from Weak Signals…*, QJE | Dataverse doi:10.7910/DVN/EGLSZC | ✅ (git-ignored, 1.6 GB) | `Experiment/data/expt1a_cleaned.dta`, `experiment1b_data.dta`, `expt2_cleaned.dta`, raw CSVs, `benjamin2019_data.dta`; `Empirical.zip` (finance/betting-market data, left zipped) |
| `raw/enke_graeber2023/` | Enke & Graeber (2023), *Cognitive Uncertainty*, QJE | Dataverse doi:10.7910/DVN/IQ47ZB | ✅ | `replication_files_rev2/{A,B} Experiments/data/Beliefs/*.dta` (also Risk, Survey) |
| `raw/xie_hayes2022/` | Xie & Hayes (2022), *Sensitivity to Evidential Dependencies* | OSF qng93 | ✅ | `Data & Data Analysis/csv/expt1_data_{1,2}of2.csv`, `expt2/csv-participants/*.csv`, `expt3/`, model predictions |
| `raw/howe2022/` | Howe et al. (2022), *Base rate neglect and conservatism…* | GitHub perfors/probability | ✅ | `repo/Experiment{1,2,3}_Andy.csv`, `repo/datafiles/` |
| `raw/loibl2025/` | Loibl & Leuders (2025), *Bayesian Reasoning Without Numbers* (3 studies) | PsychArchives cf88c52b-… (doi:10.23668/psycharchives.21544) | ❌ **manual** | — |

## Processed data

`processed/` holds the 7 on-hand datasets converted to one shared trial-level format
(one row per elicited probability; see `processed/SCHEMA.md`):

- `processed/<dataset>.csv.gz` (one file per dataset) and `processed/all_datasets.csv.gz` (98,656 rows, 8,623 subjects)
- `processed/summary.csv`: per-study coverage and how closely reported posteriors track Bayes
- `processed/notes/<dataset>.md`: design, column mapping, exclusion flags, and caveats for each dataset

Rebuild everything from `raw/`:

```bash
uv run python scripts/convert_data/combine.py --rebuild
```

The authors' exclusion criteria are kept as flags in `extra`; no rows are dropped.

## Manual downloads needed

**Möbius et al. (2022)** — the INFORMS page requires ticking an agreement and entering an
email address. Note the terms: files are "to be used only for verifying replicability of the
relevant paper's main results"; other use requires permission from all authors. Worth
emailing the authors if this data is used for model comparison.
Download from https://services.informs.org/dataset/mnsc/download.php?doi=mnsc.2021.4294
and unzip into `raw/mobius2022/`.

**Loibl & Leuders (2025)** — PsychArchives bitstreams redirect to a Leibniz-Psychology
login. Sign in at https://www.psycharchives.org/en/item/cf88c52b-5324-4217-a287-bfce217c9f46
and save the four files into `raw/loibl2025/`:
- Codebook for studies 1–3 (bitstream ee7ab7b6-58d8-41e6-b643-db49253840e3)
- Study 1 data (415a7e66-957f-4885-9476-0b13924f7dbc)
- Study 2 data (eb43e5cf-1bce-466e-a204-b8fff1b06652)
- Study 3 data (9a7bf8ca-0a9d-4715-a1eb-b4611fc3fa54)

## Re-downloading

```bash
# Dataverse (whole dataset as zip)
curl -L -o x.zip "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/<ID>"
# OSF (one node's osfstorage as zip; Tappin's files live in child components)
curl -L -o x.zip "https://files.osf.io/v1/resources/<NODE>/providers/osfstorage/?zip="
```

Downloaded 2026-11-16.
