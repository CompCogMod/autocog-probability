# Shared trial-level schema

One row = one elicited probability judgment. All probabilities are in [0,1] and refer to P(H) for the row's `hypothesis`.
Generated from `scripts/convert_data/common.py` (`SCHEMA`). Per-dataset mapping decisions are in `notes/<dataset>.md`.

| column | meaning |
|---|---|
| `dataset` | dataset key, e.g. 'hill2017' |
| `study` | study / experiment / wave label within the dataset |
| `source_file` | path of the raw file relative to data/raw/ |
| `source_row` | 0-based row index in the raw file (or identifying key if reshaped) |
| `subject_id` | globally unique subject id: '<dataset>:<study>:<original id>' |
| `trial` | 0-based order of this judgment within the subject (NaN if unknown) |
| `round` | 0-based update step within an item/sequence (0 = first posterior; -1 = prior-only row) |
| `condition` | between- or within-subject manipulation label(s); '' if none |
| `item_id` | identifier of the item / question / urn configuration |
| `item_desc` | human-readable item info (question text, urn composition, etc.) |
| `task_domain` | one of: abstract (urns/balls/bags/cards), ego (own ability/rank), political_fact, other |
| `hypothesis` | what the probability is about, e.g. 'urn A', 'statement true', 'top half' |
| `prior_objective` | P(H) given by the design before this step's evidence (NaN if none) |
| `prior_reported` | subject's own reported P(H) before this step's evidence (NaN if not elicited) |
| `signal_desc` | the evidence shown at this step, as text |
| `signal_dir` | +1 evidence favours H, -1 favours not-H, 0 neutral/uninformative, NaN unknown |
| `n_signals` | number of signals in this step's evidence |
| `p_signal_h` | P(observed evidence at this step | H) |
| `p_signal_not_h` | P(observed evidence at this step | not-H) |
| `log_lr` | ln[p_signal_h / p_signal_not_h] of this step's evidence |
| `posterior_reported` | subject's reported P(H) after the evidence (the judgment of this row) |
| `posterior_bayes` | Bayesian P(H) from prior_objective (or cumulative objective prior) and log_lr |
| `posterior_bayes_rep_prior` | Bayesian P(H) from prior_reported and log_lr |
| `response_scale` | original response format, e.g. 'percent 0-100', 'slider 0-1', 'distribution' |
| `rt_sec` | response time in seconds (NaN if unavailable) |
| `extra` | JSON object with any remaining trial/subject info (covariates, incentives, raw codes, ...) |

## Files

- `<dataset>.csv.gz`: one file per dataset, written by `scripts/convert_data/<dataset>.py`
- `all_datasets.csv.gz`: all datasets stacked (`uv run python scripts/convert_data/combine.py [--rebuild]`)
- `summary.csv`: per dataset/study counts, column coverage, and the correlation between reported and Bayesian posteriors

`posterior_bayes` and `posterior_bayes_rep_prior` are filled automatically by `finalize()` from `prior_objective`/`prior_reported` and `log_lr` (from `p_signal_*`) unless a converter sets them (for example, chained multi-step updates).
