# Recorded results

Plain-text output of the experiments quoted in the main README and the project report.
All of them use dataset 2 (`data/policies_v2`, `data/labels_v2.csv`), the same
5-fold `StratifiedGroupKFold` split (grouped by family, seed 42) and the same
training settings.

| File | Produced by |
|---|---|
| `baselines.txt` | `python -m pg.baselines data/policies_v2 data/labels_v2.csv` |
| `train_base.txt` | `python -m pg.train data/policies_v2 data/labels_v2.csv base` |
| `train_kinds.txt` | `python -m pg.train data/policies_v2 data/labels_v2.csv kinds` |
| `ablate.txt` | `python -m pg.ablate data/policies_v2 data/labels_v2.csv 5` |
| `analyze_base.txt` | `python -m pg.analyze data/policies_v2 data/labels_v2.csv base 5` |
| `analyze_kinds.txt` | `python -m pg.analyze data/policies_v2 data/labels_v2.csv kinds 5` |

## Environment these were produced in

Linux x86_64, CPU only (one core), Python 3.12.3, PyTorch 2.14.1, PyTorch Geometric
2.8.0.post1, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.6, networkx 3.7.

An earlier single-seed run on Windows (Python 3.14.7, seed 42, base features) gave
mean precision/recall/F1 of 0.809 / 0.782 / 0.752. The Linux run with the same seed
gave 0.828 / 0.773 / 0.761. The gap is about the size of the seed-to-seed spread, so
expect small differences when you rerun on another machine.

`ablate` takes about 15 minutes on a single CPU core.
