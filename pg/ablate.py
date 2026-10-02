"""Compare feature sets over several seeds (5-fold grouped CV, same folds each time).

usage: python -m pg.ablate [folder] [labels.csv] [n_seeds]
"""
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from pg.graph import load_dataset, FEATURE_SETS
from pg.engine import run_fold

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "data/policies_v2"
CSV = sys.argv[2] if len(sys.argv) > 2 else "data/labels_v2.csv"
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5

df = pd.read_csv(CSV)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(range(len(df)), df["label"].tolist(), df["family"].tolist()))

print(f"{SEEDS} seeds (42..{42 + SEEDS - 1}), {len(splits)} folds each; "
      f"mean +/- std across seeds of the fold-averaged score\n")
fold_f1 = {}
for fs in FEATURE_SETS:
    data = load_dataset(FOLDER, CSV, fs)
    per_seed, per_fold = [], []
    for s in range(SEEDS):
        torch.manual_seed(42 + s)
        res = [run_fold(data, tr, te, verbose=False)[1] for tr, te in splits]
        per_seed.append(np.mean(res, axis=0))
        per_fold.append([r[2] for r in res])
    per_seed = np.array(per_seed)
    m, sd = per_seed.mean(0), per_seed.std(0)
    fold_f1[fs] = np.mean(per_fold, axis=0)
    print(f"{fs:<6} ({data[0].x.size(1):>2} features)  "
          f"precision {m[0]:.3f}+/-{sd[0]:.3f}  "
          f"recall {m[1]:.3f}+/-{sd[1]:.3f}  "
          f"F1 {m[2]:.3f}+/-{sd[2]:.3f}")

print("\nF1 per fold (mean over seeds):")
for fs in FEATURE_SETS:
    print(f"  {fs:<6}", "  ".join(f"{x:.3f}" for x in fold_f1[fs]))
