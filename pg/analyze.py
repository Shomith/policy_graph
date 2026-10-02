import sys

import torch
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from pg.train import data, df, labels, groups, FEATURES
from pg.engine import run_fold, predict

# usage: python -m pg.analyze [folder] [labels.csv] [base|kinds] [n_seeds]
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 1

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(range(len(data)), labels, groups))
correct = np.zeros((SEEDS, len(data)))

for s in range(SEEDS):
    torch.manual_seed(42 + s)
    for tr, te in splits:
        model, _ = run_fold(data, tr, te, verbose=False)
        preds = predict(model, [data[i] for i in te])
        correct[s, te] = np.array(preds) == df["label"].values[te]

df = df.copy()
df["accuracy"] = correct.mean(axis=0)  # averaged over seeds
summary = df.groupby("family").agg(
    label=("label", "first"), n=("accuracy", "size"), accuracy=("accuracy", "mean"),
)
print(f"Feature set: {FEATURES}   seeds: {SEEDS}\n")
print(summary.sort_values("accuracy").to_string())
