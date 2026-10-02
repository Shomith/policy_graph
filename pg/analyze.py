import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from pg.train import data, df, labels, groups, run_fold

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(data), dtype=int)

for fold, (tr, te) in enumerate(skf.split(range(len(data)), labels, groups)):
    model, _ = run_fold(tr, te)
    model.eval()
    loader = DataLoader([data[i] for i in te], batch_size=16)
    out = []
    with torch.no_grad():
        for b in loader:
            out += model(b).argmax(1).tolist()
    preds[te] = out

df["correct"] = preds == df["label"].values
summary = df.groupby("family").agg(
    label=("label", "first"),
    n=("correct", "size"),
    accuracy=("correct", "mean"),
)
print(summary.sort_values("accuracy").to_string())
