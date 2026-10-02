import sys

import torch
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from pg.graph import load_dataset
from pg.engine import run_fold

torch.manual_seed(42)

# usage: python -m pg.train [folder] [labels.csv] [base|kinds]
FOLDER = sys.argv[1] if len(sys.argv) > 1 else "data/policies"
CSV = sys.argv[2] if len(sys.argv) > 2 else "data/labels.csv"
FEATURES = sys.argv[3] if len(sys.argv) > 3 else "base"
data = load_dataset(FOLDER, CSV, FEATURES)
df = pd.read_csv(CSV)
labels = df["label"].tolist()
groups = df["family"].tolist()


if __name__ == "__main__":
    print(f"Feature set: {FEATURES}  ({data[0].x.size(1)} features per node)\n")
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    best_model, best_f1 = None, -1

    for fold, (tr, te) in enumerate(skf.split(range(len(data)), labels, groups)):
        print(f"Fold {fold + 1}  (train {len(tr)}, test {len(te)})")
        model, (p, r, f) = run_fold(data, tr, te)
        print(f"  precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}\n")
        scores.append((p, r, f))
        if f > best_f1:
            best_model, best_f1 = model, f

    n = len(scores)
    print("MEAN over folds:")
    print(f"  precision {sum(s[0] for s in scores) / n:.3f}")
    print(f"  recall    {sum(s[1] for s in scores) / n:.3f}")
    print(f"  F1        {sum(s[2] for s in scores) / n:.3f}")

    torch.save(best_model.state_dict(), "model.pt")
    print("\nSaved model.pt")
