import sys

import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_fscore_support

from pg.graph import load_dataset
from pg.model import PolicyGAT

torch.manual_seed(42)

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "data/policies"
CSV = sys.argv[2] if len(sys.argv) > 2 else "data/labels.csv"
data = load_dataset(FOLDER, CSV)
df = pd.read_csv(CSV)
labels = df["label"].tolist()
groups = df["family"].tolist()


def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            ps += model(b).argmax(1).tolist()
            ys += b.y.tolist()
    p, r, f, _ = precision_recall_fscore_support(
        ys, ps, average="binary", zero_division=0
    )
    return p, r, f


def run_fold(train_idx, test_idx, epochs=60):
    train_set = [data[i] for i in train_idx]
    test_set = [data[i] for i in test_idx]

    counts = torch.bincount(torch.tensor([int(d.y) for d in train_set]), minlength=2)
    weights = counts.sum() / (2 * counts.float())

    model = PolicyGAT(in_dim=data[0].x.size(1), edge_dim=data[0].edge_attr.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for b in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(b), b.y)
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  loss {total / len(train_loader):.4f}")

    return model, evaluate(model, test_loader)


if __name__ == "__main__":
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    best_model, best_f1 = None, -1

    for fold, (tr, te) in enumerate(skf.split(range(len(data)), labels, groups)):
        print(f"Fold {fold + 1}  (train {len(tr)}, test {len(te)})")
        model, (p, r, f) = run_fold(tr, te)
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
