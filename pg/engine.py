"""Shared training / evaluation code, used by train.py, analyze.py and ablate.py."""
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from pg.model import PolicyGAT


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


def predict(model, graphs):
    """Predicted class for each graph, in order."""
    model.eval()
    out = []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=16):
            out += model(b).argmax(1).tolist()
    return out


def run_fold(dataset, train_idx, test_idx, epochs=60, verbose=True):
    """Train a fresh PolicyGAT on train_idx and score it on test_idx."""
    train_set = [dataset[i] for i in train_idx]
    test_set = [dataset[i] for i in test_idx]

    counts = torch.bincount(torch.tensor([int(d.y) for d in train_set]), minlength=2)
    weights = counts.sum() / (2 * counts.float())

    model = PolicyGAT(in_dim=dataset[0].x.size(1), edge_dim=dataset[0].edge_attr.size(1))
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
        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  loss {total / len(train_loader):.4f}")

    return model, evaluate(model, test_loader)
