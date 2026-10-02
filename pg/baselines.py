import sys, json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from pg.graph import as_list, is_dangerous

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "data/policies"
CSV = sys.argv[2] if len(sys.argv) > 2 else "data/labels.csv"

df = pd.read_csv(CSV)
policies = [json.load(open(os.path.join(FOLDER, f))) for f in df["filename"]]
y = df["label"].values
groups = df["family"].values

PASSROLE = "iam:passrole"
COMPUTE = {"lambda:createfunction", "ec2:runinstances",
           "cloudformation:createstack", "glue:createdevendpoint"}
STANDALONE = {
    "iam:createpolicyversion", "iam:setdefaultpolicyversion",
    "iam:attachuserpolicy", "iam:attachrolepolicy", "iam:putuserpolicy",
    "iam:putrolepolicy", "iam:createaccesskey", "iam:createloginprofile",
    "iam:updateloginprofile", "lambda:updatefunctioncode",
}


def actions_of(p):
    acts = set()
    for st in as_list(p["Statement"]):
        if st.get("Effect") == "Allow":
            acts.update(a.lower() for a in as_list(st["Action"]))
    return acts


def rule_naive(p):
    """Flag anything containing a known-dangerous action."""
    return int(any(is_dangerous(a) for a in actions_of(p)))


def rule_combo(p):
    """Smarter rule: wildcards, standalone escalators, or PassRole + compute.
    It ignores WHICH resource each action applies to."""
    acts = actions_of(p)
    if any("*" in a and is_dangerous(a) for a in acts):
        return 1
    if acts & STANDALONE:
        return 1
    if PASSROLE in acts and acts & COMPUTE:
        return 1
    return 0


# Bag-of-actions: one column per action seen anywhere in the dataset
all_actions = sorted(set().union(*[actions_of(p) for p in policies]))
X = np.array([[a in actions_of(p) for a in all_actions] for p in policies], dtype=float)


def prf(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return p, r, f


def report(name, p, r, f):
    print(f"{name:<30} precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}")


skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(X, y, groups))


def fold_mean(preds):
    """Average of the five per-fold scores (the way the GNN is reported)."""
    return tuple(np.mean([prf(y[te], preds[te]) for _, te in splits], axis=0))


def cv_preds(make):
    preds = np.zeros(len(y), dtype=int)
    for tr, te in splits:
        preds[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return preds


methods = [
    ("Rule: any dangerous action", np.array([rule_naive(p) for p in policies])),
    ("Rule: combo, scope-blind", np.array([rule_combo(p) for p in policies])),
    ("Logistic regression (bag)", cv_preds(lambda: LogisticRegression(max_iter=1000))),
    ("Random forest (bag)", cv_preds(lambda: RandomForestClassifier(n_estimators=200, random_state=0))),
]

print("Pooled over all policies (rules need no training; the models use out-of-fold predictions)")
for name, preds in methods:
    report(name, *prf(y, preds))

print("\nMean of the 5 per-fold scores, same folds as the GNN (use this for GNN comparisons)")
for name, preds in methods:
    report(name, *fold_mean(preds))
