import json
import os
import fnmatch

import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

NTYPES = ["policy", "action", "resource", "principal"]
ETYPES = ["allows", "applies_to", "can_assume", "attached_to"]

# Actions known to enable privilege escalation (lowercase)
DANGEROUS = {
    "iam:passrole", "iam:createpolicyversion", "iam:setdefaultpolicyversion",
    "iam:attachuserpolicy", "iam:attachrolepolicy", "iam:putuserpolicy",
    "iam:putrolepolicy", "iam:createaccesskey", "iam:createloginprofile",
    "iam:updateloginprofile", "sts:assumerole", "lambda:createfunction",
    "lambda:updatefunctioncode", "ec2:runinstances",
    "cloudformation:createstack", "glue:createdevendpoint",
    "datapipeline:createpipeline",
}

# What each escalation-capable action DOES. Used by the optional "kinds" feature
# set so the model can tell PassRole apart from PutRolePolicy, instead of seeing
# one identical "dangerous" flag for both.
KINDS = {
    "passes_role": {"iam:passrole"},
    "creates_compute": {
        "lambda:createfunction", "ec2:runinstances", "cloudformation:createstack",
        "glue:createdevendpoint", "datapipeline:createpipeline",
    },
    "edits_permissions": {
        "iam:createpolicyversion", "iam:setdefaultpolicyversion",
        "iam:attachuserpolicy", "iam:attachrolepolicy",
        "iam:putuserpolicy", "iam:putrolepolicy",
    },
    "makes_credentials": {
        "iam:createaccesskey", "iam:createloginprofile", "iam:updateloginprofile",
    },
    "edits_code": {"lambda:updatefunctioncode"},
    "assumes_role": {"sts:assumerole"},
}
KIND_NAMES = list(KINDS)
FEATURE_SETS = ("base", "kinds")

# every dangerous action must belong to exactly one kind
assert set().union(*KINDS.values()) == DANGEROUS


def as_list(x):
    """IAM lets Action/Resource be a string or a list. Always return a list."""
    return x if isinstance(x, list) else [x]


def is_dangerous(action):
    """True if the action (including wildcards like iam:*) covers a dangerous one."""
    a = action.lower()
    return any(fnmatch.fnmatch(d, a) for d in DANGEROUS)


def action_kinds(action):
    """One 0/1 flag per kind. Wildcards expand, so iam:* sets several kinds."""
    a = action.lower()
    return [float(any(fnmatch.fnmatch(m, a) for m in KINDS[k])) for k in KIND_NAMES]


def policy_to_nx(policy, name="policy"):
    """Turn one IAM policy dict into a NetworkX graph."""
    g = nx.MultiDiGraph()
    pol = f"policy:{name}"
    g.add_node(pol, ntype="policy", name=name)

    for st in as_list(policy.get("Statement", [])):
        if st.get("Effect") != "Allow":
            continue
        for action in as_list(st.get("Action", [])):
            a = f"action:{action}"
            g.add_node(a, ntype="action", name=action)
            g.add_edge(pol, a, etype="allows")
            for res in as_list(st.get("Resource", ["*"])):
                r = f"resource:{res}"
                g.add_node(r, ntype="resource", name=res)
                g.add_edge(a, r, etype="applies_to")
    return g


def node_features(attrs, features="base"):
    """Base: 7 features = 4 one-hot node type + wildcard + dangerous + admin.
    "kinds" appends 6 more: what kind of action the node is (0 for non-actions)."""
    if features not in FEATURE_SETS:
        raise ValueError(f"features must be one of {FEATURE_SETS}, got {features!r}")
    t = [0.0] * len(NTYPES)
    t[NTYPES.index(attrs["ntype"])] = 1.0

    name = attrs.get("name", "")
    wild = float("*" in name)
    danger = 0.0
    admin = 0.0
    kinds = [0.0] * len(KIND_NAMES)
    if attrs["ntype"] == "action":
        danger = float(is_dangerous(name))
        admin = float(name.lower() in ("*", "*:*"))
        kinds = action_kinds(name)
    out = t + [wild, danger, admin]
    return out + kinds if features == "kinds" else out


def nx_to_data(g, label, features="base"):
    """Convert a NetworkX graph into a PyTorch Geometric Data object."""
    idx = {n: i for i, n in enumerate(g.nodes)}
    x = torch.tensor([node_features(g.nodes[n], features) for n in g.nodes],
                     dtype=torch.float)

    src, dst, et = [], [], []
    for u, v, d in g.edges(data=True):
        e = ETYPES.index(d["etype"])
        # add both directions so information can flow both ways
        src += [idx[u], idx[v]]
        dst += [idx[v], idx[u]]
        et += [e, e]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = F.one_hot(torch.tensor(et, dtype=torch.long), len(ETYPES)).float()
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([label]))


def load_dataset(folder, labels_csv, features="base"):
    """Load every policy listed in labels.csv and return a list of Data objects."""
    df = pd.read_csv(labels_csv)
    out = []
    for _, row in df.iterrows():
        path = os.path.join(folder, row["filename"])
        with open(path) as f:
            pol = json.load(f)
        g = policy_to_nx(pol, row["filename"])
        out.append(nx_to_data(g, int(row["label"]), features))
    return out
