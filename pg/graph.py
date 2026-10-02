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


def as_list(x):
    """IAM lets Action/Resource be a string or a list. Always return a list."""
    return x if isinstance(x, list) else [x]


def is_dangerous(action):
    """True if the action (including wildcards like iam:*) covers a dangerous one."""
    a = action.lower()
    return any(fnmatch.fnmatch(d, a) for d in DANGEROUS)


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


def node_features(attrs):
    """7 features: 4 one-hot node type + wildcard + dangerous + admin."""
    t = [0.0] * len(NTYPES)
    t[NTYPES.index(attrs["ntype"])] = 1.0

    name = attrs.get("name", "")
    wild = float("*" in name)
    danger = 0.0
    admin = 0.0
    if attrs["ntype"] == "action":
        danger = float(is_dangerous(name))
        admin = float(name.lower() in ("*", "*:*"))
    return t + [wild, danger, admin]


def nx_to_data(g, label):
    """Convert a NetworkX graph into a PyTorch Geometric Data object."""
    idx = {n: i for i, n in enumerate(g.nodes)}
    x = torch.tensor([node_features(g.nodes[n]) for n in g.nodes], dtype=torch.float)

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


def load_dataset(folder, labels_csv):
    """Load every policy listed in labels.csv and return a list of Data objects."""
    df = pd.read_csv(labels_csv)
    out = []
    for _, row in df.iterrows():
        path = os.path.join(folder, row["filename"])
        with open(path) as f:
            pol = json.load(f)
        g = policy_to_nx(pol, row["filename"])
        out.append(nx_to_data(g, int(row["label"])))
    return out
