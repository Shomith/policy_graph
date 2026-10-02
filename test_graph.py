import json
from pg.graph import policy_to_nx, nx_to_data, load_dataset

policy = json.load(open("data/policies/passrole_lambda_0.json"))
g = policy_to_nx(policy, "passrole_lambda_0")

print("NODES:")
for n, attrs in g.nodes(data=True):
    print(" ", n, "->", attrs["ntype"])

data = nx_to_data(g, 1)
print("\nx shape:", data.x.shape)
print("edge_index shape:", data.edge_index.shape)
print("dangerous flags:", data.x[:, 5].tolist())

ds = load_dataset("data/policies", "data/labels.csv")
print("\nLoaded", len(ds), "graphs")
