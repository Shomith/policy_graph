import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class PolicyGAT(torch.nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=64, heads=4, layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.convs.append(
                GATv2Conv(d, hidden, heads=heads, edge_dim=edge_dim, concat=False)
            )
            d = hidden

        # mean + max pooled vectors are concatenated, hence hidden * 2
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, data):
        x, ei, ea, b = data.x, data.edge_index, data.edge_attr, data.batch
        for conv in self.convs:
            x = F.elu(conv(x, ei, ea))
            x = F.dropout(x, 0.2, self.training)
        g = torch.cat([global_mean_pool(x, b), global_max_pool(x, b)], dim=1)
        return self.head(g)
