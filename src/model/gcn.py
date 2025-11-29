from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GraphConv
from torch_geometric.data import HeteroData


class HeteroGCNEncoder(nn.Module):

    def __init__(
        self,
        in_channels_user: int,
        in_channels_item: int,
        hidden_channels: int = 32,
        out_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        conv0 = HeteroConv(
            {
                ("user", "rates", "item"): GraphConv(
                    in_channels_user,
                    hidden_channels,
                    aggr="mean",
                ),
                ("item", "rev_rates", "user"): GraphConv(
                    in_channels_item,
                    hidden_channels,
                    aggr="mean",
                ),
            },
            aggr="sum",
        )
        self.convs.append(conv0)

        for _ in range(num_layers - 1):
            conv = HeteroConv(
                {
                    ("user", "rates", "item"): GraphConv(
                        hidden_channels,
                        hidden_channels,
                        aggr="mean",
                    ),
                    ("item", "rev_rates", "user"): GraphConv(
                        hidden_channels,
                        hidden_channels,
                        aggr="mean",
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.user_proj = nn.Linear(hidden_channels, out_channels)
        self.item_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, data: HeteroData):
        x_dict: Dict[str, torch.Tensor] = {
            "user": data["user"].x,
            "item": data["item"].x,
        }

        edge_index_dict = {
            ("user", "rates", "item"): data["user", "rates", "item"].edge_index,
            ("item", "rev_rates", "user"): data["user", "rates", "item"].edge_index.flip(0),
        }

        for layer, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            if layer < self.num_layers - 1:
                x_dict = {
                    k: F.dropout(v, p=self.dropout, training=self.training)
                    for k, v in x_dict.items()
                }

        user_emb = self.user_proj(x_dict["user"])
        item_emb = self.item_proj(x_dict["item"])

        user_emb = F.normalize(user_emb, p=2, dim=-1)
        item_emb = F.normalize(item_emb, p=2, dim=-1)

        return user_emb, item_emb