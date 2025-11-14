from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.data import HeteroData


# GAT encoder for user/item nodes in the bipartite graph
class HeteroGATEncoder(nn.Module):

    def __init__(
        self,
        in_channels_user: int,
        in_channels_item: int,
        hidden_channels: int = 32,
        out_channels: int = 32,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
        edge_dim: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        # First layer
        conv0 = HeteroConv(
            {
                ("user", "rates", "item"): GATConv(
                    (in_channels_user, in_channels_item),
                    hidden_channels,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=False,
                ),
                ("item", "rev_rates", "user"): GATConv(
                    (in_channels_item, in_channels_user),
                    hidden_channels,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=False,
                ),
            },
            aggr="sum",
        )
        self.convs.append(conv0)

        in_dim = hidden_channels * heads
        for _ in range(num_layers - 1):
            conv = HeteroConv(
                {
                    ("user", "rates", "item"): GATConv(
                        (in_dim, in_dim),
                        hidden_channels,
                        heads=heads,
                        edge_dim=edge_dim,
                        dropout=dropout,
                        add_self_loops=False,
                    ),
                    ("item", "rev_rates", "user"): GATConv(
                        (in_dim, in_dim),
                        hidden_channels,
                        heads=heads,
                        edge_dim=edge_dim,
                        dropout=dropout,
                        add_self_loops=False,
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        out_in_dim = hidden_channels * heads
        self.user_proj = nn.Linear(out_in_dim, out_channels)
        self.item_proj = nn.Linear(out_in_dim, out_channels)

    def forward(self, data: HeteroData):
        x_dict: Dict[str, torch.Tensor] = {
            "user": data["user"].x,
            "item": data["item"].x,
        }

        edge_index_dict = {
            ("user", "rates", "item"): data["user", "rates", "item"].edge_index,
            # reverse 
            ("item", "rev_rates", "user"): data["user", "rates", "item"].edge_index.flip(0),
        }

        edge_attr_base = data["user", "rates", "item"].edge_attr
        edge_attr_dict = {
            ("user", "rates", "item"): edge_attr_base,
            ("item", "rev_rates", "user"): edge_attr_base,  
        }


        for layer, conv in enumerate(self.convs):
            x_dict = conv(
                x_dict,
                edge_index_dict,
                edge_attr_dict=edge_attr_dict,  
            )
            # GAT output (N, heads * out_channels)
            x_dict = {k: F.elu(v) for k, v in x_dict.items()}
            if layer < self.num_layers - 1:
                x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                          for k, v in x_dict.items()}

        user_emb = self.user_proj(x_dict["user"])
        item_emb = self.item_proj(x_dict["item"])

        # Normalize embeddings 
        user_emb = F.normalize(user_emb, p=2, dim=-1)
        item_emb = F.normalize(item_emb, p=2, dim=-1)

        return user_emb, item_emb


def bpr_loss(user_emb, pos_emb, neg_emb, reg_lambda: float = 1e-4):
    
    # BPR loss: log-sigmoid( s(u, i+) - s(u, i-) )
    # s(u, i) = dot product
    
    pos_scores = (user_emb * pos_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_emb).sum(dim=-1)

    loss = -F.logsigmoid(pos_scores - neg_scores).mean()

    # L2 regularization
    reg = (
        user_emb.norm(2, dim=-1).pow(2)
        + pos_emb.norm(2, dim=-1).pow(2)
        + neg_emb.norm(2, dim=-1).pow(2)
    ).mean()
    loss = loss + reg_lambda * reg

    return loss



def bpr_softplus_loss(user_emb, pos_emb, neg_emb, reg_lambda: float = 1e-4):
    pos_scores = (user_emb * pos_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_emb).sum(dim=-1)

    margin = pos_scores - neg_scores
    loss = torch.nn.functional.softplus(-margin).mean()

    reg = (
        user_emb.norm(2, dim=-1).pow(2)
        + pos_emb.norm(2, dim=-1).pow(2)
        + neg_emb.norm(2, dim=-1).pow(2)
    ).mean()

    return loss + reg_lambda * reg