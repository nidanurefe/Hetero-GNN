from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import LGConv


class LightGCNEncoder(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        # Learnable embeddings for users/items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # LightGCN uses simple neighborhood aggregation
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

        # Init embeddings (xavier uniform)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        device = data["user"].x.device

        num_users = self.num_users
        num_items = self.num_items

        # Hetero edge_index: (user_idx, item_idx) with indices already
        # in [0, num_users-1] and [0, num_items-1]
        edge_index = data["user", "rates", "item"].edge_index.to(device)

        # Build homogeneous edge_index (users + items in same space)
        edge_index_homo = edge_index.clone()
        edge_index_homo[1] += num_users  # shift item indices

        # Make it undirected, add reverse edges
        edge_index_rev = edge_index_homo.flip(0)
        edge_index_all = torch.cat([edge_index_homo, edge_index_rev], dim=1)

        # Initial node embeddings: concat [user_emb; item_emb]
        user_emb0 = self.user_embedding.weight   # [U, d]
        item_emb0 = self.item_embedding.weight   # [I, d]
        x = torch.cat([user_emb0, item_emb0], dim=0)  # [U+I, d]

        # LightGCN: average of embeddings from each layer (including 0-th)
        outs = [x]
        for conv in self.convs:
            x = conv(x, edge_index_all)
            outs.append(x)

        x_final = torch.stack(outs, dim=0).mean(dim=0)  # [U+I, d]

        # Split back to user / item embeddings
        user_emb = x_final[:num_users]
        item_emb = x_final[num_users : num_users + num_items]

        # L2 normalize for cosine-similarity friendliness
        user_emb = F.normalize(user_emb, p=2, dim=-1)
        item_emb = F.normalize(item_emb, p=2, dim=-1)

        return user_emb, item_emb