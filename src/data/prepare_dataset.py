import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData


class EdgePredictionDataset(Dataset):

    # Positive (u, i_pos) and negative (u, i_neg) samples for BPR-style (Bayesian Personalized Ranking) training.
    def __init__(self, data: HeteroData, split: str = "train"):
        assert split in ["train", "val", "test"]

        self.data = data
        edge_index = data["user", "rates", "item"].edge_index
        mask = data["user", "rates", "item"][f"{split}_mask"]

        # Positive interactions
        self.pos_u = edge_index[0][mask]
        self.pos_i = edge_index[1][mask]

        self.num_users = data["user"].x.size(0)
        self.num_items = data["item"].x.size(0)

    def __len__(self):
        return len(self.pos_u)

    def __getitem__(self, idx):
        u = self.pos_u[idx].item()
        pos_i = self.pos_i[idx].item()

        # Simple negative sample: random item
        neg_i = random.randrange(self.num_items)

        return {
            "user": torch.tensor(u, dtype=torch.long),
            "pos_item": torch.tensor(pos_i, dtype=torch.long),
            "neg_item": torch.tensor(neg_i, dtype=torch.long),
        }


def get_dataloader(data: HeteroData, split: str, batch_size=512, shuffle=True):
    dataset = EdgePredictionDataset(data, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)