from pathlib import Path
import torch
import pandas as pd
import pickle

from src.app.config import get_config
from src.app.logger import get_logger
from src.model.gat import HeteroGATEncoder
from src.model.sage import HeteroSAGEEncoder  
from src.model.gcn import HeteroGCNEncoder  
from src.model.lightgcn import LightGCNEncoder

logger = get_logger(__name__)
cfg = get_config()

PROCESSED_DIR = Path(cfg.data.processed_dir)
GRAPH_PATH = PROCESSED_DIR / "hetero_graph.pt"

MODEL_PATH = PROCESSED_DIR / f"{cfg.model.type}_model.pt"

MAPPINGS_PATH = PROCESSED_DIR / "mappings.pkl"

USER_EMB_PATH = PROCESSED_DIR / "user_embeddings.parquet"
ITEM_EMB_PATH = PROCESSED_DIR / "item_embeddings.parquet"


def build_encoder(user_in: int, item_in: int, device: torch.device, data: dict):
    if cfg.model.type == "gat":
        logger.info("Export: Using HeteroGATEncoder")
        model = HeteroGATEncoder(
            in_channels_user=user_in,
            in_channels_item=item_in,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
            heads=cfg.model.heads,
            dropout=cfg.model.dropout,
            edge_dim=cfg.model.edge_dim,
        ).to(device)

    elif cfg.model.type == "sage":
        logger.info("Export: Using HeteroSAGEEncoder")
        model = HeteroSAGEEncoder(
            in_channels_user=user_in,
            in_channels_item=item_in,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        ).to(device)

    elif cfg.model.type == "gcn":
        logger.info("Export: Using HeteroGCNEncoder")
        model = HeteroGCNEncoder(
            in_channels_user=user_in,
            in_channels_item=item_in,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        ).to(device)

    elif cfg.model.type == "lightgcn":
        logger.info("Export: Using LightGCNEncoder")
        num_users = data["user"].x.size(0)
        num_items = data["item"].x.size(0)
        model = LightGCNEncoder(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
        ).to(device)

    

    else:
        raise ValueError(f"Unknown model.type: {cfg.model.type}")

    return model


def main():
    logger.info("Loading hetero graph...")
    data = torch.load(GRAPH_PATH, weights_only=False)

    user_in = data["user"].x.size(1)
    item_in = data["item"].x.size(1)
    logger.info(f"user_in_dim={user_in}, item_in_dim={item_in}")


    if cfg.training.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.training.device)

    logger.info(f"Using device: {device}")


    model = build_encoder(user_in, item_in, device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    logger.info(f"Loading trained weights from {MODEL_PATH} ...")
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    data = data.to(device)

    with torch.no_grad():
        user_emb, item_emb = model(data)

    user_emb = user_emb.cpu()
    item_emb = item_emb.cpu()


    with MAPPINGS_PATH.open("rb") as f:
        mappings = pickle.load(f)

    id2user: dict[int, str] = mappings["id2user"]
    id2item: dict[int, str] = mappings["id2item"]

    num_users = user_emb.size(0)
    num_items = item_emb.size(0)

    user_df = pd.DataFrame(
        user_emb.numpy(),
        columns=[f"dim_{i}" for i in range(user_emb.size(1))],
    )
    user_df.insert(0, "user_idx", range(num_users))
    user_df["user_id"] = user_df["user_idx"].map(id2user)

    item_df = pd.DataFrame(
        item_emb.numpy(),
        columns=[f"dim_{i}" for i in range(item_emb.size(1))],
    )
    item_df.insert(0, "item_idx", range(num_items))
    item_df["item_id"] = item_df["item_idx"].map(id2item)

    user_df.to_parquet(USER_EMB_PATH, index=False)
    item_df.to_parquet(ITEM_EMB_PATH, index=False)

    logger.info(f"Saved user embeddings → {USER_EMB_PATH}")
    logger.info(f"Saved item embeddings → {ITEM_EMB_PATH}")
    logger.info("Export done.")


if __name__ == "__main__":
    main()