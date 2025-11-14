from pathlib import Path
import torch
import pandas as pd
import pickle

from src.app.config import get_config
from src.app.logger import get_logger
from src.model.gat import HeteroGATEncoder

logger = get_logger(__name__)
cfg = get_config()

PROCESSED_DIR = Path(cfg.data.processed_dir)  
GRAPH_PATH = PROCESSED_DIR / "hetero_graph.pt"
MODEL_PATH = PROCESSED_DIR / "gat_model.pt"
MAPPINGS_PATH = PROCESSED_DIR / "mappings.pkl"

USER_EMB_PATH = PROCESSED_DIR / "user_embeddings.parquet"
ITEM_EMB_PATH = PROCESSED_DIR / "item_embeddings.parquet"


def main():
    logger.info("Loading hetero graph...")
    data = torch.load(GRAPH_PATH, weights_only=False)

    user_in = data["user"].x.size(1)
    item_in = data["item"].x.size(1)
    logger.info(f"user_in_dim={user_in}, item_in_dim={item_in}")

    # select device
    if cfg.training.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.training.device)

    logger.info(f"Using device: {device}")

    # Initialize model
    model = HeteroGATEncoder(
        in_channels_user=user_in,
        in_channels_item=item_in,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels,
        num_layers=cfg.model.num_layers,
        heads=cfg.model.heads,
        dropout=cfg.model.dropout,
    ).to(device)

    # Load trained model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    data = data.to(device)

    with torch.no_grad():
        user_emb, item_emb = model(data)

    user_emb = user_emb.cpu()
    item_emb = item_emb.cpu()

    # idx -> id mapping
    with MAPPINGS_PATH.open("rb") as f:
        mappings = pickle.load(f)

    id2user: dict[int, str] = mappings["idx2user"]
    id2item: dict[int, str] = mappings["idx2item"]

    num_users = user_emb.size(0)
    num_items = item_emb.size(0)

    # User embedding DF
    user_df = pd.DataFrame(
        user_emb.numpy(),
        columns=[f"dim_{i}" for i in range(user_emb.size(1))],
    )
    user_df.insert(0, "user_idx", range(num_users))
    user_df["user_id"] = user_df["user_idx"].map(id2user)

    # Item embedding DF
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