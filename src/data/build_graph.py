import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch 
from torch_geometric.data import HeteroData   

from src.app.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DATA_DIR = Path("data/processed")
INTERACTIONS_PATH = PROCESSED_DATA_DIR / "interactions.parquet"
USER_FEAT_PATH = PROCESSED_DATA_DIR / "user_features.parquet"
ITEM_FEAT_PATH = PROCESSED_DATA_DIR / "item_features.parquet"

GRAPH_OUTPUT_PATH = PROCESSED_DATA_DIR / "hetero_graph.pt"


# Function to normalize edge features
def normalize_series(s: pd.Series) -> np.ndarray:
    s = s.astype(float)
    min_val = s.min()
    max_val = s.max()
    if max_val - min_val == 0:
        return np.zeros_like(s, dtype=float)
    return ((s - min_val) / (max_val - min_val)).to_numpy()


def build_graph():
    logger.info("Loading processed interaction and feature files..")
    interactions = pd.read_parquet(INTERACTIONS_PATH)
    
    user_df = pd.read_parquet(USER_FEAT_PATH)
    item_df = pd.read_parquet(ITEM_FEAT_PATH)

    logger.info(f"Loaded {len(interactions):,} interactions.")
    logger.info(f"Users: {len(user_df):,} | Items: {len(item_df):,}")

    user_df = user_df.sort_values("user_idx").reset_index(drop=True)
    item_df = item_df.sort_values("item_idx").reset_index(drop=True)


    # Node features
    user_features = user_df.filter(regex="norm$|rating_std|verified_ratio|avg_rating").copy()
    item_features = item_df.filter(regex="norm$|rating_std|verified_ratio|avg_rating").copy()

    user_feature_cols = [c for c in user_df.columns
                         if c not in ("user_idx", "user_id")]
    item_feature_cols = [c for c in item_df.columns
                         if c not in ("item_idx", "item_id")]

    user_x = torch.tensor(user_df[user_feature_cols].to_numpy(), dtype=torch.float32)
    item_x = torch.tensor(item_df[item_feature_cols].to_numpy(), dtype=torch.float32)


    logger.info(f"user.x shape = {user_x.shape}")
    logger.info(f"item.x shape = {item_x.shape}")

    # Edge index 
    if interactions["user_idx"].isna().any() or interactions["item_idx"].isna().any():
        logger.error("NaN detected in user_idx or item_idx")
        raise ValueError("user_idx/item_idx contain NaNs.")

    src = torch.tensor(interactions["user_idx"].to_numpy(), dtype=torch.long)
    dst = torch.tensor(interactions["item_idx"].to_numpy(), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    logger.info(f"edge_index shape = {edge_index.shape}")

    # Edge features
    df_edge = interactions.copy()
    df_edge["rating_norm"] = (df_edge["rating"] - 1.0) / 4.0
    df_edge["time_days"] = df_edge["timestamp_ms"] / 1000 / 3600 / 24
    df_edge["time_norm"] = normalize_series(df_edge["time_days"])
    df_edge["helpful_log"] = np.log1p(df_edge["helpful_vote"]).astype(float)
    df_edge["verified_purchase"] = df_edge["verified_purchase"].astype(int)

    edge_features = torch.tensor(
        df_edge[["rating_norm", "time_norm", "verified_purchase", "helpful_log"]].to_numpy(),
        dtype=torch.float32,
    )

    logger.info(f"edge_attribute shape = {edge_features.shape}")

    # Build HeteroData object
    logger.info("Constructing HeteroData graph...")

    data = HeteroData()
    data["user"].x = user_x
    data["item"].x = item_x
    data["user", "rates", "item"].edge_index = edge_index
    data["user", "rates", "item"].edge_attr = edge_features

    # Save graph
    torch.save(data, GRAPH_OUTPUT_PATH)
    logger.info(f"Saved hetero graph → {GRAPH_OUTPUT_PATH}")




def main():
    logger.info("Building heterogenous bipartite graph...")
    build_graph()
    logger.info("Graph build complete.")


if __name__ == "__main__":
    main()





