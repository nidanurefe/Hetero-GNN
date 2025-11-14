from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.app.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
ITEM_EMB_PATH = PROCESSED_DIR / "item_embeddings.parquet" 
INTERACTIONS_PATH = PROCESSED_DIR / "interactions.parquet"


def load_data():
    logger.info(f"Loading item embeddings from {ITEM_EMB_PATH} ...")
    emb_df = pd.read_parquet(ITEM_EMB_PATH)

    if "item_idx" not in emb_df.columns:
        emb_df = emb_df.reset_index().rename(columns={"index": "item_idx"})

    # select only numeric embedding columns excluding item_idx
    numeric_cols = [
        c for c in emb_df.columns
        if c != "item_idx" and np.issubdtype(emb_df[c].dtype, np.number)
    ]
    if not numeric_cols:
        raise ValueError("No numeric embedding columns found in item_embeddings.parquet")

    item_idx_arr = emb_df["item_idx"].to_numpy()
    item_emb = emb_df[numeric_cols].to_numpy().astype("float32")
    item_emb = torch.from_numpy(item_emb)  # [num_items, d]

    logger.info(f"item_emb shape = {item_emb.shape}")

    logger.info(f"Loading interactions from {INTERACTIONS_PATH} ...")
    inter = pd.read_parquet(INTERACTIONS_PATH)

    # item_idx -> (item_id, domain) map
    item_meta = (
        inter[["item_idx", "item_id", "domain"]]
        .drop_duplicates("item_idx")
        .set_index("item_idx")
    )

    return item_emb, item_idx_arr, item_meta


def build_tsne(item_emb, item_idx_arr, item_meta, max_points: int = 2000):
    num_items = item_emb.size(0)
    logger.info(f"Total items: {num_items}")

    if num_items <= max_points:
        sample_pos = np.arange(num_items)
    else:
        sample_pos = np.random.choice(num_items, size=max_points, replace=False)

    logger.info(f"Sampling {len(sample_pos)} items for visualization...")

    emb_sample = item_emb[sample_pos].numpy()  # [N, d]
    sampled_item_idx = item_idx_arr[sample_pos]

    # find domain for each sampled item
    domains = []
    for idx in sampled_item_idx:
        if idx in item_meta.index:
            domains.append(item_meta.loc[idx, "domain"])
        else:
            domains.append("unknown")
    domains = np.array(domains)

    logger.info("Running t-SNE (this may take a bit)...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
    )
    X_2d = tsne.fit_transform(emb_sample)  # [N, 2]

    return X_2d, domains

def plot_tsne(X_2d, domains):
    plt.figure(figsize=(8, 8))

    unique_domains = np.unique(domains)

    # Domain -> color map
    color_map = {
        "magazine": "tab:blue",
        "appliances": "tab:orange",
        "movies": "tab:green",
        "unknown": "tab:gray",
    }

    for d in unique_domains:
        mask = domains == d
        color = color_map.get(d, "tab:gray")
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=8,
            alpha=0.7,
            label=d,
            c=color,
        )

    plt.title("Item Embeddings t-SNE (colored by domain)", fontsize=12)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    item_emb, item_idx_arr, item_meta = load_data()
    X_2d, domains = build_tsne(item_emb, item_idx_arr, item_meta, max_points=2000)
    plot_tsne(X_2d, domains)


if __name__ == "__main__":
    main()