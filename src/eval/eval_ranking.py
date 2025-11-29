from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import torch
import pandas as pd
import numpy as np

from src.app.config import get_config
from src.app.logger import get_logger
from src.model.gat import HeteroGATEncoder
from src.model.sage import HeteroSAGEEncoder
from src.model.gcn import HeteroGCNEncoder
from src.eval.metrics import precision_recall_at_k, ndcg_at_k

logger = get_logger(__name__)
cfg = get_config()

PROCESSED_DIR = Path(cfg.data.processed_dir)
GRAPH_PATH = PROCESSED_DIR / "hetero_graph.pt"


def build_encoder(user_in: int, item_in: int, device: torch.device):
    if cfg.model.type == "gat":
        logger.info("Eval: Using HeteroGATEncoder")
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
        logger.info("Eval: Using HeteroSAGEEncoder")
        model = HeteroSAGEEncoder(
            in_channels_user=user_in,
            in_channels_item=item_in,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        ).to(device)

    elif cfg.model.type == "gcn":
        logger.info("Eval: Using HeteroGCNEncoder")
        model = HeteroGCNEncoder(
            in_channels_user=user_in,
            in_channels_item=item_in,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model.type: {cfg.model.type}")

    return model


def eval_ranking_at_k(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    interactions: pd.DataFrame,
    split: str = "test",
    ks: List[int] | None = None,
) -> Dict[str, Dict[int, float]]:
    """
    user_emb: [num_users, d]
    item_emb: [num_items, d]
    interactions: interactions.parquet (user_idx, item_idx, split, ...)
    split: "val" veya "test"
    ks: örn: [5, 10, 20]
    """
    if ks is None:
        ks = [5, 10, 20]

    test_df = interactions[interactions["split"] == split].copy()
    if test_df.empty:
        raise ValueError(f"No rows found with split='{split}'")

    test_group = test_df.groupby("user_idx")["item_idx"].apply(list)

    train_val_df = interactions[interactions["split"].isin(["train", "val"])].copy()
    seen_group = train_val_df.groupby("user_idx")["item_idx"].apply(set)

    num_users = user_emb.size(0)
    num_items = item_emb.size(0)

    logger.info(
        f"Eval split='{split}': {len(test_df)} interactions, "
        f"{test_group.index.nunique()} users with test items."
    )

    sum_precision = {k: 0.0 for k in ks}
    sum_recall = {k: 0.0 for k in ks}
    sum_hr = {k: 0.0 for k in ks}
    sum_ndcg = {k: 0.0 for k in ks}
    user_count = 0

    item_emb = item_emb.to("cpu")
    user_emb = user_emb.to("cpu")

    all_item_indices = torch.arange(num_items)

    for u_idx, gt_items in test_group.items():
        u_idx = int(u_idx)
        if u_idx >= num_users:
            continue

        # User'ın test pozitifleri
        gt = list(set(gt_items))
        if not gt:
            continue

        user_count += 1

        u_vec = user_emb[u_idx]  # [d]
        scores = item_emb @ u_vec  # [num_items]

        seen_items = seen_group.get(u_idx, set())
        for it in seen_items:
            if it < num_items and it not in gt:
                scores[it] = -1e9

        _, ranked_items = torch.sort(scores, descending=True)
        ranked_items = ranked_items.cpu().numpy().tolist()

        for k in ks:
            prec, rec, hr = precision_recall_at_k(ranked_items, gt, k)
            ndcg = ndcg_at_k(ranked_items, gt, k)

            sum_precision[k] += prec
            sum_recall[k] += rec
            sum_hr[k] += hr
            sum_ndcg[k] += ndcg

    if user_count == 0:
        raise ValueError("No users with test positives found for evaluation.")

    metrics = {
        "precision": {k: sum_precision[k] / user_count for k in ks},
        "recall": {k: sum_recall[k] / user_count for k in ks},
        "hr": {k: sum_hr[k] / user_count for k in ks},
        "ndcg": {k: sum_ndcg[k] / user_count for k in ks},
    }

    return metrics


def main():
    logger.info("Loading hetero graph...")
    data = torch.load(GRAPH_PATH, weights_only=False)

    user_in = data["user"].x.size(1)
    item_in = data["item"].x.size(1)
    logger.info(f"user_in_dim={user_in}, item_in_dim={item_in}")

    # device
    if cfg.training.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.training.device)
    logger.info(f"Using device: {device}")

    # model
    model = build_encoder(user_in, item_in, device)

    model_path = PROCESSED_DIR / f"{cfg.model.type}_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    data = data.to(device)
    with torch.no_grad():
        user_emb, item_emb = model(data)

    # interactions
    inter_path = PROCESSED_DIR / "interactions.parquet"
    interactions = pd.read_parquet(inter_path)

    ks = [5, 10, 20]

    logger.info("Evaluating ranking metrics on TEST split...")
    metrics_test = eval_ranking_at_k(
        user_emb, item_emb, interactions, split="test", ks=ks
    )

    for k in ks:
        logger.info(
            f"[TEST] K={k:02d} | "
            f"HR@K={metrics_test['hr'][k]:.4f}  "
            f"P@K={metrics_test['precision'][k]:.4f}  "
            f"R@K={metrics_test['recall'][k]:.4f}  "
            f"NDCG@K={metrics_test['ndcg'][k]:.4f}"
        )


if __name__ == "__main__":
    main()