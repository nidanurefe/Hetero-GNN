from pathlib import Path
import random
import pickle

import torch
import pandas as pd
import numpy as np

from src.app.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
GRAPH_PATH = PROCESSED_DIR / "hetero_graph.pt"
INTERACTIONS_PATH = PROCESSED_DIR / "interactions.parquet"
MAPPINGS_PATH = PROCESSED_DIR / "mappings.pkl"


def main():
    data = torch.load(GRAPH_PATH, weights_only=False)
    interactions = pd.read_parquet(INTERACTIONS_PATH)

    with MAPPINGS_PATH.open("rb") as f:
        mappings = pickle.load(f)

    id2user: dict[int, str] = mappings["idx2user"]
    id2item: dict[int, str] = mappings["idx2item"]

    edge_index = data['user', 'rates', 'item'].edge_index
    edge_attr = data['user', 'rates', 'item'].edge_attr

    num_edges = edge_index.shape[1]
    logger.info(f"Total edges: {num_edges}")

    # Select random edge
    e_id = random.randrange(num_edges)
    u_id = int(edge_index[0, e_id])
    i_id = int(edge_index[1, e_id])

    u_id = id2user[u_id]
    i_id = id2item[i_id]

    logger.info(f"Random edge #{e_id}")
    logger.info(f"  user_id = {u_id}, user_id = {u_id}")
    logger.info(f"  item_id = {i_id}, item_id = {i_id}")


    rating_norm, time_norm, verified_flag, helpful_log = edge_attr[e_id].tolist()
    logger.info(f"  edge_attr:")
    logger.info(f"    rating_norm = {rating_norm}")
    logger.info(f"    time_norm   = {time_norm}")
    logger.info(f"    verified    = {verified_flag}")
    logger.info(f"    helpful_log = {helpful_log}")

    # Find the same row in interactions.parquet 
    row = interactions[(interactions["user_id"] == u_id) &
                   (interactions["item_id"] == i_id)]

    if row.empty:
        logger.error("No matching row found in interactions.parquet!")
        return

    row = row.iloc[0]
    logger.info("Matching row in interactions.parquet:")
    logger.info(f"  user_id       = {row['user_id']}")
    logger.info(f"  item_id       = {row['item_id']}")
    logger.info(f"  rating        = {row['rating']}")
    logger.info(f"  timestamp_ms  = {row['timestamp_ms']}")
    logger.info(f"  helpful_vote  = {row['helpful_vote']}")
    logger.info(f"  verified_purc = {row['verified_purchase']}")

    # rating_norm ≈ (rating - 1) / 4
    expected_rating_norm = (row['rating'] - 1.0) / 4.0
    logger.info(f"  expected rating_norm = {expected_rating_norm}")

    # helpful_log ≈ log1p(helpful_vote)
    expected_helpful_log = np.log1p(row['helpful_vote'])
    logger.info(f"  expected helpful_log = {expected_helpful_log}")

    # verified_flag ≈ int(verified_purchase)
    expected_verified = int(row['verified_purchase'])
    logger.info(f"  expected verified    = {expected_verified}")

    logger.info("Spot check done")


if __name__ == "__main__":
    main()