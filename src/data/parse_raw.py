# Raw data -> clean user-item interactions and ID mappings

import json
from pathlib import Path
import pandas as pd
import pickle

from src.app.logger import get_logger
from src.app.config import get_config

cfg = get_config()

PROCESSED_DATA_PATH = Path(cfg.data.processed_dir)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Used to store large DataFrames in a fast, compressed and columnar format.
INTERACTIONS_PATH = PROCESSED_DATA_PATH / "interactions.parquet"
# Used to store ID mappings in a serialized format.
MAPPINGS_PATH = PROCESSED_DATA_PATH / "mappings.pkl"

# Logging Setup
logger = get_logger(__name__)

# Function to read JSONL file
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# Function to build interactions DataFrame
def build_interactions() -> pd.DataFrame:
    interactions = {}

    for entry in iter_all_entries():
        user_id = entry.get("user_id")
        asin = entry.get("asin") or entry.get("parent_asin") # amazon standard identification numbar
        rating = entry.get("rating")
        ts = entry.get("timestamp")
        domain = entry.get("_domain")   
        role = entry.get("_role")       # source / target

        if not user_id or not asin or rating is None or ts is None:
            continue

        # user + item + domain 
        key = (user_id, asin, domain)
        old = interactions.get(key)

        if (old is None) or (entry["timestamp"] > old["timestamp_ms"]):
            interactions[key] = {
                "user_id": user_id,
                "item_id": asin,
                "domain": domain,
                "role": role,  
                "rating": float(rating),
                "timestamp_ms": int(ts),
                "helpful_vote": int(entry.get("helpful_vote", 0) or 0),
                "verified_purchase": bool(entry.get("verified_purchase", False)),
                "title": entry.get("title", "") or "",
                "text": entry.get("text", "") or "",
            }

    df = pd.DataFrame.from_records(list(interactions.values()))
    logger.info(
        "Built interactions from multiple datasets: "
        f"{len(df)} rows, "
        f"{df['user_id'].nunique()} users, "
        f"{df['item_id'].nunique()} items, "
        f"domains={df['domain'].unique().tolist()}"
    )
    return df



def iter_all_entries():    
    
    # Target dataset
    target = cfg.data.raw.target
    target_path = Path(target.path)
    logger.info(f"Reading target dataset: {target_path} (domain={target.name})")

    for entry in read_jsonl(target_path):
        entry["_domain"] = target.name
        entry["_role"] = "target"
        yield entry

    # Source datasets
    for src in cfg.data.raw.sources:
        src_path = Path(src.path)
        logger.info(f"Reading source dataset: {src_path} (domain={src.name})")
        for entry in read_jsonl(src_path):
            entry["_domain"] = src.name
            entry["_role"] = "source"
            yield entry


# Function to build ID mappings
def build_id_mappings(df: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    user_ids = df["user_id"].unique().tolist()
    item_ids = df["item_id"].unique().tolist()

    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item2idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    idx2user = {idx: user_id for user_id, idx in user2idx.items()}
    idx2item = {idx: item_id for item_id, idx in item2idx.items()}

    return user2idx, item2idx, idx2user, idx2item


def main():
    logger.info("Building interactions DataFrame...")
    interactions_df = build_interactions()
    user2id, item2id, id2user, id2item = build_id_mappings(interactions_df)

    interactions_df["user_idx"] = interactions_df["user_id"].map(user2id)
    interactions_df["item_idx"] = interactions_df["item_id"].map(item2id)

    interactions_df.to_parquet(INTERACTIONS_PATH, index=False)
    logger.info(f"Saved interactions → {INTERACTIONS_PATH}")
    
    with MAPPINGS_PATH.open("wb") as f:
        pickle.dump(
            {
                "user2id": user2id,
                "item2id": item2id,
                "id2user": id2user,
                "id2item": id2item,
            },
            f,
        )

    logger.info(f"Saved mappings → {MAPPINGS_PATH}")
    logger.info("Finished parse_raw.")


if __name__ == "__main__":
    main()