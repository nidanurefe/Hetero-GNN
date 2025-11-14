# Raw data -> clean user-item interactions and ID mappings

import json
from pathlib import Path
import pandas as pd
import pickle
from src.app.logger import get_logger

RAW_DATA_PATH = Path("data/raw/Magazine_Subscriptions.jsonl")
PROCESSED_DATA_PATH = Path("data/processed")
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
def build_interactions()-> pd.DataFrame:
    interactions = {}

    for entry in read_jsonl(RAW_DATA_PATH):
        user_id = entry.get("user_id")
        asin = entry.get("asin") or entry.get("parent_asin") # Amazon Standard Identification Number
        rating = entry.get("rating")
        ts = entry.get("timestamp")

        if not user_id or not asin or rating is None or ts is None:
            continue

        key = (user_id, asin)
        old = interactions.get(key)

        if (old is None) or (entry["timestamp"] > old["timestamp_ms"]) :
            interactions[key] = {
                "user_id": user_id,
                "item_id": asin,
                "rating": float(rating),
                "timestamp_ms": int(ts),
                "helpful_vote": int(entry.get("helpful_vote", 0) or 0),
                "verified_purchase": bool(entry.get("verified_purchase", False)),
                "title": entry.get("title", "") or "",
                "text": entry.get("text", "") or "",
            }

    df = pd.DataFrame.from_records(list(interactions.values()))
    return df


# Function to build ID mappings
def build_id_mappings(df: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    user_ids = df["user_id"].unique().tolist()
    item_ids = df["item_id"].unique().tolist()

    user2id = {user_id: id for id, user_id in enumerate(user_ids)}
    item2id = {item_id: id for id, item_id in enumerate(item_ids)}

    id2user = {id: user_id for user_id, id in user2id.items()}
    id2item = {id: item_id for item_id, id in item2id.items()}

    return user2id, item2id, id2user, id2item


def main():
    logger.info("Building interactions DataFrame...")
    interactions_df = build_interactions()
    user2id, item2id, id2user, id2item = build_id_mappings(interactions_df)

    interactions_df["user_id"] = interactions_df["user_id"].map(user2id)
    interactions_df["item_id"] = interactions_df["item_id"].map(user2id)

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