from pathlib import Path
import pandas as pd
import numpy as np
from src.app.logger import get_logger

PROCESSED_DIR = Path("data/processed")
INTERACTIONS_PATH = PROCESSED_DIR / "interactions.parquet"
USER_FEAT_PATH = PROCESSED_DIR / "user_features.parquet"
ITEM_FEAT_PATH = PROCESSED_DIR / "item_features.parquet"

logger = get_logger(__name__)

# Function to normalize node features
def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    min_val = s.min()
    max_val = s.max()
    if max_val - min_val == 0:
        return np.zeros_like(s, dtype=float)
    return (s - min_val) / (max_val - min_val)


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp_days"] = df["timestamp_ms"] / 1000 / 3600 / 24 # Convert ms to days
    grouped = df.groupby("user_idx")

    user_feat = pd.DataFrame({
        "user_idx": grouped.size().index,
        "user_id": grouped["user_id"].first().values,  
        "num_reviews": grouped.size().values,
        "avg_rating": grouped["rating"].mean().values,
        "rating_std": grouped["rating"].std().fillna(0.0).values,
        "verified_ratio": grouped["verified_purchase"].mean().values,
        "avg_helpful": grouped["helpful_vote"].mean().values,
        "avg_review_len": grouped["text"].apply(lambda x: x.str.len().mean()).values,
        "first_ts": grouped["timestamp_days"].min().values,
        "last_ts": grouped["timestamp_days"].max().values,
    })

    # Normalize features
    for col in [
        "num_reviews",    # Activity level 
        "avg_helpful",    # Is the user generally helpful?
        "avg_review_len", # Does the user write long reviews?
        "first_ts",       # How early did the user start reviewing?
        "last_ts",        # How recently did the user review?
    ]:
        user_feat[col + "_norm"] = normalize_series(user_feat[col])

    return user_feat



def build_item_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp_days"] = df["timestamp_ms"] / 1000 / 3600 / 24 # Convert ms to days
    
    grouped = df.groupby("item_idx")

    # Item features
    item_feat = pd.DataFrame({
        "item_idx": grouped.size().index,
        "item_id": grouped["item_id"].first().values,  # string asin
        "num_reviews": grouped.size().values,
        "avg_rating": grouped["rating"].mean().values,
        "rating_std": grouped["rating"].std().fillna(0.0).values,
        "verified_ratio": grouped["verified_purchase"].mean().values,
        "avg_helpful": grouped["helpful_vote"].mean().values,
        "avg_review_len": grouped["text"].apply(lambda x: x.str.len().mean()).values,
        "first_ts": grouped["timestamp_days"].min().values,
        "last_ts": grouped["timestamp_days"].max().values,
    })

    # Normalize features
    for col in [
        "num_reviews",
        "avg_helpful",
        "avg_review_len",
        "first_ts",
        "last_ts",
    ]:
        item_feat[col + "_norm"] = normalize_series(item_feat[col])

    return item_feat
        
    
def main():
    df = pd.read_parquet(INTERACTIONS_PATH)

    logger.info(f"Loaded interactions: {len(df)}")

    user_feat = build_user_features(df)
    item_feat = build_item_features(df)

    user_feat.to_parquet(USER_FEAT_PATH, index=False)
    item_feat.to_parquet(ITEM_FEAT_PATH, index=False)

    logger.info(f"Saved user features to {USER_FEAT_PATH}")
    logger.info(f"Saved item features to {ITEM_FEAT_PATH}")


if __name__ == "__main__":
    main()