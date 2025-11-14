from pathlib import Path
import pandas as pd
from src.app.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
INTERACTIONS_PATH = PROCESSED_DIR / "interactions.parquet"

def split_interactions() -> None:
    df = pd.read_parquet(INTERACTIONS_PATH)
    logger.info(f"Loaded interactions: {len(df)}")

    df  = df.sort_values(["user_idx", "timestamp_ms"]).reset_index(drop=True)
    
    df["split"] = "train"
    grouped = df.groupby("user_idx")

    test_indices = []
    val_indices = []

    for user_idx, group in grouped:
        idxs = group.index.tolist()

        n_interactions = len(idxs)

        if n_interactions >= 3:
            test_indices.append(idxs[-1])
            val_indices.append(idxs[-2])
        elif n_interactions == 2:
            test_indices.append(idxs[-1])
    
    df.loc[test_indices, "split"] = "test"
    df.loc[val_indices, "split"] = "val"

    logger.info(f"Train interactions: {(df['split'] == 'train').sum()}")
    logger.info(f"Validation interactions: {(df['split'] == 'val').sum()}")
    logger.info(f"Test interactions: {(df['split'] == 'test').sum()})") 

    df.to_parquet(INTERACTIONS_PATH, index=False)
    logger.info(f"Updated interactions with split column → {INTERACTIONS_PATH}")


def main():
    logger.info("Starting train/val/test split process...")
    split_interactions()
    logger.info("Train/val/test split process completed.")


if __name__ == "__main__":
    main()