from pathlib import Path
import math
import torch
from torch import optim

from src.app.config import get_config
from src.app.logger import get_logger
from src.data.prepare_dataset import get_dataloader
from src.model.gat import HeteroGATEncoder, bpr_loss, bpr_softplus_loss
from src.model.sage import HeteroSAGEEncoder
from src.model.gcn import HeteroGCNEncoder
from src.model.lightgcn import LightGCNEncoder

logger = get_logger(__name__)
cfg = get_config()

PROCESSED_DIR = Path(cfg.data.processed_dir)
GRAPH_PATH = PROCESSED_DIR / "hetero_graph.pt"


# ---------------------------
# 1) TRAINING (BPR)
# ---------------------------
def train_one_epoch(model, data, train_loader, optimizer, device, epoch: int) -> float:
    model.train()
    data = data.to(device)
    total_loss = 0.0

    # Choose loss function based on config
    if cfg.training.loss_type == "bpr_sigmoid":
        def loss_fn(u, p, n):
            return bpr_loss(u, p, n, reg_lambda=cfg.training.reg_lambda)
    elif cfg.training.loss_type == "bpr_softplus":
        def loss_fn(u, p, n):
            return bpr_softplus_loss(u, p, n, reg_lambda=cfg.training.reg_lambda)
    else:
        raise ValueError(f"Unknown loss_type: {cfg.training.loss_type}")

    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader, start=1):
        # full-graph forward
        user_emb, item_emb = model(data)

        u = batch["user"].to(device)
        pos_i = batch["pos_item"].to(device)
        neg_i = batch["neg_item"].to(device)

        u_emb = user_emb[u]
        pos_emb = item_emb[pos_i]
        neg_emb = item_emb[neg_i]

        loss = loss_fn(u_emb, pos_emb, neg_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * u.size(0)

        # opsiyonel mini progress
        if batch_idx % 200 == 0:
            logger.info(
                f"[Epoch {epoch:03d}] batch {batch_idx}/{num_batches} "
                f"(last_batch_loss={loss.item():.4f})"
            )

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss


# ---------------------------
# 2) Basit pairwise eval (mevcut)
# ---------------------------
@torch.no_grad()
def eval_pairwise(model, data, val_loader, device) -> float:
    # Pairwise evaluation metric: P( s(u, i+) > s(u, i-) )

    model.eval()
    data = data.to(device)
    user_emb, item_emb = model(data)

    correct = 0
    total = 0

    for batch in val_loader:
        u = batch["user"].to(device)
        pos_i = batch["pos_item"].to(device)
        neg_i = batch["neg_item"].to(device)

        u_emb = user_emb[u]
        pos_emb = item_emb[pos_i]
        neg_emb = item_emb[neg_i]

        pos_scores = (u_emb * pos_emb).sum(dim=-1)
        neg_scores = (u_emb * neg_emb).sum(dim=-1)

        correct += (pos_scores > neg_scores).sum().item()
        total += u.size(0)

    if total == 0:
        return 0.0

    return correct / total


# ---------------------------
# 3) Ranking eval (HR@K, P@K, R@K, NDCG@K)
# ---------------------------
@torch.no_grad()
def eval_ranking(model, data, device, split: str = "val", k_list=(10,)):
    """
    Full ranking evaluation:
      - Her user için, tüm item’lar skorlanıyor
      - split='val' maskinden pozitif item’lar alınıyor
      - HR@K, Precision@K, Recall@K, NDCG@K hesaplanıyor (micro ortalama)
    """
    assert split in ("train", "val", "test")

    model.eval()
    data = data.to(device)

    # 1) Embed hesapla
    user_emb, item_emb = model(data)  # [num_users, d], [num_items, d]

    edge_index = data["user", "rates", "item"].edge_index
    split_mask = data["user", "rates", "item"].__getattr__(f"{split}_mask")

    # 2) split'teki (u, i) pozitif çiftleri al
    u_split = edge_index[0][split_mask].cpu().tolist()
    i_split = edge_index[1][split_mask].cpu().tolist()

    # user -> set(positive_items)
    user_pos_items: dict[int, set[int]] = {}
    for u, i in zip(u_split, i_split):
        user_pos_items.setdefault(u, set()).add(i)

    num_users = len(user_pos_items)
    if num_users == 0:
        logger.warning(f"No users with positive items in split='{split}'")
        return {}

    logger.info(f"Eval split='{split}': {len(i_split)} interactions, {num_users} users with {split} items.")

    num_items = item_emb.size(0)
    device_cpu = torch.device("cpu")  # skor hesaplarını CPU'da numpy ile yapacağız
    item_emb_cpu = item_emb.to(device_cpu)

    from math import log2
    k_list = sorted(k_list)

    metrics_sum = {
        k: {"hr": 0.0, "precision": 0.0, "recall": 0.0, "ndcg": 0.0}
        for k in k_list
    }

    # 3) Her user için ranking hesapla
    for u, pos_items in user_pos_items.items():
        pos_items = set(pos_items)
        if len(pos_items) == 0:
            continue

        u_vec = user_emb[u].to(device_cpu)  # [d]
        scores = (item_emb_cpu @ u_vec).numpy()  # [num_items]

        # Skorlara göre global sıralama
        ranked_items = scores.argsort()[::-1]  # en büyükten küçüğe

        for k in k_list:
            topk = ranked_items[:k]
            hits = 0
            dcg = 0.0

            for rank, item_idx in enumerate(topk):
                if item_idx in pos_items:
                    hits += 1
                    dcg += 1.0 / log2(rank + 2)  # rank 0 -> log2(2) = 1, rank 1 -> log2(3)...

            ideal_hits = min(len(pos_items), k)
            idcg = (
                sum(1.0 / log2(r + 2) for r in range(ideal_hits))
                if ideal_hits > 0
                else 1.0
            )

            hr = 1.0 if hits > 0 else 0.0
            precision = hits / k
            recall = hits / len(pos_items)
            ndcg = dcg / idcg if idcg > 0 else 0.0

            m = metrics_sum[k]
            m["hr"] += hr
            m["precision"] += precision
            m["recall"] += recall
            m["ndcg"] += ndcg

    # 4) Ortalama
    metrics_avg = {}
    for k in k_list:
        m = metrics_sum[k]
        metrics_avg[f"hr@{k}"] = m["hr"] / num_users
        metrics_avg[f"precision@{k}"] = m["precision"] / num_users
        metrics_avg[f"recall@{k}"] = m["recall"] / num_users
        metrics_avg[f"ndcg@{k}"] = m["ndcg"] / num_users

    # Log'la
    for k in k_list:
        logger.info(
            f"[{split.upper()}] K={k:02d} | "
            f"HR@K={metrics_avg[f'hr@{k}']:.4f}  "
            f"P@K={metrics_avg[f'precision@{k}']:.4f}  "
            f"R@K={metrics_avg[f'recall@{k}']:.4f}  "
            f"NDCG@K={metrics_avg[f'ndcg@{k}']:.4f}"
        )

    return metrics_avg


# ---------------------------
# 4) EARLY STOPPING HELPER
# ---------------------------
def is_improvement(best_value: float | None,
                   current_value: float,
                   mode: str,
                   min_delta: float) -> bool:
    if best_value is None:
        return True

    if mode == "max":
        return current_value > best_value + min_delta
    elif mode == "min":
        return current_value < best_value - min_delta
    else:
        raise ValueError(f"Unknown early_stopping.mode: {mode}")


# ---------------------------
# 5) MAIN TRAIN LOOP
# ---------------------------
def main():
    logger.info("Loading hetero graph...")
    data = torch.load(GRAPH_PATH, weights_only=False)

    user_in = data["user"].x.size(1)
    item_in = data["item"].x.size(1)
    logger.info(f"user_in_dim={user_in}, item_in_dim={item_in}")

    # Select device
    if cfg.training.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.training.device)

    logger.info(f"Using device: {device}")

    # -------- MODEL SEÇİMİ --------
    if cfg.model.type == "gat":
        logger.info("Using HeteroGATEncoder")
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
        logger.info("Using HeteroSAGEEncoder")
        model = HeteroSAGEEncoder(
            in_channels_user=user_in,
            in_channels_item=item_in,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        ).to(device)

    elif cfg.model.type == "gcn":
        logger.info("Using HeteroGCNEncoder")
        model = HeteroGCNEncoder(
            in_channels_user=user_in,
            in_channels_item=item_in,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        ).to(device)

    elif cfg.model.type == "lightgcn":
        logger.info("Using LightGCNEncoder")
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

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    train_loader = get_dataloader(
        data,
        "train",
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = get_dataloader(
        data,
        "val",
        batch_size=cfg.training.val_batch_size,
        shuffle=False,
    )

    logger.info(
        f"Train edges: {len(train_loader.dataset)}, "
        f"Val edges: {len(val_loader.dataset)}"
    )

    num_epochs = cfg.training.num_epochs

    # ----- Early Stopping Setup -----
    es_cfg = cfg.training.early_stopping
    metric_name = es_cfg.metric      # örn "ndcg@10"
    mode = es_cfg.mode               # "max" veya "min"
    patience = es_cfg.patience
    min_delta = es_cfg.min_delta
    eval_k = es_cfg.eval_k

    best_metric = None
    best_epoch = 0
    epochs_no_improve = 0

    # ---------------- EPOCH LOOP ----------------
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, data, train_loader, optimizer, device, epoch)
        val_pairwise = eval_pairwise(model, data, val_loader, device)

        # Ranking eval (val split, HR/P/R/NDCG)
        ranking_metrics = eval_ranking(
            model,
            data,
            device,
            split="val",
            k_list=(eval_k,),
        )

        # Early stopping metric'i çek
        if metric_name not in ranking_metrics:
            raise ValueError(
                f"Early stopping metric '{metric_name}' not found in ranking_metrics: "
                f"keys={list(ranking_metrics.keys())}"
            )

        current_metric = ranking_metrics[metric_name]

        logger.info(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}  "
            f"val_pairwise={val_pairwise:.4f}  "
            f"{metric_name}={current_metric:.4f}"
        )

        # ---- Early Stopping Kontrolü ----
        if is_improvement(best_metric, current_metric, mode, min_delta):
            best_metric = current_metric
            best_epoch = epoch
            epochs_no_improve = 0

            # şu anki modeli "en iyi" olarak kaydet
            best_model_path = PROCESSED_DIR / f"{cfg.model.type}_best.pt"
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"New best model at epoch {epoch} with {metric_name}={current_metric:.4f}. "
                f"Saved → {best_model_path}"
            )
        else:
            epochs_no_improve += 1
            logger.info(
                f"No improvement in {metric_name} for {epochs_no_improve} epoch(s). "
                f"Best={best_metric:.4f} at epoch {best_epoch}."
            )
            if epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping triggered after {epochs_no_improve} epochs "
                    f"with no improvement in {metric_name}."
                )
                break

    # Son epoch’taki modeli de kaydet (opsiyonel)
    last_model_path = PROCESSED_DIR / f"{cfg.model.type}_last.pt"
    torch.save(model.state_dict(), last_model_path)
    logger.info(f"Saved last {cfg.model.type} model → {last_model_path}")


if __name__ == "__main__":
    main()