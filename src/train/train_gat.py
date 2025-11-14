from pathlib import Path
import torch
from torch import optim

from src.app.config import get_config
from src.app.logger import get_logger
from src.data.prepare_dataset import get_dataloader
from src.model.gat import HeteroGATEncoder, bpr_loss, bpr_softplus_loss

logger = get_logger(__name__)
cfg = get_config()

PROCESSED_DIR = Path(cfg.data.processed_dir)  
GRAPH_PATH = PROCESSED_DIR / "hetero_graph.pt"


def train_one_epoch(model, data, train_loader, optimizer, device) -> float:
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

    for batch in train_loader:

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

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss


@torch.no_grad()
def eval_pairwise(model, data, val_loader, device) -> float:
    # Pairwise evaluation metric
    # p(u, i+) > p(u, i-) 

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

    model = HeteroGATEncoder(
        in_channels_user=user_in,
        in_channels_item=item_in,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels,
        num_layers=cfg.model.num_layers,
        heads=cfg.model.heads,
        dropout=cfg.model.dropout,
    ).to(device)

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

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, data, train_loader, optimizer, device)
        val_pairwise = eval_pairwise(model, data, val_loader, device)

        logger.info(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}  "
            f"val_pairwise={val_pairwise:.4f}"
        )

    model_path = PROCESSED_DIR / "gat_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved GAT model → {model_path}")


if __name__ == "__main__":
    main()