from pathlib import Path
import random
import torch
import matplotlib.pyplot as plt
import networkx as nx
from src.app.logger import get_logger
import pickle

logger = get_logger(__name__)


PROCESSED_DIR = Path("data/processed")
GRAPH_PATH = PROCESSED_DIR / "hetero_graph.pt"
MAPPINGS_PATH = PROCESSED_DIR / "mappings.pkl"


def main():
    data = torch.load(GRAPH_PATH, weights_only=False)

    with MAPPINGS_PATH.open("rb") as f:
        mappings = pickle.load(f)

    id2user: dict[int, str] = mappings["idx2user"]
    id2item: dict[int, str] = mappings["idx2item"]

    edge_index = data['user', 'rates', 'item'].edge_index
    num_edges = edge_index.shape[1]

    logger.info(f"Total edges: {num_edges}")

    # Select a random user
    user_indices = edge_index[0].tolist()
    random_user = random.choice(user_indices)
    random_user_idNM = id2user[int(random_user)]
    logger.info(f"Random user_id selected: {random_user_idNM}")

    # Find neighbor items of the user
    mask = edge_index[0] == random_user
    user_edges = edge_index[:, mask]

    user_items = user_edges[1].tolist()
    logger.info(f"User has {len(user_items)} items in sampled edges.")

    # Take first 20 neighbors
    if len(user_items) > 20:
        user_items = user_items[:20]
        logger.info("Truncated items to first 20 for visualization.")

    # Construct networkX bipartite graph 
    B = nx.Graph()

    # User node
    user_node = f"u{random_user}"
    B.add_node(user_node, bipartite=0)

    # Item nodes
    for item_id in user_items:
        item_node = f"i{item_id}"
        B.add_node(item_node, bipartite=1)
        B.add_edge(user_node, item_node)

    pos = nx.spring_layout(B, seed=42)

    plt.figure(figsize=(6, 6))

    user_nodes = [n for n in B.nodes if n.startswith("u")]
    item_nodes = [n for n in B.nodes if n.startswith("i")]

    nx.draw_networkx_nodes(B, pos, nodelist=user_nodes, node_shape="o", label="users")
    nx.draw_networkx_nodes(B, pos, nodelist=item_nodes, node_shape="s", label="items")
    nx.draw_networkx_edges(B, pos, alpha=0.5)
    nx.draw_networkx_labels(B, pos, font_size=6)

    plt.title(f"Subgraph around user_id={random_user_idNM}")
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()