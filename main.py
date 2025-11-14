from src.app.logger import get_logger

# DATA
from src.data.parse_raw import main as parse_main
from src.data.split_interactions import main as split_main
from src.data.build_features import main as feat_main
from src.data.build_graph import main as graph_main

# TRAINING
from src.train.train_gat import main as train_gat_main

# EXPORT
from src.export.export_embeddings import main as export_emb_main

# DEBUG
from src.debug.graph_visualizer import main as visualize_graph
from src.debug.check_edge import main as check_edge_main

logger = get_logger(__name__)


if __name__ == "__main__":
    logger.info("Starting pipeline...")

    logger.info("Step 1: Parsing raw datasets...")
    parse_main()

    logger.info("Step 2: Splitting interactions...")
    split_main()

    logger.info("Step 3: Building user/item features...")
    feat_main()

    logger.info("Step 4: Building heterogeneous graph...")
    graph_main()

    logger.info("Step 5: Training GAT model...")
    train_gat_main()

    logger.info("Step 6: Exporting embeddings...")
    export_emb_main()

    # Optional: enable for debugging
    # visualize_graph()
    # check_edge_main()

    logger.info("Pipeline completed successfully!")