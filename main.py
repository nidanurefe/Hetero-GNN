from src.app.logger import get_logger


# TODO: early stopping ekle
# TODO: best model state tut 
# TODO: yeni değerlendirme metrikleri ekle


# DATA
from src.data.parse_raw import main as parse
from src.data.split_interactions import main as split
from src.data.build_features import main as feat
from src.data.build_graph import main as graph
from src.eval.eval_ranking import main as evaluate

# TRAINING
from src.train.train import main as train

# EXPORT
from src.export.export_embeddings import main as export_emb

# DEBUG
from src.debug.graph_visualizer import main as visualize_graph
from src.debug.check_edge import main as check_edge

logger = get_logger(__name__)


if __name__ == "__main__":
    logger.info("Starting pipeline...")

    logger.info("Step 1: Parsing raw datasets...")
    parse()

    logger.info("Step 2: Splitting interactions...")
    split()

    logger.info("Step 3: Building user/item features...")
    feat()

    logger.info("Step 4: Building heterogeneous graph...")
    graph()
    logger.info("Step 5: Training model...")
    train()

    # logger.info("Step 6: Exporting embeddings...")
    # export_emb()

    # Optional: enable for debugging
    # visualize_graph()
    # check_edge_main()

    evaluate()



    logger.info("Pipeline completed successfully!")