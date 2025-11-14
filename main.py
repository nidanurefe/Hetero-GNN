from src.app.logger import get_logger
from src.data.parse_raw import main as parse_main
from src.data.build_features import main as feat_main
from src.data.build_graph import main as build_graph
from src.debug.graph_visualizer import main as visualize_graph
from src.debug.check_edge import main as check_edge
from src.data.split_interactions import main as split_interactions

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting full preprocessing pipeline...")
    parse_main()
    split_interactions()
    feat_main()
    logger.info("Preprocessing pipeline finished.")
    build_graph()
    logger.info("Graph build complete.")
    # visualize_graph()
    # check_edge()
    

    