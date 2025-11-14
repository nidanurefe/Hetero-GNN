from src.app.logger import get_logger
from src.data.parse_raw import main as parse_main
from src.data.build_features import main as feat_main
from src.data.build_graph import main as build_graph

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting full preprocessing pipeline...")
    parse_main()
    feat_main()
    logger.info("Preprocessing pipeline finished.")
    build_graph()
    logger.info("Graph build complete.")
    