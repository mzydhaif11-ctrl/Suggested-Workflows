import logging

from vector_store import DOCS_DIR, ensure_index, rebuild_index

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def index_documents(force_rebuild: bool = False):
    if force_rebuild:
        result = rebuild_index(DOCS_DIR)
    else:
        result = ensure_index(DOCS_DIR)

    logger.info("Knowledge indexing result: %s", result)
    return result


if __name__ == "__main__":
    index_documents(force_rebuild=True)
