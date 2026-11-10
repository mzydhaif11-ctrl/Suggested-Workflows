"""Command-line utility for indexing local Markdown and text documents."""

import argparse
import json

from dotenv import load_dotenv


def index_documents(force_rebuild=False):
    """Build or refresh the Chroma knowledge index."""
    load_dotenv()
    from vector_store import ensure_index, rebuild_index

    if force_rebuild:
        return rebuild_index()
    return ensure_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the configured knowledge collection.",
    )
    args = parser.parse_args()
    result = index_documents(force_rebuild=args.rebuild)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
