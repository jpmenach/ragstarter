"""Build the Chroma index for the Huellas Felices vet clinic FAQ.

Run from the repo root with: uv run python examples/veterinaria-demo/scripts/ingest.py
"""

from ragstarter.core.ingest import build_index

from clinic import DATA_DIR, STORAGE_DIR


def main() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    index = build_index(data_dir=DATA_DIR, persist_dir=STORAGE_DIR)
    count = index.vector_store._collection.count()
    print(f"Indexed {count} chunks from {DATA_DIR} into {STORAGE_DIR}")


if __name__ == "__main__":
    main()
