from dataclasses import dataclass


@dataclass(frozen=True)
class IngestConfig:
    """Chunking + embedding settings shared by every vertical built on this engine."""

    embed_model_name: str = "BAAI/bge-m3"
    chunk_size: int = 500
    chunk_overlap: int = 75
    collection_name: str = "docs"
