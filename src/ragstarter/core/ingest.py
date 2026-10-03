"""Format-agnostic ingestion: load documents, chunk them, embed, and persist to Chroma.

Business-specific code (which documents to load, how retrieval results get used) stays
in the calling vertical (e.g. examples/veterinaria-demo/) — this module only knows how
to turn a directory of documents into a queryable vector index.
"""

from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from ragstarter.core.config import IngestConfig


def _embed_model(config: IngestConfig) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=config.embed_model_name)


def build_index(
    data_dir: Path,
    persist_dir: Path,
    config: IngestConfig | None = None,
) -> VectorStoreIndex:
    """Read every document under `data_dir`, chunk + embed it, and persist to Chroma.

    Rebuilds the collection from scratch, so the index always mirrors the current
    contents of `data_dir`.
    """
    config = config or IngestConfig()

    documents = SimpleDirectoryReader(input_dir=str(data_dir)).load_data()

    # Split along markdown headers first so one topic doesn't bleed into the next (plain-text
    # or other formats with no headers just pass through as a single node here); then
    # token-split anything still too large (e.g. a long bullet list or table) into
    # chunk_size-bounded pieces with overlap.
    transformations = [
        MarkdownNodeParser(),
        SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        ),
    ]

    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    # Inserting into an existing collection appends: every re-ingest would duplicate all
    # chunks, and chunks from edited or deleted documents would never go away.
    try:
        chroma_client.delete_collection(config.collection_name)
    except NotFoundError:
        pass
    collection = chroma_client.create_collection(config.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=transformations,
        embed_model=_embed_model(config),
    )


def load_index(
    persist_dir: Path,
    config: IngestConfig | None = None,
) -> VectorStoreIndex:
    """Reopen a previously-built index without re-ingesting or re-embedding anything."""
    config = config or IngestConfig()

    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    collection = chroma_client.get_or_create_collection(config.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    return VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=_embed_model(config),
    )
