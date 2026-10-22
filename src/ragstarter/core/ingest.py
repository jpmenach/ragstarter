"""Format-agnostic ingestion: load documents, chunk them, embed, and persist to Chroma.

Business-specific code (which documents to load, how retrieval results get used) stays
in the calling vertical (e.g. examples/veterinaria-demo/) — this module only knows how
to turn documents (a directory of files, or documents built in memory) into a queryable
vector index.
"""

from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.ingestion import run_transformations
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TransformComponent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from ragstarter.core.config import IngestConfig


def _embed_model(config: IngestConfig) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=config.embed_model_name)


def _transformations(config: IngestConfig) -> list[TransformComponent]:
    # Split along markdown headers first so one topic doesn't bleed into the next (plain-text
    # or other formats with no headers just pass through as a single node here); then
    # token-split anything still too large (e.g. a long bullet list or table) into
    # chunk_size-bounded pieces with overlap.
    return [
        MarkdownNodeParser(),
        SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        ),
    ]


def build_index(
    data_dir: Path,
    persist_dir: Path,
    config: IngestConfig | None = None,
) -> VectorStoreIndex:
    """Read every document under `data_dir`, chunk + embed it, and persist to Chroma.

    Rebuilds the collection from scratch, so the index always mirrors the current
    contents of `data_dir`.
    """
    documents = SimpleDirectoryReader(input_dir=str(data_dir)).load_data()
    return build_index_from_documents(documents, persist_dir, config)


def build_index_from_documents(
    documents: list[Document],
    persist_dir: Path,
    config: IngestConfig | None = None,
) -> VectorStoreIndex:
    """Chunk + embed documents the caller built (e.g. from database records) into Chroma.

    Each document's metadata is copied to its chunks and comes back in
    `RetrievedChunk.metadata`, so callers can attach their own fields (a record ID, a
    page route) and trace answers back to them. LlamaIndex also embeds metadata along with
    the text unless a key is listed in the document's `excluded_embed_metadata_keys`:
    exclude fields that are labels (IDs, routes), keep ones that describe the content.

    Rebuilds the collection from scratch, like `build_index`.
    """
    config = config or IngestConfig()

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
        transformations=_transformations(config),
        embed_model=_embed_model(config),
    )


def replace_documents(
    index: VectorStoreIndex,
    documents: list[Document],
    config: IngestConfig | None = None,
) -> None:
    """Swap documents in an existing index for their new versions, matched by document ID.

    For records that change one at a time, where rebuilding everything per edit would be
    too slow: give each document a stable `id_` (e.g. the record's key) when building the
    index, and pass the edited ones here. A document not in the index yet is just added.
    Only these documents are embedded again, with the index's own embedding model.

    Between deleting the old chunks and inserting the new ones (the time it takes to embed
    them), a query won't find these documents at all.
    """
    config = config or IngestConfig()
    delete_documents(index, [document.doc_id for document in documents])
    index.insert_nodes(run_transformations(documents, _transformations(config)))


def delete_documents(index: VectorStoreIndex, document_ids: list[str]) -> None:
    """Remove documents' chunks from an existing index. Unknown IDs are ignored."""
    for document_id in document_ids:
        index.delete_ref_doc(document_id)


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
