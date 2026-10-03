from dataclasses import dataclass


@dataclass(frozen=True)
class IngestConfig:
    """Chunking + embedding settings shared by every vertical built on this engine."""

    embed_model_name: str = "BAAI/bge-m3"
    chunk_size: int = 500
    chunk_overlap: int = 75
    collection_name: str = "docs"


@dataclass(frozen=True)
class LLMConfig:
    """Which LLM backend generates answers, and how it's called."""

    backend: str = "ollama"
    model: str = "llama3.1:8b"
    # Factual Q&A over retrieved context: keep sampling close to deterministic.
    temperature: float = 0.1
    # Generous on purpose: answers take 4-10s on a partially-CPU-offloaded 8B model, but
    # the first request also loads the model into memory, and slower machines exist
    # (llama-index-llms-ollama defaults to 30s).
    request_timeout: float = 120.0
    # Must be set explicitly: if left unset, llama-index-llms-ollama requests the model's
    # maximum context (131k tokens for llama3.1), and Ollama pre-allocates memory for all
    # of it.
    context_window: int = 4096


@dataclass(frozen=True)
class QueryConfig:
    """Retrieval settings, and when to give up instead of answering."""

    top_k: int = 3
    # Below this top-1 similarity, the question is treated as unrelated to the documents
    # and the LLM is never called. Measured on the demo FAQ with bge-m3: answerable
    # questions scored 0.37-0.54, clearly off-topic ones 0.22-0.26. Scores depend on the
    # corpus and embedding model, so re-measure when either changes.
    min_score: float = 0.30
