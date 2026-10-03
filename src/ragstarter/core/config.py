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
    # Generous on purpose: a ~1.5k-token RAG prompt takes close to a minute to process
    # on a partially-CPU-offloaded 8B model (llama-index-llms-ollama defaults to 30s).
    request_timeout: float = 120.0
    # Must be set explicitly: if left unset, llama-index-llms-ollama requests the model's
    # maximum context (131k tokens for llama3.1), and Ollama pre-allocates memory for all
    # of it.
    context_window: int = 4096
