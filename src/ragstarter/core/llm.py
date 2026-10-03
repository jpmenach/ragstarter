"""Build the LLM that generates answers, from config.

The rest of the engine depends only on LlamaIndex's abstract `LLM` type, never on a
concrete backend class — so swapping Ollama for a hosted API is a change here and in
the config, not in the query pipeline.
"""

from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama

from ragstarter.core.config import LLMConfig


def build_llm(config: LLMConfig | None = None) -> LLM:
    config = config or LLMConfig()

    if config.backend == "ollama":
        return Ollama(
            model=config.model,
            temperature=config.temperature,
            request_timeout=config.request_timeout,
            context_window=config.context_window,
        )

    raise ValueError(f"Unsupported LLM backend: {config.backend!r}")
