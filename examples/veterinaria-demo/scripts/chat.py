"""Web chat for trying the vet clinic bot by hand.

Run from the repo root with: uv run python examples/veterinaria-demo/scripts/chat.py
then open http://localhost:7860 (from WSL, a Windows browser works too).

A testing tool, not the product UI — the real interface is WhatsApp (Phase 4). Each
message is answered on its own: the chat shows history, but the engine doesn't use it yet.
"""

import time

import gradio as gr

from ragstarter.core.ingest import load_index
from ragstarter.core.llm import build_llm
from ragstarter.core.query import answer

from clinic import HANDOFF_MESSAGE, PERSONA, STORAGE_DIR

# Loaded once at startup, not per message: the embedding model alone takes seconds to load.
INDEX = load_index(persist_dir=STORAGE_DIR)
LLM = build_llm()


def respond(message: str, history: list) -> list[gr.ChatMessage]:
    start = time.perf_counter()
    result = answer(message, index=INDEX, llm=LLM, persona=PERSONA)
    elapsed = time.perf_counter() - start

    sources = "\n\n".join(
        f"**[{chunk.score:.3f}]** {chunk.text.strip()}" for chunk in result.sources
    )
    verdict = "answered" if result.grounded else "handed off"

    return [
        # Rendered by Gradio as a collapsed section above the reply.
        gr.ChatMessage(
            content=sources,
            metadata={"title": f"Retrieved chunks ({verdict})", "duration": elapsed},
        ),
        gr.ChatMessage(content=result.text if result.grounded else HANDOFF_MESSAGE),
    ]


demo = gr.ChatInterface(
    fn=respond,
    title="Clínica Veterinaria Huellas Felices — test chat",
    examples=[
        "¿Cuál es el horario de atención los sábados?",
        "How much does a general consultation cost?",
        "¿Atienden caballos?",
    ],
)

if __name__ == "__main__":
    demo.launch()
