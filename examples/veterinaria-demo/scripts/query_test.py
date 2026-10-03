"""Smoke-test retrieval (no LLM generation yet) against the vet clinic FAQ index.

Run from the repo root with: uv run python examples/veterinaria-demo/scripts/query_test.py

Mixes Spanish and English questions to check that the multilingual embedding model
actually retrieves relevant chunks across languages, not just within Spanish.
"""

from pathlib import Path

from ragstarter.core.ingest import load_index

DEMO_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = DEMO_DIR / "storage" / "chroma"

TEST_QUERIES = [
    "¿Cuál es el horario de atención los sábados?",
    "What does a general consultation cost?",
    "¿Qué debo hacer si mi mascota tiene una emergencia de madrugada?",
    "Do you offer overnight boarding for healthy pets?",
    "¿Cuánto cuesta esterilizar a mi gata?",
    "What documents do I need for my pet's first visit?",
    "¿Atienden reptiles o aves?",
    "How much notice do you need to cancel an appointment?",
]


def main() -> None:
    index = load_index(persist_dir=STORAGE_DIR)
    retriever = index.as_retriever(similarity_top_k=3)

    for query in TEST_QUERIES:
        print(f"\n=== {query}")
        for node in retriever.retrieve(query):
            snippet = node.text.strip().replace("\n", " ")[:160]
            print(f"  [{node.score:.3f}] {snippet}...")


if __name__ == "__main__":
    main()
