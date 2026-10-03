"""Ask the vet clinic bot a set of test questions and print answers with their sources.

Run from the repo root with: uv run python examples/veterinaria-demo/scripts/query_test.py

Mixes Spanish and English questions to check that the multilingual embedding model
retrieves relevant chunks across languages, and that answers come back in the language
the question was asked in.
"""

import time
from pathlib import Path

from ragstarter.core.ingest import load_index
from ragstarter.core.llm import build_llm
from ragstarter.core.query import answer

DEMO_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = DEMO_DIR / "storage" / "chroma"

PERSONA = (
    "You are the virtual assistant of Clínica Veterinaria Huellas Felices. "
    "You help pet owners with questions about the clinic in a friendly, warm tone."
)

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
    llm = build_llm()

    for query in TEST_QUERIES:
        start = time.perf_counter()
        result = answer(query, index=index, llm=llm, persona=PERSONA)
        elapsed = time.perf_counter() - start

        print(f"\n=== {query}  ({elapsed:.1f}s)")
        print(result.text)
        for chunk in result.sources:
            snippet = chunk.text.strip().replace("\n", " ")[:100]
            print(f"  [{chunk.score:.3f}] {snippet}...")


if __name__ == "__main__":
    main()
