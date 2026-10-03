"""Ask the vet clinic bot a single question.

Run from the repo root with:
    uv run python examples/veterinaria-demo/scripts/ask.py "¿Atienden los domingos?"

Add --sources to also print the retrieved chunks and their similarity scores.
"""

import argparse

from ragstarter.core.ingest import load_index
from ragstarter.core.llm import build_llm
from ragstarter.core.query import answer

from clinic import HANDOFF_MESSAGE, PERSONA, STORAGE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--sources", action="store_true", help="print retrieved chunks")
    args = parser.parse_args()

    result = answer(
        args.question,
        index=load_index(persist_dir=STORAGE_DIR),
        llm=build_llm(),
        persona=PERSONA,
    )

    print(result.text if result.grounded else HANDOFF_MESSAGE)

    if args.sources:
        print()
        for chunk in result.sources:
            snippet = chunk.text.strip().replace("\n", " ")[:100]
            print(f"  [{chunk.score:.3f}] {snippet}...")


if __name__ == "__main__":
    main()
