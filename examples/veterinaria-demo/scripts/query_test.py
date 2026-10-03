"""Check which questions the vet clinic bot answers and which it hands off to a human.

Run from the repo root with: uv run python examples/veterinaria-demo/scripts/query_test.py

Three groups of questions, mixing Spanish and English:
- answerable: the FAQ contains the answer, so the bot should answer.
- near-miss: on-topic for a vet clinic, but the FAQ doesn't say — the bot must not guess.
- off-topic: unrelated to the clinic.
"""

import time

from ragstarter.core.ingest import load_index
from ragstarter.core.llm import build_llm
from ragstarter.core.query import answer

from clinic import PERSONA, STORAGE_DIR

ANSWERABLE = [
    "¿Cuál es el horario de atención los sábados?",
    "What does a general consultation cost?",
    "¿Qué debo hacer si mi mascota tiene una emergencia de madrugada?",
    "Do you offer overnight boarding for healthy pets?",
    "¿Cuánto cuesta esterilizar a mi gata?",
    "What documents do I need for my pet's first visit?",
    "¿Atienden reptiles o aves?",
    "How much notice do you need to cancel an appointment?",
]

NEAR_MISS = [
    "¿Hacen resonancias magnéticas?",
    "Do you do home visits?",
    "¿Atienden caballos?",
    "Is there parking at the clinic?",
    "¿Venden comida para perros?",
    "Do you offer pet physiotherapy or acupuncture?",
    "¿Tienen gatos en adopción?",
    "¿Cuánto cuesta una cirugía de cadera?",
]

OFF_TOPIC = [
    "What's the capital of France?",
    "¿Cómo preparo una lasaña?",
    "Who won the 2022 World Cup?",
    "Explain quantum computing in simple terms",
]

# (question, whether the bot is expected to answer it)
CASES = (
    [(q, True) for q in ANSWERABLE]
    + [(q, False) for q in NEAR_MISS]
    + [(q, False) for q in OFF_TOPIC]
)


def main() -> None:
    index = load_index(persist_dir=STORAGE_DIR)
    llm = build_llm()
    passed = 0

    for query, should_answer in CASES:
        start = time.perf_counter()
        result = answer(query, index=index, llm=llm, persona=PERSONA)
        elapsed = time.perf_counter() - start

        ok = result.grounded == should_answer
        passed += ok
        top_score = max((chunk.score for chunk in result.sources), default=0.0)

        print(f"\n[{'PASS' if ok else 'FAIL'}] {query}")
        print(f"  expected: {'answer' if should_answer else 'hand off'}"
              f"  |  top score: {top_score:.3f}  |  {elapsed:.1f}s")
        print(f"  {result.text if result.grounded else '-> hand off to staff'}")

    print(f"\n{passed}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
