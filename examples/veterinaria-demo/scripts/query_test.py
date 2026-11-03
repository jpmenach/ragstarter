"""Check that the vet clinic bot answers what it should, correctly, and hands off the rest.

Run from the repo root with: uv run python examples/veterinaria-demo/scripts/query_test.py

Three groups of questions, mixing Spanish and English:
- answerable: the FAQ contains the answer, so the bot should answer — and the answer must
  contain the key facts from the FAQ.
- near-miss: on-topic for a vet clinic, but the FAQ doesn't say — the bot must not guess.
- off-topic: unrelated to the clinic.

This is an evaluation script, not a unit test: it needs a running Ollama, takes a couple
of minutes, and the LLM isn't fully deterministic, so a case can pass on one run and fail
on the next. Run it more than once before trusting a result.
"""

import re
import time
from dataclasses import dataclass

from ragstarter.core.ingest import load_index
from ragstarter.core.llm import build_llm
from ragstarter.core.query import answer

from clinic import PERSONA, STORAGE_DIR


@dataclass(frozen=True)
class Case:
    question: str
    should_answer: bool
    # Case-insensitive regexes that a correct answer must all match. They accept both
    # Spanish and English wording, since the answer language isn't fully reliable.
    must_match: tuple[str, ...] = ()


ANSWERABLE = [
    Case("¿Cuál es el horario de atención los sábados?", True, (r"9:00", r"3:00")),
    Case("What does a general consultation cost?", True, (r"\$\s?25\b",)),
    # The FAQ gives two valid answers: call the emergency line, or come to the clinic.
    Case("¿Qué debo hacer si mi mascota tiene una emergencia de madrugada?", True,
         (r"000 123 4567|directamente|directly",)),
    Case("Do you offer overnight boarding for healthy pets?", True, (r"\bno\b",)),
    Case("¿Cuánto cuesta esterilizar a mi gata?", True, (r"\$\s?70\b",)),
    Case("How much is it to spay my female cat?", True, (r"\$\s?70\b",)),
    Case("What documents do I need for my pet's first visit?", True,
         (r"carnet|vaccination (card|record)",)),
    Case("¿Atienden reptiles o aves?", True, (r"\b(sí|si|yes)\b", r"llam|call")),
    Case("How much notice do you need to cancel an appointment?", True,
         (r"\b4 (horas|hours)\b",)),
    Case("¿Atienden los domingos?", True, (r"^no\b", r"cerrad|closed")),
    Case("¿Abren los festivos?", True, (r"^no\b", r"cerrad|closed")),
]

NEAR_MISS = [
    Case(q, False) for q in [
        "¿Hacen resonancias magnéticas?",
        "Do you do home visits?",
        "¿Atienden caballos?",
        "Is there parking at the clinic?",
        "¿Venden comida para perros?",
        "Do you offer pet physiotherapy or acupuncture?",
        "¿Tienen gatos en adopción?",
        "¿Cuánto cuesta una cirugía de cadera?",
    ]
]

OFF_TOPIC = [
    Case(q, False) for q in [
        "What's the capital of France?",
        "¿Cómo preparo una lasaña?",
        "Who won the 2022 World Cup?",
        "Explain quantum computing in simple terms",
    ]
]


def check(case: Case, grounded: bool, text: str) -> str | None:
    """Return why the case failed, or None if it passed."""
    if grounded != case.should_answer:
        return "answered" if grounded else "handed off"
    missing = [p for p in case.must_match if not re.search(p, text, re.IGNORECASE)]
    if missing:
        return f"answer doesn't match {', '.join(missing)}"
    return None


def main() -> None:
    index = load_index(persist_dir=STORAGE_DIR)
    llm = build_llm()
    failures = []

    for case in ANSWERABLE + NEAR_MISS + OFF_TOPIC:
        start = time.perf_counter()
        result = answer(case.question, index=index, llm=llm, persona=PERSONA)
        elapsed = time.perf_counter() - start

        problem = check(case, result.grounded, result.text)
        if problem:
            failures.append((case.question, problem))
        top_score = max((chunk.score for chunk in result.sources), default=0.0)

        print(f"\n[{'FAIL' if problem else 'PASS'}] {case.question}")
        print(f"  expected: {'answer' if case.should_answer else 'hand off'}"
              f"  |  top score: {top_score:.3f}  |  {elapsed:.1f}s")
        print(f"  {result.text if result.grounded else '-> hand off to staff'}")
        if problem:
            print(f"  why: {problem}")

    total = len(ANSWERABLE) + len(NEAR_MISS) + len(OFF_TOPIC)
    print(f"\n{total - len(failures)}/{total} passed")
    for question, problem in failures:
        print(f"  FAIL  {question}  ({problem})")


if __name__ == "__main__":
    main()
