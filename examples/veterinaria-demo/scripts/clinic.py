"""Clinic-specific settings shared by the demo scripts."""

from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DEMO_DIR / "data"
STORAGE_DIR = DEMO_DIR / "storage" / "chroma"

PERSONA = (
    "You are the virtual assistant of Clínica Veterinaria Huellas Felices. "
    "You help pet owners with questions about the clinic in a friendly, warm tone."
)

# Shown when the engine can't answer from the FAQ. In Phase 4 this becomes a real
# hand-off to clinic staff on WhatsApp.
HANDOFF_MESSAGE = (
    "No tengo esa información, pero un miembro de nuestro equipo te responderá pronto. "
    "Si es una emergencia, llama al +57 000 123 4567 (opción 2)."
)
