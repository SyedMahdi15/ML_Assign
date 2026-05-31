"""Run one-time setup for missing trained artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def check_environment() -> None:
    version = sys.version_info
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)

    if version >= (3, 13):
        print("ERROR: Python 3.13 makes TensorFlow crash on Mac (Python quit unexpectedly).")
        print("Use the project virtualenv instead:\n")
        print(f"  cd \"{ROOT}\"")
        print("  source .venv/bin/activate")
        print("  python scripts/setup_project.py")
        print("\nIf .venv does not exist yet:")
        print("  brew install python@3.11")
        print("  python3.11 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  pip install -r requirements.txt")
        raise SystemExit(1)

    if not in_venv:
        print("WARNING: Not running inside .venv.")
        if VENV_PYTHON.is_file():
            print(f"Recommended: source \"{ROOT / '.venv' / 'bin' / 'activate'}\" first.\n")


def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=str(ROOT), check=False)


def main() -> None:
    check_environment()
    print("COS30082 project setup")
    print("=" * 40)
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")

    run([sys.executable, str(ROOT / "scripts" / "ensure_face_encoder.py")])

    emotion_model = ROOT / "models" / "emotion_model.h5"
    if not emotion_model.is_file():
        print("\nEmotion model missing. Train with:")
        print('  python scripts/train_emotion.py --data-root "dataset/Emotion_Detection"')
    else:
        print(f"\nEmotion model found: {emotion_model}")

    liveness_model = ROOT / "models" / "liveness_model.h5"
    if not liveness_model.is_file():
        print("\nLiveness CNN missing. Train with:")
        print(
            '  python scripts/train.py --task liveness '
            '--live-dir "dataset/liveness/live" --spoof-dir "dataset/liveness/spoof"'
        )
    else:
        print(f"Liveness model found: {liveness_model}")

    print("\nEvaluate face verification:")
    print('  python scripts/evaluate.py --data-dir "dataset/Face Recognition/train"')
    print("\nRun integrated system:")
    print("  python final_system_clean.py")


if __name__ == "__main__":
    main()
