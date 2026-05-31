"""Train the face embedding checkpoint if it is missing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.face.verification import default_encoder_path


def main() -> None:
    encoder_path = default_encoder_path(_ROOT)
    if encoder_path.is_file():
        print(f"Face encoder already exists: {encoder_path}")
        return

    data_dir = _ROOT / "dataset" / "Face Recognition" / "train"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Training data not found: {data_dir}\n"
            "Add the Roboflow/Kaggle face dataset before training."
        )

    print("No trained face encoder found. Starting classification-based training...")
    print("This satisfies PDF §2.1 (transfer learning with partial fine-tuning).")

    commands = [
        [
            sys.executable,
            str(_ROOT / "scripts" / "train.py"),
            "--task",
            "classifier",
            "--data-dir",
            str(data_dir),
            "--epochs",
            "20",
            "--batch-size",
            "16",
        ],
        [
            sys.executable,
            str(_ROOT / "scripts" / "train.py"),
            "--task",
            "metric",
            "--data-dir",
            str(data_dir),
            "--epochs",
            "20",
            "--batch-size",
            "16",
        ],
    ]

    for command in commands:
        print("\nRunning:", " ".join(command))
        subprocess.run(command, check=True, cwd=str(_ROOT))

    if not encoder_path.is_file():
        raise RuntimeError(f"Training finished but encoder was not created: {encoder_path}")

    print(f"\nReady: {encoder_path}")
    print("Evaluate verification with:")
    print('  python scripts/evaluate.py --data-dir "dataset/Face Recognition/train"')


if __name__ == "__main__":
    main()
