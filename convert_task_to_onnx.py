import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


MODEL_NAMES = (
    "hand_detector",
    "hand_landmarks_detector",
)


def convert_model(source_path, output_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--tflite",
            str(source_path),
            "--output",
            str(output_path),
            "--opset",
            "17",
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert MediaPipe task models to ONNX."
    )
    parser.add_argument(
        "--task",
        type=Path,
        default=Path("hand_landmarker.task"),
        help="Path to hand_landmarker.task.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory for generated ONNX models.",
    )
    args = parser.parse_args()

    task_path = args.task.expanduser().resolve()
    if not task_path.is_file():
        raise FileNotFoundError(f"Task model not found: {task_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_path = Path(temporary_dir)
        with zipfile.ZipFile(task_path) as task_archive:
            task_archive.extractall(temporary_path)

        for model_name in MODEL_NAMES:
            source_path = temporary_path / f"{model_name}.tflite"
            output_path = args.output_dir / f"{model_name}.onnx"
            convert_model(source_path, output_path)
            print(f"Created {output_path}")


if __name__ == "__main__":
    main()