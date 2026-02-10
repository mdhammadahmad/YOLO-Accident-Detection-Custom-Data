#    Run YOLO prediction on a single image or a directory of images.

from ultralytics import YOLO
from pathlib import Path
import argparse


def run_prediction(model_path, source_path, save_dir, conf=0.25):

    model = YOLO(model_path)

    source_path = Path(source_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if source_path.is_file():
        model(
            source=str(source_path),
            save=True,
            conf=conf,
            project=str(save_dir),
            name="results",
            exist_ok=True
        )

    elif source_path.is_dir():
        for img_path in source_path.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                model(
                    source=str(img_path),
                    save=True,
                    conf=conf,
                    project=str(save_dir),
                    name="results",
                    exist_ok=True
                )
    else:
        raise ValueError("Source path is neither a file nor a directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Accident Detection Prediction")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO model (.pt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image OR directory of images"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="predictions",
        help="Directory to save prediction results"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )

    args = parser.parse_args()

    run_prediction(
        model_path=args.model,
        source_path=args.source,
        save_dir=args.save_dir,
        conf=args.conf
    )
