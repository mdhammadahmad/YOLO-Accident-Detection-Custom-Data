from ultralytics import YOLO
from pathlib import Path
import argparse


def predict(model, source, save_dir, conf=0.25):

    save_dir.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        print(f"[INFO] Predicting single image: {source.name}")
        model(
            source=str(source),
            save=True,
            conf=conf,
            project=str(save_dir),
            name="results",
            exist_ok=True
        )

    elif source.is_dir():
        print(f"[INFO] Predicting all images in folder: {source}")
        for img_path in source.iterdir():
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
        raise ValueError("Source is neither a file nor a directory")


def main():
    parser = argparse.ArgumentParser(
        description="Accident Detection - Single Image or Folder Prediction"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to a single image OR a folder of images"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="../runs/accident_train_v1/weights/best.pt",
        help="Path to trained YOLO model"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="../output",
        help="Directory to save predictions"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )

    args = parser.parse_args()
    model = YOLO(args.model)
    source_path = Path(args.source)
    save_dir = Path(args.save_dir)
    predict(model, source_path, save_dir, args.conf)


if __name__ == "__main__":
    main()
