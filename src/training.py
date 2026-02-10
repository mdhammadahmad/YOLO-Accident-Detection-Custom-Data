# Train YOLO model on a custom dataset.

from ultralytics import YOLO

def train_custom_yolo(
    model_path,
    data_yaml,
    epochs=120,
    imgsz=640,
    batch=8,
    device=0,
    project_dir=None,
    run_name="experiment"
):

    model = YOLO(model_path)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_dir,
        name=run_name
    )

    return results
