import os

from ultralytics import YOLO
import torch

def main():

    project_path = os.path.abspath("runs/detect/outputs")

    # Check if GPU is available
    if torch.cuda.is_available():
        device = 0
        print("✅ GPU detected:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("⚠ GPU not detected, using CPU")

    # Load pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")
    
    # Train the model
    model.train(
        data="configs/dataset.yaml",   # dataset config
        epochs=10,                     # training epochs
        imgsz=640,                     # image size
        batch=16,                      # batch size (good for RTX 3050)
        device=device,                 # GPU
        project=project_path,           # Aligned with app.py
        name="smartwastenet_training", # Aligned with app.py
        workers=2,                     # faster data loading
        optimizer="auto",              # optimizer selection
        patience=10,                   # early stopping
        cache="disk"                   # added for faster data loading
    )

    print("\n🎉 Training finished!")

if __name__ == "__main__":
    main()