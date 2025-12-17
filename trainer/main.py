from ultralytics import YOLO
from pathlib import Path
import torch

# Load a pretrained YOLO11n model
model = YOLO("yolo11s.pt")
path = Path(__file__).parent / "datasetv4" / "data.yaml"


def main():

    def train_model(modelname):
        results = model.train(

            # Hyperparameters
            data=path,  # Path to dataset configuration file
            epochs=250,  # Number of training epochs
            imgsz=640,  # Image size for training (pixels)
            batch=-1,  # Batch size (-1 for auto-batch)
            patience=50,  # Early stopping patience (epochs without improvement)
            #lr0=0.01, # Initial learning rate
            #lrf=0.01, # Final learning rate (as fraction of lr0)
            name=modelname,  # Name of the training experiment
            optimizer="auto",  # Optimizer selection (auto/SGD/Adam/AdamW)
            plots=True,  # Generate training plots

            augment=True,
            # Augmentation
            hsv_h=0.015,  # HSV-Hue augmentation (fraction)
            hsv_s=0.4,  # HSV-Saturation augmentation (fraction)
            hsv_v=0.4,  # Brightness-Value augmentation (fraction)
            translate=0.1,  # Image translation (+/- fraction)
            scale=0.5,  # Image scale (+/- gain)
            fliplr=0.5,  # Horizontal flip probability
            flipud=0.5,  # Vertical flip probability
            mosaic=0.5,  # Mosaic augmentation probability
            degrees=180,  # Rotation range (+/- degrees)

            )

   
    #names model
    model_name=input("Name your custom trained model: ")
    if model_name=="":
        print("No Name entered. Standard Naming will be implemented")
        model_name=None
    else:
        print(f"model will be named: {model_name}")

    if path.exists():
        print("Path exists ")
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            train_model(model_name)
        else:
            print("CUDA is not available. Using CPU for training.")
            cpu_yes=input("Do you want to continue training on CPU? (y/n): ")
            if cpu_yes.lower() == 'y':
                train_model(model_name)
            else:
                print("Training aborted by the user.")
    else:
        print(f"Dataset configuration file not found at {path}")


if __name__ == "__main__":
    main()