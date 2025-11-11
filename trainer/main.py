from ultralytics import YOLO
from pathlib import Path
import torch

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")
path = Path(__file__).parent / "datasetv1" / "dataset_annotated_v2" / "data.yaml"


def main():

    def train_model():
        first_model = model.train(
                data=path,  # Path to dataset configuration file
                epochs=100,  # Number of training epochs
                imgsz=640,  # Image size for training
                name="minimal_model_paind"  # Name of the training experiment
            )
    # Train the model on the COCO8 dataset for 100 epochs
    if path.exists():
        print("Path exists ")
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            train_model()
        else:
            print("CUDA is not available. Using CPU for training.")
            cpu_yes=input("Do you want to continue training on CPU? (y/n): ")
            if cpu_yes.lower() == 'y':
                train_model()
            else:
                print("Training aborted by the user.")
    else:
        print(f"Dataset configuration file not found at {path}")


if __name__ == "__main__":
    main()