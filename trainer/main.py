from ultralytics import YOLO
from pathlib import Path
import torch

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")
path = Path(__file__).parent / "datasetv2" / "data.yaml"


def main():

    def train_model(modelname):
        first_model = model.train(

                #hyperparameters
                data=path,  # Path to dataset configuration file
                epochs=200,  # Number of training epochs
                imgsz=640,  # Image size for training
                batch=-1,
                patience=30,
                #lr0=0.01, #initial learning rate
                #lrf=0.01, #final learningrate
                name=modelname,  # Name of the training experiment
                optimizer="auto",
                plots=True,

                #augentation
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.6,
                translate=0.1,
                scale=0.3,
                fliplr=0.5,
                flipud=0.5,
                mosaic=0.9,
                degrees=180,
                

            )

    # Fine tune the pretrained model on custom dataset

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