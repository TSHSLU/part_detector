from ultralytics import YOLO

"""# Load a YOLO11n PyTorch model
model = YOLO("modelv2.pt")

# Export the model to NCNN format
model.export(format="ncnn")  # creates 'modelv2_ncnn_model'
"""
# Load the exported NCNN model (use correct name!)
ncnn_model = YOLO("modelv2_ncnn_model")

# Run inference
results = ncnn_model("parts.jpg",save=False,verbose=True,task="detect",show=True)