from ultralytics import YOLO

# Ensure the model path points to a valid .pt file or correct NCNN model directory
model = YOLO("./modelv5.pt")

results = model("parts2.png", save=True)
for result in results:
	print(result)
