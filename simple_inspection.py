# simple_inspection.py
from ultralytics import YOLO
import cv2

class SimpleInspection:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.model = YOLO('yolov8n.pt')
    
    def run(self):
        while True:
            ret, frame = self.camera.read()
            
            # Step 1: Detect objects with YOLO
            results = self.model(frame)
            
            # Step 2: Count what we found
            detections = results[0].boxes
            object_count = len(detections)
            
            # Step 3: Show it
            annotated = results[0].plot()
            cv2.putText(annotated, f'Objects: {object_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Inspection', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.camera.release()

# Run it
system = SimpleInspection()
system.run()