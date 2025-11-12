# Box Inspection System - Complete Usage Guide

## 📋 System Overview

This is an **automated box inspection system** that uses computer vision to:
1. **Detect a box** placed under a camera
2. **Align and crop** the box region (handles rotation)
3. **Detect objects** inside the box using YOLO
4. **Verify completeness** - check if all expected objects are present

---

## ✅ Code Quality Assessment

### What Works Correctly
✓ **All Python files have no syntax errors**
✓ **Module imports are properly structured**
✓ **Camera abstraction supports both IDS and OpenCV**
✓ **YOLO model files are present** (`firstmodelv1.pt` exists)
✓ **Object detection pipeline is well-designed**
✓ **Error handling is robust throughout**

### Potential Issues & Solutions

#### 1. **Box Detection Color Thresholds**
```python
# In box_detector.py, line 24:
self.box_color_lower = np.array([0, 0, 0])
self.box_color_upper = np.array([180, 255, 80])
```
**Issue**: These HSV values detect **dark objects** (low V=80). If your box is bright or colored differently, detection will fail.

**Solution**: You may need to calibrate these values for your specific box. See "Calibration" section below.

#### 2. **Model Path Inconsistency**
```python
# main.py uses:
YOLO_MODEL = model_dir / 'firstmodelv1.pt'

# But also references in comments:
# 'yolov8s.pt', 'yolov8m.pt'
```
**Status**: ✓ `firstmodelv1.pt` exists, so this is fine. Just ensure it's trained for your custom objects.

#### 3. **Expected Objects**
```python
EXPECTED_OBJECTS = {
    'filters':1,
    'milkjug':1,
    'stamp':1,
    'tool':1,
    'tray':1,
    'watercontainer':1,
    'wood':1
}
```
**Requirement**: Your YOLO model (`firstmodelv1.pt`) **must be trained** to recognize these exact class names. If the model was trained on different classes, you'll get no detections.

---

## 🚀 How to Run the System

### Prerequisites

1. **Install dependencies** (in your Pipenv):
```powershell
cd 'C:\Users\Timo\OneDrive - Hochschule Luzern\Studium\Semester 5\PAIND\part_detector'
pipenv install numpy opencv-python ultralytics
```

2. **Camera setup**:
   - **IDS uEye**: Install IDS SDK + Python bindings (`ids_peak`)
   - **Webcam**: Any USB camera will work as fallback

3. **YOLO model**: Ensure `models/firstmodelv1.pt` is trained on your target objects

### Basic Usage

#### Option A: Run with Default Settings
```powershell
pipenv run python main.py
```

This will:
- Initialize the camera (IDS if available, else webcam)
- Open a window showing live detection
- Display "COMPLETE" when all 7 objects are detected
- Press **'q'** to quit

#### Option B: Modify Configuration Before Running
Edit `main.py` to customize:

```python
# Change the YOLO model
YOLO_MODEL = model_dir / 'yolov8n.pt'  # Use pretrained COCO model

# Change expected objects
EXPECTED_OBJECTS = {
    'bottle': 2,    # Expect 2 bottles
    'cup': 1,       # Expect 1 cup
}

# Change verification mode
system.verification_mode = 'minimum'  # Options: 'exact', 'minimum', 'any'
#   'exact'   = must have EXACTLY the specified count
#   'minimum' = must have AT LEAST the specified count
#   'any'     = must have at least 1 of each type (ignores count)

# Change detection sensitivity
system.required_consecutive_detections = 5  # More stable but slower
system.object_detector.confidence_threshold = 0.7  # Higher = fewer false positives
```

---

## 🎯 Understanding the Pipeline

### Step-by-Step Process

```
Camera Frame → Box Detection → Rotation Correction → Crop Box → YOLO Detection → Verification
```

#### 1. **Camera Capture** (`camera_capture.py`)
- Tries IDS uEye camera first
- Falls back to system webcam if IDS unavailable
- Returns BGR image (OpenCV format)

#### 2. **Box Detection** (`box_detector.py`)
```python
box_detector.process_frame(image, visualize=True)
```
- Converts to HSV color space
- Applies color threshold to isolate dark box
- Finds largest contour (the box)
- Calculates rotation angle
- Rotates image to align box horizontally
- Crops just the box region (640x640)
- Returns processed image ready for YOLO

#### 3. **Object Detection** (`object_detector.py`)
```python
object_detector.detect_objects(processed_frame)
```
- Runs YOLO inference on cropped box
- Returns list of detections with bounding boxes
- Each detection has: class_name, confidence, bbox

#### 4. **Verification** (`main.py`)
```python
object_detector.check_expected_objects(detections, EXPECTED_OBJECTS, mode='minimum')
```
- Counts detected objects
- Compares with expected counts
- Returns: is_complete, missing_objects, extra_objects

#### 5. **State Management**
- Requires 3 consecutive complete detections to trigger "COMPLETE"
- Prevents false positives from temporary detection glitches
- Resets if box is removed or objects change

---

## 🔧 Customization Guide

### 1. **Calibrate Box Detection Colors**

If the box is not detected, you need to adjust color thresholds:

```python
# Create a test script: test_box_colors.py
import cv2
import numpy as np
from camera_capture import CameraCapture

camera = CameraCapture()
camera.initialize()

def nothing(x):
    pass

cv2.namedWindow('HSV Tuning')
cv2.createTrackbar('H_low', 'HSV Tuning', 0, 180, nothing)
cv2.createTrackbar('S_low', 'HSV Tuning', 0, 255, nothing)
cv2.createTrackbar('V_low', 'HSV Tuning', 0, 255, nothing)
cv2.createTrackbar('H_high', 'HSV Tuning', 180, 180, nothing)
cv2.createTrackbar('S_high', 'HSV Tuning', 255, 255, nothing)
cv2.createTrackbar('V_high', 'HSV Tuning', 80, 255, nothing)

while True:
    frame = camera.capture_frame()
    if frame is None:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    h_low = cv2.getTrackbarPos('H_low', 'HSV Tuning')
    s_low = cv2.getTrackbarPos('S_low', 'HSV Tuning')
    v_low = cv2.getTrackbarPos('V_low', 'HSV Tuning')
    h_high = cv2.getTrackbarPos('H_high', 'HSV Tuning')
    s_high = cv2.getTrackbarPos('S_high', 'HSV Tuning')
    v_high = cv2.getTrackbarPos('V_high', 'HSV Tuning')
    
    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])
    
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\nUse these values in box_detector.py:")
        print(f"box_color_lower = np.array([{h_low}, {s_low}, {v_low}])")
        print(f"box_color_upper = np.array([{h_high}, {s_high}, {v_high}])")
        break

camera.release()
cv2.destroyAllWindows()
```

Run this, adjust trackbars until the box is white in the mask, then update `box_detector.py`.

### 2. **Use Pretrained YOLO Model (for Testing)**

If your custom model isn't working, test with COCO pretrained model:

```python
# In main.py
YOLO_MODEL = model_dir / 'yolov8n.pt'  # Has 80 common objects

EXPECTED_OBJECTS = {
    'person': 1,
    'bottle': 2,
    'cell phone': 1,
}
```

This uses standard COCO classes - good for testing the pipeline.

### 3. **Custom Callbacks**

Modify `on_box_complete()` to trigger actions:

```python
def on_box_complete(self):
    """Triggered when all objects detected"""
    print("✓ BOX COMPLETE!")
    
    # Example: Send HTTP request
    # import requests
    # requests.post('http://your-server/api/box-complete')
    
    # Example: Save timestamp to file
    # with open('completion_log.txt', 'a') as f:
    #     f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Example: Play sound
    # import winsound
    # winsound.Beep(1000, 500)  # 1000 Hz for 500ms
    
    # Example: Serial port signal (Arduino/PLC)
    # import serial
    # ser = serial.Serial('COM3', 9600)
    # ser.write(b'COMPLETE\n')
```

### 4. **Headless Mode (No Display)**

For deployment without GUI:

```python
# In main.py, change:
system.run(display_window=False)

# Or modify process loop to log to file instead
```

---

## 📊 Testing Strategy

### Test 1: Camera Only
```python
# test_camera.py
from camera_capture import CameraCapture
import cv2

cam = CameraCapture()
if cam.initialize():
    print("✓ Camera OK:", cam.get_camera_info())
    for _ in range(10):
        frame = cam.capture_frame()
        if frame is not None:
            cv2.imshow('Test', frame)
            cv2.waitKey(100)
    cam.release()
else:
    print("✗ Camera failed")
```

### Test 2: Box Detection Only
```python
# test_box.py
from camera_capture import CameraCapture
from box_detector import BoxDetector
import cv2

cam = CameraCapture()
cam.initialize()
detector = BoxDetector(min_box_area=50000)

while True:
    frame = cam.capture_frame()
    if frame is None:
        continue
    
    processed, vis = detector.process_frame(frame, visualize=True)
    
    if processed is not None:
        cv2.imshow('Processed Box', processed)
    cv2.imshow('Detection', vis)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
```

### Test 3: YOLO Only (No Box Detection)
```python
# test_yolo.py
from camera_capture import CameraCapture
from object_detector import ObjectDetector
import cv2

cam = CameraCapture()
cam.initialize()
detector = ObjectDetector(model_path='models/firstmodelv1.pt')

while True:
    frame = cam.capture_frame()
    if frame is None:
        continue
    
    detections = detector.detect_objects(frame, verbose=True)
    vis = detector.visualize_detections(frame, detections)
    cv2.imshow('YOLO Test', vis)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
```

---

## 🐛 Troubleshooting

### Problem: "No camera available"
**Cause**: Neither IDS nor webcam detected
**Fix**: 
1. Check webcam is connected: `ls (Get-PnpDevice -Class Camera)`
2. Try different camera_id: `CameraCapture(camera_id=1)`

### Problem: "Box not detected"
**Cause**: Color threshold doesn't match your box
**Fix**: Run HSV calibration script (see Customization section)

### Problem: "No objects detected" 
**Cause**: YOLO model doesn't recognize those classes
**Fix**: 
1. Check model classes: `detector.class_names`
2. Use COCO pretrained model for testing
3. Retrain model on your specific objects

### Problem: System is too slow
**Solutions**:
```python
# Reduce resolution
camera.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Use smaller YOLO model
YOLO_MODEL = 'yolov8n.pt'  # nano (fastest)

# Increase frame skip
time.sleep(0.1)  # ~10 FPS instead of 20
```

### Problem: False "COMPLETE" detections
**Solutions**:
```python
# Require more consecutive detections
system.required_consecutive_detections = 5  # Instead of 3

# Increase confidence threshold
system.object_detector.confidence_threshold = 0.7  # Instead of 0.5
```

---

## 📁 File Structure

```
part_detector/
├── main.py                  # Main application entry point
├── camera_capture.py        # Camera abstraction (IDS/OpenCV)
├── box_detector.py          # Box detection & alignment
├── object_detector.py       # YOLO object detection
├── ueye.py                  # IDS camera utility (standalone)
├── models/
│   ├── firstmodelv1.pt     # Your custom trained model ✓
│   ├── yolov8n.pt          # Pretrained COCO model
│   └── yolo11n.pt          # Newer pretrained model
├── requirements.txt         # Python dependencies
└── camsettings.cset        # IDS camera calibration file
```

---

## 🎓 How the Code Works (Architecture)

### Class Hierarchy
```
BoxInspectionSystem (main.py)
├── CameraCapture (camera_capture.py)
│   ├── IDS uEye backend
│   └── OpenCV backend
├── BoxDetector (box_detector.py)
│   ├── detect_box()
│   ├── rotate_image()
│   └── crop_box_region()
└── ObjectDetector (object_detector.py)
    ├── YOLO model
    ├── detect_objects()
    └── check_expected_objects()
```

### Data Flow
```
1. Camera → BGR image (numpy array)
2. BoxDetector → Cropped 640x640 aligned box
3. YOLO → List of detections [{class_name, bbox, confidence}, ...]
4. Verification → Boolean (complete/incomplete) + missing objects
5. Visualization → Annotated image with boxes and status
6. State Machine → Trigger callbacks when complete
```

---

## 🚀 Quick Start Checklist

- [ ] Dependencies installed: `pipenv install numpy opencv-python ultralytics`
- [ ] Camera connected and working
- [ ] YOLO model exists: `models/firstmodelv1.pt`
- [ ] Model trained on your objects (filters, milkjug, stamp, tool, tray, watercontainer, wood)
- [ ] Box color threshold calibrated (if needed)
- [ ] Run: `pipenv run python main.py`
- [ ] Press 'q' to quit

---

## 💡 Next Steps

1. **Test individual components** (camera → box → YOLO)
2. **Calibrate box detection** if needed
3. **Verify YOLO model classes** match expected objects
4. **Adjust thresholds** for your environment
5. **Add custom callbacks** for production integration

---

## ⚠️ Important Notes

1. **YOLO Model Training**: Your `firstmodelv1.pt` must be trained specifically for:
   - filters, milkjug, stamp, tool, tray, watercontainer, wood
   - If not, the system won't detect these objects

2. **Box Appearance**: The current color threshold detects **dark boxes**. Bright/white boxes won't work without recalibration.

3. **Lighting**: Good, consistent lighting is critical for both box detection and YOLO accuracy.

4. **Camera Position**: Camera should be mounted directly above the box, looking down.

5. **IDS Camera**: Requires SDK installation. Without it, system falls back to webcam automatically.

---

**Status**: ✅ Code is well-structured and should work IF:
- YOLO model is properly trained
- Box color threshold matches your box
- Camera is connected
- Dependencies are installed

The code quality is good with robust error handling throughout!
