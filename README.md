# Part Detector - Box Inspection System

A real-time object detection system for inspecting the completeness of parts in a box using YOLO deep learning models and camera capture.

## Overview

This project implements an automated quality control system that:
- Continuously monitors a box placed under a camera
- Detects and identifies parts using YOLO (You Only Look Once) neural networks
- Verifies that all expected parts are present
- Provides visual feedback and notifications when the box is complete
- Supports Raspberry Pi with Sense HAT for headless operation

## Features

### Core Functionality
- **Real-time Object Detection**: Uses state-of-the-art YOLO models (YOLOv8, YOLO11) for fast and accurate part detection
- **Flexible Verification Modes**:
  - `exact`: Exact part counts must match expected quantities
  - `minimum`: At least the specified number of each part must be present
  - `any`: At least one of each expected part type must be present
- **Consecutive Detection Logic**: Requires multiple consecutive complete detections to confirm box completeness (reduces false positives)
- **Live Visualization**: Real-time display of detected parts with bounding boxes and confidence scores

### Hardware Support
- **IDS Industrial Cameras**: Automatic detection and configuration of IDS uEye cameras with custom settings
- **Standard Webcams**: Fallback support for USB webcams and built-in cameras
- **Raspberry Pi Integration**:
  - Sense HAT LED matrix for visual status indication (red/yellow/green)
  - Headless operation mode for embedded deployments
  - Custom status messages on Sense HAT display

### Model Training
- Custom model training pipeline using multiple datasets
- Support for data augmentation (rotation, flip, HSV, mosaic)
- GPU acceleration with CUDA support
- Model versioning and experiment tracking
- NCNN model conversion for edge deployment

## Project Structure

```
part_detector/
├── main.py                     # Main application entry point
├── package/
│   ├── camera_capture.py       # Camera abstraction layer (IDS/OpenCV)
│   ├── object_detector.py      # YOLO detection and verification logic
│   └── path.py                 # Path utilities
├── models/                     # Trained YOLO models
│   ├── modelv2.pt              # Custom trained model v2
│   ├── modelv5.pt              # Custom trained model v5
│   ├── yolo11n.pt              # Pretrained YOLO11 nano
│   ├── converter.py            # NCNN model conversion
│   ├── test_model.py           # Model testing utilities
│   ├── modelv5_ncnn_model/     # NCNN optimized model
│   └── runs/                   # Inference results
├── trainer/
│   ├── main.py                 # Model training script
│   ├── datasetv1-v4/           # Training datasets with annotations
│   └── runs/                   # Training experiment results
├── config/
│   ├── camsettings.cset        # IDS camera configuration
│   └── requirements.txt        # Python dependencies
└── test_files/                 # Testing and diagnostic scripts
```

## Installation

### Requirements
- Python 3.8+
- OpenCV
- Ultralytics YOLO
- NumPy

### Optional Dependencies
- IDS Peak SDK (for IDS industrial cameras)
- Sense HAT library (for Raspberry Pi)
- CUDA toolkit (for GPU acceleration)

### Setup

1. **Clone the repository**:
```bash
cd part_detector
```

2. **Install dependencies**:
```bash
pip install -r config/requirements.txt
```

3. **For IDS camera support**:
   - Install IDS Peak SDK from [IDS Imaging website](https://www.ids-imaging.com/)
   - Install Python bindings: `pip install ids-peak`

4. **For Raspberry Pi**:
```bash
pip install sense-hat
```

## Usage

### Basic Object Detection

1. **Configure expected parts** in [main.py](main.py):
```python
EXPECTED_OBJECTS = {
    'filters': 1,
    'milkjug': 1,
    'stamp': 1,
    'tool': 1,
    'tray': 1,
    'watercontainer': 1,
    'wood': 1
}
```

2. **Select YOLO model**:
```python
YOLO_MODEL = 'models/modelv5_ncnn_model'  # or any .pt file
```

3. **Run the system**:
```bash
python main.py
```

4. **Controls**:
   - Press `q` to quit the application
   - `Ctrl+C` for emergency shutdown

### Training Custom Models

1. **Prepare dataset**:
   - Place annotated images in `trainer/datasetv4/` (or create a new dataset)
   - Ensure `data.yaml` contains class definitions and paths

2. **Run training**:
```bash
cd trainer
python main.py
```

3. **Training configuration**:
   - Epochs: 250 (with early stopping patience of 50)
   - Image size: 640x640
   - Automatic batch size selection
   - Data augmentation enabled (rotation, flip, HSV, mosaic)

4. **GPU vs CPU**:
   - Automatically detects CUDA availability
   - Prompts for confirmation if training on CPU

5. **Results**:
   - Trained models saved in `trainer/runs/detect/<model_name>/`
   - Training plots and metrics available in the same directory

### Configuration Options

**Box Inspection System Parameters**:
```python
system = BoxInspectionSystem(
    yolo_model_path='models/modelv5.pt',
    expected_objects=EXPECTED_OBJECTS,
    conftresh=0.6  # Confidence threshold (0.0-1.0)
)

# Verification settings
system.verification_mode = 'minimum'  # 'exact', 'minimum', or 'any'
system.required_consecutive_detections = 3  # Confirmations needed
```

**Object Detector Parameters**:
- `confidence_threshold`: Minimum confidence for detections (default: 0.5)
- `iou_threshold`: IoU threshold for NMS (default: 0.45)
- `min_box_area`: Minimum bounding box area in pixels (default: 100)

## System Workflow

1. **Initialization**:
   - Camera detection (IDS or fallback to OpenCV)
   - YOLO model loading
   - Raspberry Pi detection (if applicable)
   - Configuration display

2. **Continuous Scanning**:
   - Capture frame from camera
   - Run YOLO inference
   - Count detected objects by class
   - Verify against expected objects

3. **State Management**:
   - Track consecutive complete detections
   - Update Sense HAT LEDs (Raspberry Pi):
     - 🔴 Red: No objects or incomplete
     - 🟡 Yellow: Some objects detected
     - 🟢 Green: Box complete
   - Trigger completion callback when threshold reached

4. **Completion Action**:
   - Display completion message
   - Log timestamp
   - Show "Complete!" on Sense HAT
   - Custom actions can be added in `on_box_complete()`

## Camera Support

### IDS Industrial Cameras
- Auto-detection of IDS cameras
- Loads saved settings from `config/camsettings.cset`
- High-resolution capture with buffer management
- Recommended for production environments

### OpenCV Cameras
- Automatic fallback for standard USB cameras
- Compatible with built-in webcams
- Suitable for development and testing

## Model Formats

- **PyTorch (.pt)**: Standard YOLO format for training and inference
- **NCNN**: Optimized for edge devices and embedded systems
  - Faster inference on ARM processors
  - Lower memory footprint
  - Use `converter.py` to convert .pt to NCNN format

## Datasets

The project includes multiple dataset versions for iterative model improvement:

- **datasetv1**: Initial annotated dataset with basic parts
- **datasetv2-v4**: Progressively improved datasets with:
  - More training examples
  - Better annotations
  - Additional part variations
  - Augmented samples

Each dataset contains:
- `train/`: Training images and labels
- `valid/`: Validation images and labels
- `test/`: Test images and labels
- `data.yaml`: Class definitions and paths

## Development

### Testing
```bash
# Test camera functionality
python test_files/test_camera_diagnostics.py

# Test model inference
python models/test_model.py

# Simple inspection demo
python test_files/simple_inspection.py
```

### Adding New Parts
1. Collect training images with new parts
2. Annotate using Roboflow or similar tools
3. Update `data.yaml` with new class names
4. Retrain model with expanded dataset
5. Update `EXPECTED_OBJECTS` in [main.py](main.py)

### Custom Actions on Completion
Modify the `on_box_complete()` method in [main.py](main.py):
```python
def on_box_complete(self):
    # Add custom logic here:
    # - Send notification
    # - Log to database
    # - Trigger conveyor belt
    # - Sound alarm, etc.
    pass
```

## Troubleshooting

**Camera not detected**:
- Check camera connection
- Verify IDS Peak SDK installation
- Test with `test_camera_diagnostics.py`

**Low detection accuracy**:
- Increase `confidence_threshold`
- Use higher quality camera
- Improve lighting conditions
- Retrain model with more samples

**False positives**:
- Increase `consecutive_complete_detections`
- Raise `confidence_threshold`
- Adjust `min_box_area` to filter noise

**Slow inference**:
- Use smaller YOLO model (yolo11n vs yolo11s)
- Convert to NCNN format
- Enable GPU acceleration
- Reduce input image resolution

## Performance

- **Inference Speed**: ~30-60 FPS (GPU), ~5-15 FPS (CPU)
- **Detection Accuracy**: >90% on trained parts (model v5)
- **False Positive Rate**: <5% with consecutive detection logic

## License

This project was developed as part of the PAIND course at Hochschule Luzern.

## Acknowledgments

- **Ultralytics**: YOLO implementation
- **IDS Imaging**: Industrial camera SDK
- **Roboflow**: Dataset annotation and management
- **Raspberry Pi Foundation**: Embedded platform support

## Contact

For questions or issues, please contact the development team at Hochschule Luzern.

---

**Last Updated**: December 2025
