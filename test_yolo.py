"""
Test Script 3: YOLO Object Detection
Tests YOLO model on live camera feed (without box detection).
Press 'q' to quit.
"""

from camera_capture import CameraCapture
from object_detector import ObjectDetector
from pathlib import Path
import cv2
import time

def test_yolo():
    print("=" * 60)
    print("YOLO Object Detection Test")
    print("=" * 60)
    
    # Select model
    model_dir = Path(__file__).parent / 'models'
    model_path = model_dir / 'firstmodelv1.pt'
    
    print(f"\nModel: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    
    if not model_path.exists():
        print("✗ FAILED: Model file not found")
        return False
    
    # Initialize camera
    print("\nInitializing camera...")
    cam = CameraCapture()
    if not cam.initialize():
        print("✗ FAILED: Could not initialize camera")
        return False
    
    # Initialize YOLO
    print("Loading YOLO model...")
    try:
        detector = ObjectDetector(
            model_path=str(model_path),
            confidence_threshold=0.5
        )
    except Exception as e:
        print(f"✗ FAILED: Could not load YOLO model: {e}")
        cam.release()
        return False
    
    print(f"\n✓ Model loaded successfully")
    print(f"Available classes ({len(detector.class_names)}):")
    for i, name in enumerate(detector.class_names.values()):
        print(f"  {i}: {name}")
    
    print("\n" + "=" * 60)
    print("Instructions:")
    print("  - Point camera at objects")
    print("  - Detections will be shown with bounding boxes")
    print("  - Press 'q' to quit")
    print("=" * 60)
    
    frame_count = 0
    total_detections = 0
    detection_history = {}
    
    try:
        while True:
            # Capture frame
            frame = cam.capture_frame()
            if frame is None:
                print("Warning: Failed to capture frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Detect objects
            detections = detector.detect_objects(frame, verbose=False)
            total_detections += len(detections)
            
            # Track what we've detected
            for det in detections:
                class_name = det['class_name']
                detection_history[class_name] = detection_history.get(class_name, 0) + 1
            
            # Visualize
            vis = detector.visualize_detections(frame, detections, show_confidence=True)
            
            # Add statistics overlay
            cv2.putText(vis, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, f"Objects: {len(detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show detected classes
            y_offset = 90
            summary = detector.get_detection_summary(detections)
            for obj_name, count in summary.items():
                cv2.putText(vis, f"{obj_name}: {count}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            cv2.imshow('YOLO Detection Test', vis)
            
            # Print detections to console (first 5 frames only)
            if frame_count <= 5 and len(detections) > 0:
                print(f"\nFrame {frame_count} detections:")
                for det in detections:
                    print(f"  - {det['class_name']}: {det['confidence']:.2f}")
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            time.sleep(0.05)  # ~20 FPS
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Cleanup
        cam.release()
        cv2.destroyAllWindows()
        
        # Statistics
        print("\n" + "=" * 60)
        print("Test Statistics:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Total detections: {total_detections}")
        if frame_count > 0:
            print(f"  Average detections/frame: {total_detections/frame_count:.2f}")
        
        if detection_history:
            print(f"\nDetection History (all frames):")
            for obj_name, count in sorted(detection_history.items(), key=lambda x: x[1], reverse=True):
                print(f"  {obj_name}: {count} times")
            print("\n✓ TEST PASSED: YOLO is detecting objects")
            return True
        else:
            print("\n⚠ WARNING: No objects detected")
            print("  Possible reasons:")
            print("    1. No objects in view")
            print("    2. Objects not in trained classes")
            print("    3. Confidence threshold too high")
            print("    4. Model not properly trained")
            return False
        
        print("=" * 60)

if __name__ == "__main__":
    success = test_yolo()
    exit(0 if success else 1)
