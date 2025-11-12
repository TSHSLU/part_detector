"""
Test Script 2: Box Detection
Tests box detection and alignment functionality.
Press 'q' to quit, 's' to save current frame.
"""

from camera_capture import CameraCapture
from box_detector import BoxDetector
import cv2
import time

def test_box_detection():
    print("=" * 60)
    print("Box Detection Test")
    print("=" * 60)
    print("Instructions:")
    print("  - Place box under camera")
    print("  - Green outline should appear around box")
    print("  - Right window shows cropped/aligned box")
    print("  - Press 's' to save current frame")
    print("  - Press 'q' to quit")
    print("=" * 60)
    
    # Initialize camera
    cam = CameraCapture()
    if not cam.initialize():
        print("✗ FAILED: Could not initialize camera")
        return False
    
    # Initialize box detector
    # Adjust min_box_area based on your setup
    detector = BoxDetector(min_box_area=50000)
    print(f"\nBox detector settings:")
    print(f"  Color range (HSV): {detector.box_color_lower} to {detector.box_color_upper}")
    print(f"  Minimum area: {detector.min_box_area} pixels")
    
    frame_count = 0
    detected_count = 0
    save_counter = 0
    
    try:
        while True:
            # Capture frame
            frame = cam.capture_frame()
            if frame is None:
                print("Warning: Failed to capture frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Process frame
            processed, vis = detector.process_frame(frame, visualize=True)
            
            if processed is not None:
                detected_count += 1
                
                # Show both visualization and processed result
                cv2.imshow('Camera View (Box Detection)', vis)
                cv2.imshow('Processed Box (YOLO Input)', processed)
                
                # Add statistics on visualization
                stats_text = f"Detected: {detected_count}/{frame_count} frames"
                cv2.putText(vis, stats_text, (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # Box not detected
                no_box_frame = vis.copy()
                cv2.putText(no_box_frame, "NO BOX DETECTED", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(no_box_frame, f"Frames: {frame_count}", (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('Camera View (Box Detection)', no_box_frame)
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s') and processed is not None:
                # Save current frame
                save_counter += 1
                filename = f"box_test_{save_counter:03d}.png"
                cv2.imwrite(filename, processed)
                print(f"✓ Saved: {filename}")
            
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
        print(f"  Total frames: {frame_count}")
        print(f"  Boxes detected: {detected_count}")
        if frame_count > 0:
            detection_rate = (detected_count / frame_count) * 100
            print(f"  Detection rate: {detection_rate:.1f}%")
            
            if detection_rate >= 80:
                print("\n✓ TEST PASSED: Box detection is working well")
                return True
            elif detection_rate >= 50:
                print("\n⚠ TEST PARTIAL: Box detection is inconsistent")
                print("  Try adjusting box_color_lower/upper values")
                return False
            else:
                print("\n✗ TEST FAILED: Box detection not working")
                print("  Check:")
                print("    1. Box is visible in camera")
                print("    2. Color threshold matches box color")
                print("    3. Box area > min_box_area")
                return False
        print("=" * 60)

if __name__ == "__main__":
    success = test_box_detection()
    exit(0 if success else 1)
