"""
Test Script 1: Camera Functionality
Tests if camera initializes and captures frames correctly.
"""

from camera_capture import CameraCapture
import cv2
import time

def test_camera():
    print("=" * 60)
    print("Camera Test")
    print("=" * 60)
    
    # Initialize camera
    print("\n1. Initializing camera...")
    cam = CameraCapture()
    
    if not cam.initialize():
        print("✗ FAILED: Could not initialize camera")
        return False
    
    print("✓ Camera initialized successfully")
    
    # Get camera info
    info = cam.get_camera_info()
    print(f"\n2. Camera Information:")
    print(f"   Model: {info['model']}")
    print(f"   Resolution: {info['resolution']}")
    print(f"   Status: {info['status']}")
    print(f"   Backend: {'IDS uEye' if cam.use_ids else 'OpenCV VideoCapture'}")
    
    # Capture test frames
    print(f"\n3. Capturing test frames...")
    success_count = 0
    fail_count = 0
    
    for i in range(10):
        frame = cam.capture_frame()
        if frame is not None:
            success_count += 1
            if i == 0:  # Show first frame
                cv2.imshow('Camera Test - Press any key to continue', frame)
                print(f"   Frame shape: {frame.shape}")
                print("   Showing first frame... press any key in the window")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            fail_count += 1
        
        time.sleep(0.1)
    
    print(f"   Successful captures: {success_count}/10")
    print(f"   Failed captures: {fail_count}/10")
    
    # Cleanup
    print("\n4. Cleaning up...")
    cam.release()
    print("✓ Camera released")
    
    # Results
    print("\n" + "=" * 60)
    if success_count >= 8:
        print("✓ TEST PASSED: Camera is working properly")
        print("=" * 60)
        return True
    else:
        print("✗ TEST FAILED: Camera is not working reliably")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = test_camera()
    exit(0 if success else 1)
