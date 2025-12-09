"""
Test Mode - Run the system with a static test image or video file
Use this when no camera is available for testing purposes.
"""

import cv2
import numpy as np
from pathlib import Path

def create_test_image(width=1920, height=1080):
    """
    Creates a test image with some objects drawn on it.
    You can replace this with an actual image file if you have one.
    """
    # Create a blank image (white background)
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw a colored box (simulating the box we want to detect)
    box_color = (100, 150, 200)  # BGR
    cv2.rectangle(img, (200, 200), (1700, 900), box_color, -1)
    
    # Add some text
    cv2.putText(img, "TEST MODE - No Camera Connected", 
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(img, "Replace this with real camera feed", 
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Draw some colored rectangles (simulating objects)
    cv2.rectangle(img, (400, 400), (600, 600), (0, 255, 0), -1)
    cv2.putText(img, "Object 1", (420, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.rectangle(img, (800, 400), (1000, 600), (255, 0, 0), -1)
    cv2.putText(img, "Object 2", (820, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.rectangle(img, (1200, 400), (1400, 600), (0, 0, 255), -1)
    cv2.putText(img, "Object 3", (1220, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

class TestCamera:
    """
    Mock camera class that returns a test image instead of real camera feed.
    Drop-in replacement for CameraCapture when no camera is available.
    """
    
    def __init__(self, camera_id=0, image_path=None):
        """
        Args:
            camera_id: Ignored in test mode
            image_path: Optional path to use a real image file instead of generated one
        """
        self.camera_id = camera_id
        self.image_path = image_path
        self.test_image = None
        self.is_initialized = False
        self.use_ids = False
        
    def initialize(self):
        """Initialize test mode camera."""
        print("\n" + "=" * 60)
        print("TEST MODE ACTIVE - Using simulated camera")
        print("=" * 60)
        
        if self.image_path and Path(self.image_path).exists():
            # Load real image file
            self.test_image = cv2.imread(self.image_path)
            if self.test_image is not None:
                print(f"✓ Loaded test image from: {self.image_path}")
                h, w = self.test_image.shape[:2]
                print(f"  Image size: {w}x{h}")
            else:
                print(f"✗ Failed to load image from: {self.image_path}")
                print("  Using generated test image instead")
                self.test_image = create_test_image()
        else:
            # Generate test image
            self.test_image = create_test_image()
            print("✓ Generated test image (1920x1080)")
            
        print("  Note: All frames will be identical (no live feed)")
        print("  Connect a real camera for actual detection")
        print("=" * 60 + "\n")
        
        self.is_initialized = True
        return True
        
    def capture_frame(self):
        """Return the test image."""
        if not self.is_initialized or self.test_image is None:
            return None
        # Return a copy so it can be modified without affecting the original
        return self.test_image.copy()
        
    def get_camera_info(self):
        """Return test camera info."""
        if self.test_image is not None:
            h, w = self.test_image.shape[:2]
            return {
                "model": "Test Camera (Simulated)",
                "width": w,
                "height": h,
                "status": "active (test mode)"
            }
        return {"model": "Test Camera", "status": "not initialized"}
        
    def release(self):
        """Clean up test camera."""
        self.is_initialized = False
        print("✓ Test camera released")

def patch_main_for_test_mode():
    """
    Instructions to modify main.py to use test mode:
    
    In main.py, replace the import:
        FROM: from camera_capture import CameraCapture
        TO:   from test_mode import TestCamera as CameraCapture
        
    Or add a command line argument to switch between modes.
    """
    pass

if __name__ == "__main__":
    # Demo of test camera
    print("Testing TestCamera class...\n")
    
    camera = TestCamera()
    
    if camera.initialize():
        print("\nCapturing frames...")
        for i in range(3):
            frame = camera.capture_frame()
            if frame is not None:
                print(f"  Frame {i+1}: {frame.shape}")
                
        print("\nCamera info:")
        info = camera.get_camera_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
            
        # Show the test image
        cv2.imshow("Test Mode Preview", camera.capture_frame())
        print("\nPress any key in the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        camera.release()
        print("\n✓ Test completed successfully")
