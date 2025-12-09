"""
Camera Diagnostics Tool
Tests different backends and camera indices to find available cameras.
"""

import cv2
import sys

def test_camera_backend(index, backend_name, backend_flag):
    """Test a specific camera index with a specific backend."""
    print(f"\nTesting Camera {index} with {backend_name}...")
    try:
        cap = cv2.VideoCapture(index, backend_flag)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✓ SUCCESS: Camera {index} works with {backend_name} ({w}x{h})")
                cap.release()
                return True
            else:
                print(f"  ✗ FAILED: Camera opened but couldn't read frame")
                cap.release()
                return False
        else:
            print(f"  ✗ FAILED: Couldn't open camera")
            return False
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False

def main():
    print("=" * 60)
    print("Camera Diagnostics Tool")
    print("=" * 60)
    
    # Test backends
    backends = [
        ("DSHOW (DirectShow)", cv2.CAP_DSHOW),
        ("MSMF (Media Foundation)", cv2.CAP_MSMF),
        ("ANY (Auto-detect)", cv2.CAP_ANY),
    ]
    
    working_configs = []
    
    # Test first 3 camera indices
    for index in range(3):
        print(f"\n{'=' * 60}")
        print(f"Testing Camera Index {index}")
        print('=' * 60)
        
        for backend_name, backend_flag in backends:
            if test_camera_backend(index, backend_name, backend_flag):
                working_configs.append((index, backend_name, backend_flag))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if working_configs:
        print("\n✓ Found working camera configurations:")
        for idx, name, flag in working_configs:
            print(f"  - Camera {idx} with {name}")
        
        # Suggest fix
        print("\n" + "=" * 60)
        print("RECOMMENDED FIX")
        print("=" * 60)
        best = working_configs[0]
        print(f"\nUse camera index {best[0]} with {best[1]}")
        
        if best[2] == cv2.CAP_MSMF:
            print("\nIn camera_capture.py, line ~149, change:")
            print("  FROM: self.camera = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)")
            print("  TO:   self.camera = cv2.VideoCapture(self.camera_id, cv2.CAP_MSMF)")
        elif best[2] == cv2.CAP_ANY:
            print("\nIn camera_capture.py, line ~149, change:")
            print("  FROM: self.camera = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)")
            print("  TO:   self.camera = cv2.VideoCapture(self.camera_id)")
            
    else:
        print("\n✗ No working cameras found!")
        print("\nPossible causes:")
        print("  1. No camera is connected to your computer")
        print("  2. Camera is in use by another application")
        print("  3. Camera permissions are blocked")
        print("  4. Camera drivers need to be updated")
        print("\nTroubleshooting steps:")
        print("  1. Check if camera works in Windows Camera app")
        print("  2. Close all other apps that might use the camera")
        print("  3. Check Windows Privacy Settings > Camera permissions")
        print("  4. Try reconnecting/rebooting the camera")

if __name__ == "__main__":
    main()
