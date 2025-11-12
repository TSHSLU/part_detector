"""
HSV Color Calibration Tool
Use this to find the correct HSV color range for your box.

Instructions:
1. Run this script
2. Point camera at the box
3. Adjust the trackbars until the box appears WHITE in the "Mask" window
4. The "Result" window shows what will be detected
5. Press 'q' when satisfied - it will print the values to use
"""

import cv2
import numpy as np
from camera_capture import CameraCapture

def nothing(x):
    """Trackbar callback (does nothing)"""
    pass

def calibrate_colors():
    print("=" * 60)
    print("HSV Color Calibration Tool")
    print("=" * 60)
    print("Instructions:")
    print("  1. Adjust trackbars until BOX is WHITE in Mask window")
    print("  2. Everything else should be BLACK")
    print("  3. Press 'q' when satisfied")
    print("  4. Copy the printed values to box_detector.py")
    print("=" * 60 + "\n")
    
    # Initialize camera
    camera = CameraCapture()
    if not camera.initialize():
        print("✗ FAILED: Could not initialize camera")
        return
    
    print("✓ Camera initialized\n")
    
    # Create window and trackbars
    cv2.namedWindow('HSV Calibration')
    
    # HSV ranges: H=0-180, S=0-255, V=0-255
    # Default values for dark box detection
    cv2.createTrackbar('H_low', 'HSV Calibration', 0, 180, nothing)
    cv2.createTrackbar('S_low', 'HSV Calibration', 0, 255, nothing)
    cv2.createTrackbar('V_low', 'HSV Calibration', 0, 255, nothing)
    cv2.createTrackbar('H_high', 'HSV Calibration', 180, 180, nothing)
    cv2.createTrackbar('S_high', 'HSV Calibration', 255, 255, nothing)
    cv2.createTrackbar('V_high', 'HSV Calibration', 80, 255, nothing)
    
    print("Calibrating... (press 'q' to finish)\n")
    
    while True:
        # Capture frame
        frame = camera.capture_frame()
        if frame is None:
            print("Warning: Failed to capture frame")
            continue
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get current trackbar positions
        h_low = cv2.getTrackbarPos('H_low', 'HSV Calibration')
        s_low = cv2.getTrackbarPos('S_low', 'HSV Calibration')
        v_low = cv2.getTrackbarPos('V_low', 'HSV Calibration')
        h_high = cv2.getTrackbarPos('H_high', 'HSV Calibration')
        s_high = cv2.getTrackbarPos('S_high', 'HSV Calibration')
        v_high = cv2.getTrackbarPos('V_high', 'HSV Calibration')
        
        # Create threshold
        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Show what will be detected
        result = cv2.bitwise_and(frame, frame, mask=mask_cleaned)
        
        # Find contours to show detection
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = frame.copy()
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            cv2.drawContours(vis, [largest], 0, (0, 255, 0), 3)
            cv2.putText(vis, f"Area: {area:.0f} pixels", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display windows
        cv2.imshow('Original', vis)
        cv2.imshow('Mask (box should be WHITE)', mask_cleaned)
        cv2.imshow('Result (detected region)', result)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n" + "=" * 60)
            print("Calibration Complete!")
            print("=" * 60)
            print("\nCopy these values to box_detector.py:\n")
            print(f"box_color_lower = np.array([{h_low}, {s_low}, {v_low}])")
            print(f"box_color_upper = np.array([{h_high}, {s_high}, {v_high}])")
            print("\nOr pass them to BoxDetector constructor:")
            print(f"detector = BoxDetector(")
            print(f"    box_color_lower=np.array([{h_low}, {s_low}, {v_low}]),")
            print(f"    box_color_upper=np.array([{h_high}, {s_high}, {v_high}])")
            print(f")")
            print("=" * 60)
            break
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate_colors()
