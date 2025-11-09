"""
Main Program for Object Detection in Box
Continuously scans for objects in a box placed under a camera and verifies completeness.
"""

import cv2
import time
from camera_capture import CameraCapture
from box_detector import BoxDetector
from object_detector import ObjectDetector


class BoxInspectionSystem:
    """
    Main system that orchestrates camera capture, box detection, and object verification.
    """
    
    def __init__(self, yolo_model_path='yolov11s.pt', expected_objects=None):
        """
        Initialize the box inspection system.
        
        Args:
            yolo_model_path (str): Path to YOLO model file
            expected_objects (dict): Dictionary of expected objects and their counts
                                    Example: {'bottle': 2, 'cup': 1, 'person': 1}
                                    If None, system will just report detected objects
        """
        # Initialize components
        self.camera = CameraCapture()
        self.box_detector = BoxDetector(min_box_area=50000)
        self.object_detector = ObjectDetector(
            model_path=yolo_model_path,
            confidence_threshold=0.5
        )
        
        # Configuration
        self.expected_objects = expected_objects or {}
        self.verification_mode = 'minimum'  # 'exact', 'minimum', or 'any'
        
        # State tracking
        self.is_running = False
        self.last_check_time = 0
        self.check_interval = 0.5  # seconds between checks
        self.box_complete = False
        self.consecutive_complete_detections = 0
        self.required_consecutive_detections = 3  # Require 3 consecutive complete detections
        
    def initialize(self):
        """
        Initialize all system components.
        
        Returns:
            bool: True if initialization was successful
        """
        print("=" * 60)
        print("Box Inspection System - Initializing")
        print("=" * 60)
        
        # Initialize camera
        print("\n1. Initializing camera...")
        if not self.camera.initialize():
            print("ERROR: Failed to initialize camera")
            return False
        print("   ✓ Camera initialized")
        
        # Display camera info
        cam_info = self.camera.get_camera_info()
        print(f"   Camera: {cam_info.get('model', 'Unknown')}")
        print(f"   Resolution: {cam_info.get('resolution', 'Unknown')}")
        
        # Display expected objects
        if self.expected_objects:
            print(f"\n2. Expected objects in box (mode: {self.verification_mode}):")
            for obj_name, count in self.expected_objects.items():
                print(f"   - {obj_name}: {count}")
        else:
            print("\n2. No expected objects specified - will report all detections")
        
        print("\n✓ System initialized successfully")
        print("=" * 60)
        return True
    
    def on_box_complete(self):
        """
        Callback function triggered when all expected objects are detected in the box.
        Override this method or modify it to trigger custom actions (e.g., send signal, save data, etc.)
        """
        print("\n" + "=" * 60)
        print("✓✓✓ BOX COMPLETE - ALL OBJECTS DETECTED ✓✓✓")
        print("=" * 60)
        
        # You can add custom actions here:
        # - Trigger a signal to a PLC or robot
        # - Save detection results to a database
        # - Send a notification
        # - Log the event with timestamp
        # - Play a sound
        # - Update a dashboard
        
        # Example: Log with timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Completion detected at: {timestamp}")
        
    def on_box_incomplete(self, missing_objects, extra_objects=None):
        """
        Callback function triggered when box is incomplete or has incorrect objects.
        
        Args:
            missing_objects (dict): Objects that are missing
            extra_objects (dict): Objects that are extra (only in 'exact' mode)
        """
        # Optional: Add custom actions for incomplete box
        # This is called frequently, so avoid heavy operations
        pass
    
    def process_single_frame(self, visualize=True):
        """
        Process a single frame: capture, detect box, align, detect objects, verify.
        
        Args:
            visualize (bool): If True, return visualization images
            
        Returns:
            dict: Dictionary containing processing results and status
        """
        result = {
            'success': False,
            'box_detected': False,
            'objects_detected': [],
            'is_complete': False,
            'missing_objects': {},
            'extra_objects': {},
            'visualization': None
        }
        
        # Capture frame from camera
        raw_frame = self.camera.capture_frame()
        if raw_frame is None:
            return result
        
        # Detect and preprocess box
        if visualize:
            processed_frame, vis_frame = self.box_detector.process_frame(raw_frame, visualize=True)
        else:
            processed_frame = self.box_detector.process_frame(raw_frame, visualize=False)
            vis_frame = None
        
        if processed_frame is None:
            result['visualization'] = vis_frame
            return result
        
        result['box_detected'] = True
        
        # Detect objects in the processed frame
        detections = self.object_detector.detect_objects(processed_frame, verbose=False)
        result['objects_detected'] = detections
        
        # Verify if expected objects are present
        if self.expected_objects:
            is_complete, missing, extra = self.object_detector.check_expected_objects(
                detections, self.expected_objects, mode=self.verification_mode
            )
            result['is_complete'] = is_complete
            result['missing_objects'] = missing
            result['extra_objects'] = extra
        else:
            # If no expected objects specified, consider it "complete" if any objects detected
            result['is_complete'] = len(detections) > 0
        
        # Create visualization with detections
        if visualize and vis_frame is not None:
            # Also show detected objects on the processed frame
            processed_vis = self.object_detector.visualize_detections(processed_frame, detections)
            
            # Combine visualizations side by side
            h1, w1 = vis_frame.shape[:2]
            h2, w2 = processed_vis.shape[:2]
            
            # Resize processed frame to match height of original
            scale = h1 / h2
            new_w2 = int(w2 * scale)
            processed_vis_resized = cv2.resize(processed_vis, (new_w2, h1))
            
            combined = cv2.hconcat([vis_frame, processed_vis_resized])
            
            # Add status text
            status_text = "COMPLETE" if result['is_complete'] else "INCOMPLETE"
            status_color = (0, 255, 0) if result['is_complete'] else (0, 0, 255)
            cv2.putText(combined, f"Status: {status_text}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Add detected objects count
            obj_summary = self.object_detector.get_detection_summary(detections)
            y_offset = 110
            for obj_name, count in obj_summary.items():
                text = f"{obj_name}: {count}"
                cv2.putText(combined, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
            
            result['visualization'] = combined
        
        result['success'] = True
        return result
    
    def run(self, display_window=True):
        """
        Main loop: continuously capture and process frames.
        
        Args:
            display_window (bool): If True, display live video feed with detections
        """
        if not self.initialize():
            return
        
        self.is_running = True
        print("\nStarting continuous scanning...")
        print("Press 'q' to quit\n")
        
        try:
            while self.is_running:
                # Process frame
                result = self.process_single_frame(visualize=display_window)
                
                # Update state based on result
                if result['success'] and result['box_detected']:
                    if result['is_complete']:
                        self.consecutive_complete_detections += 1
                        
                        # Trigger completion callback if threshold reached
                        if (self.consecutive_complete_detections >= self.required_consecutive_detections
                            and not self.box_complete):
                            self.box_complete = True
                            self.on_box_complete()
                    else:
                        # Box incomplete - reset counter
                        if self.consecutive_complete_detections > 0:
                            self.consecutive_complete_detections = 0
                            self.box_complete = False
                        
                        self.on_box_incomplete(
                            result['missing_objects'],
                            result['extra_objects']
                        )
                else:
                    # No box detected - reset state
                    self.consecutive_complete_detections = 0
                    self.box_complete = False
                
                # Display visualization if enabled
                if display_window and result['visualization'] is not None:
                    cv2.imshow('Box Inspection System', result['visualization'])
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nShutting down...")
                        break
                
                # Rate limiting
                time.sleep(0.05)  # ~20 FPS
                
        except KeyboardInterrupt:
            print("\n\nShutdown requested by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Clean up resources.
        """
        print("\nCleaning up...")
        self.is_running = False
        self.camera.release()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")


def main():
    """
    Main entry point for the application.
    """
    # Configuration
    YOLO_MODEL = 'yolov8n.pt'  # Can use 'yolov8s.pt', 'yolov8m.pt', or custom trained model
    
    # Define expected objects in the box
    # Modify this dictionary based on your specific use case
    EXPECTED_OBJECTS = {
        'bottle': 2,    # Expect 2 bottles
        'cup': 1,       # Expect 1 cup
        # Add more objects as needed
    }
    
    # Create and run the inspection system
    system = BoxInspectionSystem(
        yolo_model_path=YOLO_MODEL,
        expected_objects=EXPECTED_OBJECTS
    )
    
    # You can modify system parameters before running
    system.verification_mode = 'minimum'  # 'exact', 'minimum', or 'any'
    system.required_consecutive_detections = 3
    system.check_interval = 0.5
    
    # Run the system
    system.run(display_window=True)


if __name__ == "__main__":
    main()
