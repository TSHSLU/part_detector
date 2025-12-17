"""
Main Program for Object Detection in Box
Continuously scans for objects in a box placed under a camera and verifies completeness.
"""

import cv2
import time
from pathlib import Path
import platform
import os
import time
from package.camera_capture import CameraCapture
from package.object_detector import ObjectDetector



class BoxInspectionSystem:
    """
    Main system that orchestrates camera capture, box detection, and object verification.
    """
    
    def __init__(self, yolo_model_path='models/yolo11n.pt', expected_objects=None):
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
        self.object_detector = ObjectDetector(
            model_path=yolo_model_path,
            confidence_threshold=0.7,  # Increased threshold to reduce false positives
        )
        
        # Configuration
        self.expected_objects = expected_objects
        self.verification_mode = 'minimum'  # 'exact', 'minimum', or 'any' # new
        self.required_consecutive_detections = 3  # Number of consecutive complete detections to confirm box completeness

        
        # State tracking
        self.is_running = False
        self.last_check_time = 0
        self.consecutive_complete_detections = 0  # Counter for consecutive complete detections # new

        self.israspi=False

        # is true if 3 consecutive complete detections
        self.box_complete = False
        
        
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
        print("Camera initialized")
        
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


        #detect raspberry pi
        print("\n3. Detecting Raspberry Pi...")
        self.detect_rpi()
            


        print("\n✓ System initialized successfully")
        print("=" * 60)
        
        return True
        



    def detect_rpi(self):
        syst=platform.uname()
        print("Operating System:", syst)

        if "rpi" in syst.release.lower() or "raspberrypi" in syst.node.lower():
            self.israspi=True
            print("Raspberry Pi detected.")
            try:
                from sense_hat import SenseHat

                self.sense = SenseHat()
                self.sense.show_message("Start")
                def sense_red():
                    self.sense.clear((255, 0, 0))  # Red light to indicate startup or no objects
                def sense_yellow():
                    self.sense.clear((255, 255, 0))  # yellow light to indicate some objects detected
                def sense_green():
                    self.sense.clear((0, 255, 0))  # Green light to indicate complete box
                sense_red()

            except Exception as e:
                print(f"Error importing SenseHat: {e}")
                self.israspi=False

            return True
        
        print("Not a Raspberry Pi.")
        self.israspi=False
        return False
    

    def check_headless(self):
        """
        Check if running on Raspberry Pi in headless mode (no GUI).
        
        Returns:
            bool: True if running on Raspberry Pi without GUI
        """
        "does not work"
        if self.israspi:
            display = os.environ.get('DISPLAY')
            if display is None or display == '':
                print("Raspberry Pi running in headless mode (no GUI detected).")
                return True
            else:
                print("Raspberry Pi with GUI detected.")
                return False
        pass
    


    def on_box_complete(self):
        """
        Callback function triggered when all expected objects are detected in the box.
        modify it to trigger custom actions
        """
        print("\n" + "=" * 60)
        print("✓✓✓ BOX COMPLETE - ALL OBJECTS DETECTED ✓✓✓")
        print("=" * 60)
       


        # if raspberry pi detected use sense hat as indicator
        if self.israspi: # if raspberry pi detected use sense hat as indicator
            try:
                self.sense.clear((0, 255, 0))  # Green light
                time.sleep(3)  # Keep green light on for 3 seconds
                self.sense.show_message("Complete!", text_colour=(0, 255, 0), scroll_speed=0.3)
                self.sense.clear((0, 255, 0))  # Green light

            except Exception as e:
                print(f"Error controlling SenseHat: {e}")
        
            

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Completion detected at: {timestamp}")
        

    def on_box_incomplete(self):
        """
        Callback function triggered when box is incomplete or has incorrect objects.
        
        Args:
            missing_objects (dict): Objects that are missing
            extra_objects (dict): Objects that are extra (only in 'exact' mode)
        """
        
    


    def process_single_frame(self, visualize=True):  # new - completely rewritten
        """
        Process a single frame: capture, detect objects, verify completeness.
        
        Args:
            visualize (bool): If True, return visualization image
            
        Returns:
            dict: Dictionary containing processing results and status
        """
        # new - Initialize result dictionary with all required keys
        result = {
            'success': False,
            'objects_detected': [],
            'is_complete': False,
            'missing_objects': {},
            'extra_objects': {},
            'visualization': None
        }
        
        # new - Capture frame from camera
        frame = self.camera.capture_frame()
        if frame is None:
            return result
        
        # new - Detect objects in the frame
        detections = self.object_detector.detect_objects(frame, verbose=False)
        result['objects_detected'] = detections

        
        # new - Verify if expected objects are present
        if self.expected_objects:
            is_complete, missing, extra = self.object_detector.check_expected_objects(
                detections, self.expected_objects, mode=self.verification_mode
            )

            result['is_complete'] = is_complete
            result['missing_objects'] = missing
            result['extra_objects'] = extra
        else:
            # new - If no expected objects specified, consider it "complete" if any objects detected
            result['is_complete'] = len(detections) > 0
        
        # check fill state of box and control sense hat lights
        if self.israspi:
            try:
                if result['is_complete']:
                    self.sense.clear((0, 255, 0))  # Green light
                elif len(detections) > 0:
                    self.sense.clear((255, 255, 0))  # Yellow light
                else:
                    self.sense.clear((255, 0, 0))  # Red light

            except Exception as e:
                print(f"Error controlling SenseHat: {e}")
                
        # new - Create visualization if requested
        if visualize:
            vis_frame = self.object_detector.visualize_detections(frame, detections)
            
            # new - Add status text
            status_text = "COMPLETE" if result['is_complete'] else "INCOMPLETE"
            status_color = (0, 255, 0) if result['is_complete'] else (0, 0, 255)
            cv2.putText(vis_frame, f"Status: {status_text}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # new - Add detected objects count
            obj_summary = self.object_detector.get_detection_summary(detections)
            y_offset = 80
            for obj_name, count in obj_summary.items():
                text = f"{obj_name}: {count}"
                cv2.putText(vis_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
            
            result['visualization'] = vis_frame
        
        result['success'] = True  # new
        return result  # new
    
    def run(self, display_window=True):
        """
        Main loop: continuously capture and process frames.
        
        Args:
            display_window (bool): If True, display live video feed with detections
        """
        if not self.initialize():
            return
        # starts program
        self.is_running = True
        print("\nStarting continuous scanning...")
        print("Press 'q' to quit\n")
        
        # Create resizable window if display is enabled
        if display_window:
            cv2.namedWindow('Box Inspection System', cv2.WINDOW_NORMAL)
        
        try:
            while self.is_running:
                # Process frame
                result = self.process_single_frame(visualize=display_window)
                
                # Update state based on result
                if result['success']:
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
                        
                        self.on_box_incomplete()
                else:
                    # No items detected - reset state
                    self.consecutive_complete_detections = 0
                    self.box_complete = False
                    if self.israspi:
                        try:
                            self.sense.clear((255, 0, 0))  # Red light if no object detected

                        except Exception as e:
                            print(f"Error controlling SenseHat: {e}")
                
                # Display visualization if enabled
                if display_window and result['visualization'] is not None:
                    cv2.imshow('Box Inspection System', result['visualization'])
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nShutting down...")
                        break
                
                # Rate limiting
                time.sleep(0.01)  
                
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
    model_dir= Path(__file__).parent/'models'
    YOLO_MODEL = model_dir / 'modelv5_ncnn_model'  #insert  model here in path ./models/
    
    # Define expected objects in the box
    EXPECTED_OBJECTS = {
     'filters':1,
     'milkjug':1,
     'stamp':1,
     'tool':1,
     'tray':1,
     'watercontainer':1,
     'wood':1
    
    }
    
    # Create and run the inspection system
    system = BoxInspectionSystem(
        yolo_model_path=YOLO_MODEL,
        expected_objects=EXPECTED_OBJECTS
    )
    
    # system parameters
    system.required_consecutive_detections = 3
    

    # Determine if running in headless mode on Raspberry Pi
    display_window = not system.israspi
    display_window=True

    # Run the system
    system.run(display_window)


if __name__ == "__main__":
    main()
