"""
Object Detection Module 
Handles YOLO-based object detection and verification of expected objects in the box.
"""

from ultralytics import YOLO
import cv2
import numpy as np


class ObjectDetector:
    """
    Detects objects in images using YOLO and verifies if all expected objects are present.
    """
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the YOLO object detector.
        
        Args:
            model_path (str): Path to YOLO model file (e.g., 'yolov8n.pt', 'yolov8s.pt', or custom model)
            confidence_threshold (float): Minimum confidence score for detections (0.0 to 1.0)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []
        
        # Load the YOLO model
        self._load_model()
        
    def _load_model(self):
        """
        Load the YOLO model.
        """
        try:
            print(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            # Get class names from the model
            self.class_names = self.model.names
            print(f"Model loaded successfully. Available classes: {len(self.class_names)}")
        except Exception as e:
            print(f"ERROR: Failed to load YOLO model: {e}")
            raise
    
    def detect_objects(self, image, verbose=False):
        """
        Detect objects in the given image.
        
        Args:
            image (numpy.ndarray): Input BGR image
            verbose (bool): If True, print detailed detection information
            
        Returns:
            list: List of detections, each detection is a dict with keys:
                  'class_id', 'class_name', 'confidence', 'bbox' (x1, y1, x2, y2)
        """
        if self.model is None:
            print("ERROR: Model not loaded")
            return []
        
        # Run inference
        results = self.model(image, conf=self.confidence_threshold, verbose=verbose)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box information
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                confidence = float(box.conf[0].cpu().numpy())  # Confidence score
                class_id = int(box.cls[0].cpu().numpy())  # Class ID
                class_name = self.class_names[class_id]  # Class name
                
                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                }
                detections.append(detection)
                
                if verbose:
                    print(f"Detected: {class_name} (confidence: {confidence:.2f})")
        
        return detections
    
    def visualize_detections(self, image, detections, show_confidence=True):
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image (numpy.ndarray): Input image
            detections (list): List of detection dictionaries from detect_objects()
            show_confidence (bool): If True, show confidence scores on labels
            
        Returns:
            numpy.ndarray: Image with visualized detections
        """
        vis_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label text
            if show_confidence:
                label = f"{class_name} {confidence:.2f}"
            else:
                label = class_name
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_image, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return vis_image
    
    def check_expected_objects(self, detections, expected_objects, mode='exact'):
        """
        Check if all expected objects are present in the detections.
        
        Args:
            detections (list): List of detection dictionaries from detect_objects()
            expected_objects (dict): Dictionary mapping class names to expected counts
                                    Example: {'person': 1, 'bottle': 2, 'cup': 1}
            mode (str): Verification mode:
                       'exact' - exact count must match
                       'minimum' - at least the specified count must be present
                       'any' - at least one of each type must be present (ignores count)
            
        Returns:
            tuple: (is_complete, missing_objects, extra_objects)
                   is_complete (bool): True if requirements are met
                   missing_objects (dict): Objects that are missing or insufficient
                   extra_objects (dict): Objects that exceed expected count (only in 'exact' mode)
        """
        # Count detected objects by class name
        detected_counts = {}
        for det in detections:
            class_name = det['class_name']
            detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
        
        missing_objects = {}
        extra_objects = {}
        
        # Check each expected object
        for obj_name, expected_count in expected_objects.items():
            detected_count = detected_counts.get(obj_name, 0)
            
            if mode == 'exact':
                if detected_count < expected_count:
                    missing_objects[obj_name] = expected_count - detected_count
                elif detected_count > expected_count:
                    extra_objects[obj_name] = detected_count - expected_count
            elif mode == 'minimum':
                if detected_count < expected_count:
                    missing_objects[obj_name] = expected_count - detected_count
            elif mode == 'any':
                if detected_count == 0:
                    missing_objects[obj_name] = 1
        
        is_complete = len(missing_objects) == 0 and (mode != 'exact' or len(extra_objects) == 0)
        
        return is_complete, missing_objects, extra_objects
    
    def get_detection_summary(self, detections):
        """
        Get a summary of detected objects.
        
        Args:
            detections (list): List of detection dictionaries
            
        Returns:
            dict: Dictionary with object names as keys and counts as values
        """
        summary = {}
        for det in detections:
            class_name = det['class_name']
            summary[class_name] = summary.get(class_name, 0) + 1
        return summary
