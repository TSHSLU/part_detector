"""
Box Detection and Alignment Module
Detects the box in the image and aligns/crops it for optimal object detection.
"""

import cv2
import numpy as np


class BoxDetector:
    """
    Detects and aligns boxes in images for consistent object detection.
    Handles box detection, rotation correction, and image cropping.
    """
    
    def __init__(self, box_color_lower=None, box_color_upper=None, min_box_area=10000):
        """
        Initialize the box detector.
        
        Args:
            box_color_lower (tuple): Lower HSV color threshold for box detection (default: None)
            box_color_upper (tuple): Upper HSV color threshold for box detection (default: None)
            min_box_area (int): Minimum area in pixels for valid box detection
        """
        # Default color range for box detection (can be adjusted based on box color)
        # These defaults work well for darker boxes
        self.box_color_lower = box_color_lower if box_color_lower else np.array([0, 0, 0])
        self.box_color_upper = box_color_upper if box_color_upper else np.array([180, 255, 80])
        self.min_box_area = min_box_area
        
    def detect_box(self, image):
        """
        Detect the box in the image using color-based segmentation and contour detection.
        
        Args:
            image (numpy.ndarray): Input BGR image
            
        Returns:
            tuple: (box_contour, box_rect) where box_contour is the detected contour
                   and box_rect is the rotated rectangle (cv2.RotatedRect), or (None, None) if not found
        """
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apply color thresholding to isolate the box
        mask = cv2.inRange(hsv, self.box_color_lower, self.box_color_upper)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Find the largest contour (assuming it's the box)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if the contour is large enough
        if cv2.contourArea(largest_contour) < self.min_box_area:
            return None, None
        
        # Get the minimum area rectangle (handles rotation)
        box_rect = cv2.minAreaRect(largest_contour)
        
        return largest_contour, box_rect
    
    def get_rotation_angle(self, box_rect):
        """
        Calculate the rotation angle needed to align the box.
        
        Args:
            box_rect (tuple): Rotated rectangle from cv2.minAreaRect
            
        Returns:
            float: Rotation angle in degrees
        """
        # Extract angle from the rotated rectangle
        # cv2.minAreaRect returns (center, size, angle)
        _, (width, height), angle = box_rect
        
        # Adjust angle based on box orientation
        # If width < height, add 90 degrees to correct orientation
        if width < height:
            angle = angle + 90
        
        return angle
    
    def rotate_image(self, image, angle, center=None):
        """
        Rotate image around a center point.
        
        Args:
            image (numpy.ndarray): Input image
            angle (float): Rotation angle in degrees (positive = counter-clockwise)
            center (tuple): Rotation center (x, y), defaults to image center
            
        Returns:
            numpy.ndarray: Rotated image
        """
        if center is None:
            center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions to prevent cropping
        cos_val = np.abs(rotation_matrix[0, 0])
        sin_val = np.abs(rotation_matrix[0, 1])
        new_width = int((image.shape[0] * sin_val) + (image.shape[1] * cos_val))
        new_height = int((image.shape[0] * cos_val) + (image.shape[1] * sin_val))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
        
        return rotated
    
    def crop_box_region(self, image, box_rect, padding=20):
        """
        Crop the box region from the image with optional padding.
        
        Args:
            image (numpy.ndarray): Input image (should be aligned/rotated first)
            box_rect (tuple): Rotated rectangle defining the box
            padding (int): Additional padding around the box in pixels
            
        Returns:
            numpy.ndarray: Cropped box region, or None if crop failed
        """
        # Get box vertices
        box_points = cv2.boxPoints(box_rect)
        box_points = np.int32(box_points)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(box_points)
        
        # Apply padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop the region
        cropped = image[y:y+h, x:x+w]
        
        return cropped if cropped.size > 0 else None
    
    def preprocess_for_yolo(self, image, target_size=(640, 640)):
        """
        Preprocess the cropped box image for YOLO inference.
        
        Args:
            image (numpy.ndarray): Input image (cropped box region)
            target_size (tuple): Target size for YOLO (width, height)
            
        Returns:
            numpy.ndarray: Preprocessed image ready for YOLO inference
        """
        # Resize image while maintaining aspect ratio
        h, w = image.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create a canvas with target size and paste resized image
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Optional: Apply additional preprocessing
        # - Brightness/contrast adjustment
        # - Denoising
        # - Sharpening
        # canvas = cv2.detailEnhance(canvas, sigma_s=10, sigma_r=0.15)
        
        return canvas
    
    def process_frame(self, image, visualize=False):
        """
        Complete pipeline: detect box, align, crop, and preprocess.
        
        Args:
            image (numpy.ndarray): Input BGR image from camera
            visualize (bool): If True, return visualization image alongside processed image
            
        Returns:
            tuple: (processed_image, visualization_image) if visualize=True,
                   otherwise just processed_image. Returns None if box not detected.
        """
        # Detect the box
        contour, box_rect = self.detect_box(image)
        
        if box_rect is None:
            return (None, image.copy()) if visualize else None
        
        # Get rotation angle and align the image
        angle = self.get_rotation_angle(box_rect)
        center = box_rect[0]  # Center of the rotated rectangle
        aligned = self.rotate_image(image, angle, center)
        
        # Detect box in aligned image (to get new coordinates)
        _, aligned_box_rect = self.detect_box(aligned)
        
        if aligned_box_rect is None:
            return (None, image.copy()) if visualize else None
        
        # Crop the box region
        cropped = self.crop_box_region(aligned, aligned_box_rect, padding=20)
        
        if cropped is None:
            return (None, image.copy()) if visualize else None
        
        # Preprocess for YOLO
        processed = self.preprocess_for_yolo(cropped)
        
        # Create visualization if requested
        if visualize:
            vis_image = image.copy()
            # Draw the detected box
            box_points = cv2.boxPoints(box_rect)
            box_points = np.int32(box_points)
            cv2.drawContours(vis_image, [box_points], 0, (0, 255, 0), 3)
            # Add text
            cv2.putText(vis_image, f"Box detected (angle: {angle:.1f}°)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return processed, vis_image
        
        return processed
