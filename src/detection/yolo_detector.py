"""
YOLO Object Detection Module
Handles loading YOLO models and performing object detection on images.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLODetector:
    """
    YOLO Object Detector class for real-time object detection.
    """
    
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the YOLO detector.
        
        Args:
            model_name (str): Name of the YOLO model to use
            confidence_threshold (float): Minimum confidence threshold for detections
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # COCO class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the YOLO model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on device: {self.device}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform object detection on an image.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            List[Dict[str, Any]]: List of detected objects with their properties
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image (np.ndarray): Input image
            detections (List[Dict[str, Any]]): List of detections
            
        Returns:
            np.ndarray: Image with drawn detections
        """
        annotated_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                annotated_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return annotated_image
    
    def get_detection_stats(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the detections.
        
        Args:
            detections (List[Dict[str, Any]]): List of detections
            
        Returns:
            Dict[str, Any]: Detection statistics
        """
        if not detections:
            return {
                'total_objects': 0,
                'class_counts': {},
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0
            }
        
        class_counts = {}
        confidences = []
        
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(confidence)
        
        return {
            'total_objects': len(detections),
            'class_counts': class_counts,
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences)
        }
    
    def update_confidence_threshold(self, threshold: float):
        """
        Update the confidence threshold.
        
        Args:
            threshold (float): New confidence threshold
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold updated to: {self.confidence_threshold}")
