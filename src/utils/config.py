"""
Configuration Module
Handles application configuration and settings.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """Configuration for object detection."""
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    max_detections: int = 100
    nms_threshold: float = 0.45

@dataclass
class CameraConfig:
    """Configuration for camera settings."""
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps_target: int = 30

@dataclass
class UIConfig:
    """Configuration for UI settings."""
    page_title: str = "Real-time Object Detection with YOLO"
    sidebar_width: int = 300
    show_fps: bool = True
    show_stats: bool = True
    show_confidence: bool = True

@dataclass
class PerformanceConfig:
    """Configuration for performance settings."""
    detection_interval: int = 1  # Process every N frames
    max_fps: int = 30
    enable_gpu: bool = True
    batch_size: int = 1

class AppConfig:
    """Main application configuration class."""
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize application configuration.
        
        Args:
            config_file (str): Path to configuration file
        """
        self.config_file = config_file
        self.detection = DetectionConfig()
        self.camera = CameraConfig()
        self.ui = UIConfig()
        self.performance = PerformanceConfig()
        
        # Available YOLO models
        self.available_models = [
            "yolov8n.pt",  # Nano - fastest, least accurate
            "yolov8s.pt",  # Small
            "yolov8m.pt",  # Medium
            "yolov8l.pt",  # Large
            "yolov8x.pt",  # Extra Large - slowest, most accurate
        ]
        
        # Model descriptions
        self.model_descriptions = {
            "yolov8n.pt": "YOLOv8 Nano - Fastest, good for real-time applications",
            "yolov8s.pt": "YOLOv8 Small - Balanced speed and accuracy",
            "yolov8m.pt": "YOLOv8 Medium - Better accuracy, moderate speed",
            "yolov8l.pt": "YOLOv8 Large - High accuracy, slower inference",
            "yolov8x.pt": "YOLOv8 Extra Large - Highest accuracy, slowest inference"
        }
        
        self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update detection config
                if 'detection' in config_data:
                    detection_data = config_data['detection']
                    self.detection.model_name = detection_data.get('model_name', self.detection.model_name)
                    self.detection.confidence_threshold = detection_data.get('confidence_threshold', self.detection.confidence_threshold)
                    self.detection.max_detections = detection_data.get('max_detections', self.detection.max_detections)
                    self.detection.nms_threshold = detection_data.get('nms_threshold', self.detection.nms_threshold)
                
                # Update camera config
                if 'camera' in config_data:
                    camera_data = config_data['camera']
                    self.camera.camera_index = camera_data.get('camera_index', self.camera.camera_index)
                    self.camera.frame_width = camera_data.get('frame_width', self.camera.frame_width)
                    self.camera.frame_height = camera_data.get('frame_height', self.camera.frame_height)
                    self.camera.fps_target = camera_data.get('fps_target', self.camera.fps_target)
                
                # Update UI config
                if 'ui' in config_data:
                    ui_data = config_data['ui']
                    self.ui.page_title = ui_data.get('page_title', self.ui.page_title)
                    self.ui.sidebar_width = ui_data.get('sidebar_width', self.ui.sidebar_width)
                    self.ui.show_fps = ui_data.get('show_fps', self.ui.show_fps)
                    self.ui.show_stats = ui_data.get('show_stats', self.ui.show_stats)
                    self.ui.show_confidence = ui_data.get('show_confidence', self.ui.show_confidence)
                
                # Update performance config
                if 'performance' in config_data:
                    perf_data = config_data['performance']
                    self.performance.detection_interval = perf_data.get('detection_interval', self.performance.detection_interval)
                    self.performance.max_fps = perf_data.get('max_fps', self.performance.max_fps)
                    self.performance.enable_gpu = perf_data.get('enable_gpu', self.performance.enable_gpu)
                    self.performance.batch_size = perf_data.get('batch_size', self.performance.batch_size)
                
                logger.info(f"Configuration loaded from {self.config_file}")
                
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                logger.info("Using default configuration")
        else:
            logger.info("Configuration file not found, using defaults")
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            config_data = {
                'detection': {
                    'model_name': self.detection.model_name,
                    'confidence_threshold': self.detection.confidence_threshold,
                    'max_detections': self.detection.max_detections,
                    'nms_threshold': self.detection.nms_threshold
                },
                'camera': {
                    'camera_index': self.camera.camera_index,
                    'frame_width': self.camera.frame_width,
                    'frame_height': self.camera.frame_height,
                    'fps_target': self.camera.fps_target
                },
                'ui': {
                    'page_title': self.ui.page_title,
                    'sidebar_width': self.ui.sidebar_width,
                    'show_fps': self.ui.show_fps,
                    'show_stats': self.ui.show_stats,
                    'show_confidence': self.ui.show_confidence
                },
                'performance': {
                    'detection_interval': self.performance.detection_interval,
                    'max_fps': self.performance.max_fps,
                    'enable_gpu': self.performance.enable_gpu,
                    'batch_size': self.performance.batch_size
                }
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'name': model_name,
            'description': self.model_descriptions.get(model_name, "Unknown model"),
            'available': model_name in self.available_models
        }
    
    def validate_config(self) -> List[str]:
        """
        Validate current configuration.
        
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate detection config
        if not (0.0 <= self.detection.confidence_threshold <= 1.0):
            errors.append("Confidence threshold must be between 0.0 and 1.0")
        
        if self.detection.max_detections <= 0:
            errors.append("Max detections must be positive")
        
        if not (0.0 <= self.detection.nms_threshold <= 1.0):
            errors.append("NMS threshold must be between 0.0 and 1.0")
        
        # Validate camera config
        if self.camera.camera_index < 0:
            errors.append("Camera index must be non-negative")
        
        if self.camera.frame_width <= 0 or self.camera.frame_height <= 0:
            errors.append("Frame dimensions must be positive")
        
        if self.camera.fps_target <= 0:
            errors.append("FPS target must be positive")
        
        # Validate performance config
        if self.performance.detection_interval <= 0:
            errors.append("Detection interval must be positive")
        
        if self.performance.max_fps <= 0:
            errors.append("Max FPS must be positive")
        
        if self.performance.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        return errors
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.detection = DetectionConfig()
        self.camera = CameraConfig()
        self.ui = UIConfig()
        self.performance = PerformanceConfig()
        logger.info("Configuration reset to defaults")


# Global configuration instance
app_config = AppConfig()
