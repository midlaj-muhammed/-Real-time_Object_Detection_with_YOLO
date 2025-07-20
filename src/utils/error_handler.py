"""
Error Handler Module
Provides comprehensive error handling and recovery mechanisms.
"""

import logging
import traceback
import functools
from typing import Any, Callable, Optional
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionError(Exception):
    """Custom exception for detection-related errors."""
    pass

class CameraError(Exception):
    """Custom exception for camera-related errors."""
    pass

class ModelError(Exception):
    """Custom exception for model-related errors."""
    pass

def handle_exceptions(default_return=None, log_error=True):
    """
    Decorator for handling exceptions in functions.
    
    Args:
        default_return: Default value to return on exception
        log_error (bool): Whether to log the error
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    logger.debug(traceback.format_exc())
                return default_return
        return wrapper
    return decorator

class ErrorHandler:
    """
    Centralized error handling and recovery system.
    """
    
    def __init__(self):
        self.error_counts = {}
        self.max_retries = 3
        self.recovery_strategies = {
            'camera_error': self._recover_camera,
            'model_error': self._recover_model,
            'detection_error': self._recover_detection
        }
    
    def handle_error(self, error_type: str, error: Exception, context: dict = None) -> bool:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error_type (str): Type of error
            error (Exception): The exception that occurred
            context (dict): Additional context information
            
        Returns:
            bool: True if recovery was successful, False otherwise
        """
        # Log the error
        logger.error(f"{error_type}: {str(error)}")
        
        # Track error count
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Check if we've exceeded max retries
        if self.error_counts[error_type] > self.max_retries:
            logger.error(f"Max retries exceeded for {error_type}")
            return False
        
        # Try recovery strategy
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](error, context)
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {error_type}: {str(recovery_error)}")
                return False
        
        return False
    
    def _recover_camera(self, error: Exception, context: dict = None) -> bool:
        """
        Attempt to recover from camera errors.
        
        Args:
            error (Exception): The camera error
            context (dict): Context information
            
        Returns:
            bool: True if recovery successful
        """
        logger.info("Attempting camera recovery...")
        
        if context and 'webcam' in context:
            webcam = context['webcam']
            
            # Try to reinitialize camera
            try:
                webcam.stop_capture()
                return webcam.start_capture()
            except Exception as e:
                logger.error(f"Camera recovery failed: {str(e)}")
                return False
        
        return False
    
    def _recover_model(self, error: Exception, context: dict = None) -> bool:
        """
        Attempt to recover from model errors.
        
        Args:
            error (Exception): The model error
            context (dict): Context information
            
        Returns:
            bool: True if recovery successful
        """
        logger.info("Attempting model recovery...")
        
        if context and 'detector' in context:
            detector = context['detector']
            
            # Try to reload model
            try:
                return detector.load_model()
            except Exception as e:
                logger.error(f"Model recovery failed: {str(e)}")
                return False
        
        return False
    
    def _recover_detection(self, error: Exception, context: dict = None) -> bool:
        """
        Attempt to recover from detection errors.
        
        Args:
            error (Exception): The detection error
            context (dict): Context information
            
        Returns:
            bool: True if recovery successful
        """
        logger.info("Attempting detection recovery...")
        
        # For detection errors, we can try reducing confidence threshold
        if context and 'detector' in context:
            detector = context['detector']
            current_threshold = detector.confidence_threshold
            
            if current_threshold > 0.1:
                new_threshold = max(0.1, current_threshold - 0.1)
                detector.update_confidence_threshold(new_threshold)
                logger.info(f"Reduced confidence threshold to {new_threshold}")
                return True
        
        return False
    
    def reset_error_counts(self):
        """Reset all error counts."""
        self.error_counts.clear()
        logger.info("Error counts reset")
    
    def get_error_summary(self) -> dict:
        """
        Get summary of errors encountered.
        
        Returns:
            dict: Error summary
        """
        return {
            'error_counts': self.error_counts.copy(),
            'total_errors': sum(self.error_counts.values()),
            'error_types': list(self.error_counts.keys())
        }

class SafeDetector:
    """
    Wrapper for YOLO detector with error handling and fallbacks.
    """
    
    def __init__(self, detector, error_handler: ErrorHandler):
        self.detector = detector
        self.error_handler = error_handler
        self.fallback_frame = None
    
    @handle_exceptions(default_return=[])
    def detect_objects(self, image: np.ndarray) -> list:
        """
        Safe object detection with error handling.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            list: List of detections or empty list on error
        """
        try:
            return self.detector.detect_objects(image)
        except Exception as e:
            # Try to recover
            context = {'detector': self.detector}
            if self.error_handler.handle_error('detection_error', e, context):
                # Retry detection after recovery
                return self.detector.detect_objects(image)
            else:
                raise DetectionError(f"Detection failed: {str(e)}")
    
    @handle_exceptions(default_return=None)
    def draw_detections(self, image: np.ndarray, detections: list) -> Optional[np.ndarray]:
        """
        Safe detection drawing with error handling.
        
        Args:
            image (np.ndarray): Input image
            detections (list): List of detections
            
        Returns:
            Optional[np.ndarray]: Annotated image or None on error
        """
        try:
            return self.detector.draw_detections(image, detections)
        except Exception as e:
            logger.error(f"Error drawing detections: {str(e)}")
            return image  # Return original image as fallback

class SafeWebcam:
    """
    Wrapper for webcam capture with error handling and fallbacks.
    """
    
    def __init__(self, webcam, error_handler: ErrorHandler):
        self.webcam = webcam
        self.error_handler = error_handler
        self.last_good_frame = None
    
    @handle_exceptions(default_return=None)
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Safe frame capture with error handling.
        
        Returns:
            Optional[np.ndarray]: Frame or None on error
        """
        try:
            frame = self.webcam.get_frame()
            if frame is not None:
                self.last_good_frame = frame.copy()
                return frame
            else:
                # Try to recover camera
                context = {'webcam': self.webcam}
                if self.error_handler.handle_error('camera_error', CameraError("No frame received"), context):
                    return self.webcam.get_frame()
                else:
                    # Return last good frame as fallback
                    return self.last_good_frame
        except Exception as e:
            context = {'webcam': self.webcam}
            if self.error_handler.handle_error('camera_error', e, context):
                return self.webcam.get_frame()
            else:
                return self.last_good_frame
    
    def get_fps(self) -> float:
        """Get FPS with error handling."""
        try:
            return self.webcam.get_fps()
        except Exception:
            return 0.0
    
    def is_camera_available(self) -> bool:
        """Check camera availability with error handling."""
        try:
            return self.webcam.is_camera_available()
        except Exception:
            return False

def create_fallback_frame(width: int = 640, height: int = 480, message: str = "Camera Error") -> np.ndarray:
    """
    Create a fallback frame to display when camera fails.
    
    Args:
        width (int): Frame width
        height (int): Frame height
        message (str): Error message to display
        
    Returns:
        np.ndarray: Fallback frame
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add error message
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)  # Red
    thickness = 2
    
    # Get text size
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    
    # Center the text
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(frame, message, (text_x, text_y), font, font_scale, color, thickness)
    
    return frame

# Global error handler instance
global_error_handler = ErrorHandler()
