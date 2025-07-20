"""
Webcam Capture Module
Handles webcam initialization, frame capture, and video processing.
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebcamCapture:
    """
    Webcam capture class for real-time video processing.
    """
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Initialize the webcam capture.
        
        Args:
            camera_index (int): Camera index (usually 0 for default camera)
            width (int): Frame width
            height (int): Frame height
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.fps_counter = FPSCounter()
        
    def initialize_camera(self) -> bool:
        """
        Initialize the camera.
        
        Returns:
            bool: True if camera initialized successfully, False otherwise
        """
        try:
            logger.info(f"Initializing camera {self.camera_index}")
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test frame capture
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture test frame")
                self.cap.release()
                return False
            
            logger.info(f"Camera initialized successfully. Frame size: {frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            return False
    
    def start_capture(self) -> bool:
        """
        Start the video capture in a separate thread.
        
        Returns:
            bool: True if capture started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Capture is already running")
            return True
        
        if not self.initialize_camera():
            return False
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Video capture started")
        return True
    
    def stop_capture(self):
        """
        Stop the video capture.
        """
        if not self.is_running:
            return
        
        logger.info("Stopping video capture")
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Video capture stopped")
    
    def _capture_loop(self):
        """
        Main capture loop running in a separate thread.
        """
        while self.is_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                    self.fps_counter.update()
                else:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.01)  # Small delay to prevent busy waiting
                    
            except Exception as e:
                logger.error(f"Error in capture loop: {str(e)}")
                break
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame.
        
        Returns:
            Optional[np.ndarray]: Current frame or None if no frame available
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_fps(self) -> float:
        """
        Get the current FPS.
        
        Returns:
            float: Current FPS
        """
        return self.fps_counter.get_fps()
    
    def is_camera_available(self) -> bool:
        """
        Check if camera is available and working.
        
        Returns:
            bool: True if camera is available, False otherwise
        """
        return self.is_running and self.cap is not None and self.cap.isOpened()
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get the frame size.
        
        Returns:
            Tuple[int, int]: (width, height) of frames
        """
        return (self.width, self.height)
    
    def __del__(self):
        """
        Destructor to ensure proper cleanup.
        """
        self.stop_capture()


class FPSCounter:
    """
    FPS counter for measuring frame rate.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            window_size (int): Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self):
        """
        Update the FPS counter with a new frame.
        """
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        
        # Keep only the last window_size frame times
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def get_fps(self) -> float:
        """
        Get the current FPS.
        
        Returns:
            float: Current FPS
        """
        if len(self.frame_times) < 2:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0


def test_camera_availability(camera_index: int = 0) -> bool:
    """
    Test if a camera is available.
    
    Args:
        camera_index (int): Camera index to test
        
    Returns:
        bool: True if camera is available, False otherwise
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return ret
        return False
    except Exception:
        return False


def get_available_cameras(max_cameras: int = 5) -> list:
    """
    Get list of available camera indices.
    
    Args:
        max_cameras (int): Maximum number of cameras to check
        
    Returns:
        list: List of available camera indices
    """
    available_cameras = []
    for i in range(max_cameras):
        if test_camera_availability(i):
            available_cameras.append(i)
    return available_cameras
