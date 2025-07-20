"""
Helper Utilities Module
Contains utility functions for the object detection application.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resize_image(image: np.ndarray, target_width: int, target_height: int, 
                maintain_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize an image to target dimensions.
    
    Args:
        image (np.ndarray): Input image
        target_width (int): Target width
        target_height (int): Target height
        maintain_aspect_ratio (bool): Whether to maintain aspect ratio
        
    Returns:
        np.ndarray: Resized image
    """
    if maintain_aspect_ratio:
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > target_width / target_height:
            # Width is the limiting factor
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Height is the limiting factor
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create a black canvas of target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate position to center the resized image
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        # Place the resized image on the canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        
        return canvas
    else:
        return cv2.resize(image, (target_width, target_height))

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (List[int]): First bounding box [x1, y1, x2, y2]
        box2 (List[int]): Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def filter_detections_by_area(detections: List[Dict[str, Any]], 
                             min_area: int = 100, max_area: int = None) -> List[Dict[str, Any]]:
    """
    Filter detections by bounding box area.
    
    Args:
        detections (List[Dict[str, Any]]): List of detections
        min_area (int): Minimum bounding box area
        max_area (int): Maximum bounding box area (None for no limit)
        
    Returns:
        List[Dict[str, Any]]: Filtered detections
    """
    filtered_detections = []
    
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        if area >= min_area and (max_area is None or area <= max_area):
            filtered_detections.append(detection)
    
    return filtered_detections

def get_detection_center(bbox: List[int]) -> Tuple[int, int]:
    """
    Get the center point of a bounding box.
    
    Args:
        bbox (List[int]): Bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple[int, int]: Center coordinates (x, y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y

def create_color_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Create a color palette for visualization.
    
    Args:
        num_colors (int): Number of colors to generate
        
    Returns:
        List[Tuple[int, int, int]]: List of BGR color tuples
    """
    colors = []
    for i in range(num_colors):
        hue = int(180 * i / num_colors)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    return colors

def format_detection_info(detection: Dict[str, Any]) -> str:
    """
    Format detection information for display.
    
    Args:
        detection (Dict[str, Any]): Detection dictionary
        
    Returns:
        str: Formatted detection information
    """
    class_name = detection['class_name']
    confidence = detection['confidence']
    bbox = detection['bbox']
    
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    return f"{class_name} ({confidence:.2f}) - Size: {width}x{height} (Area: {area})"

def calculate_detection_density(detections: List[Dict[str, Any]], 
                              image_width: int, image_height: int) -> float:
    """
    Calculate detection density (detections per unit area).
    
    Args:
        detections (List[Dict[str, Any]]): List of detections
        image_width (int): Image width
        image_height (int): Image height
        
    Returns:
        float: Detection density
    """
    if not detections:
        return 0.0
    
    image_area = image_width * image_height
    return len(detections) / image_area * 1000000  # Per million pixels

def benchmark_detection_speed(detector, image: np.ndarray, num_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark detection speed.
    
    Args:
        detector: Detection object with detect_objects method
        image (np.ndarray): Test image
        num_runs (int): Number of runs for benchmarking
        
    Returns:
        Dict[str, float]: Benchmark results
    """
    times = []
    
    # Warm-up run
    detector.detect_objects(image)
    
    # Benchmark runs
    for _ in range(num_runs):
        start_time = time.time()
        detector.detect_objects(image)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    
    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'avg_fps': 1.0 / avg_time if avg_time > 0 else 0.0,
        'max_fps': 1.0 / min_time if min_time > 0 else 0.0
    }

def create_detection_heatmap(detections: List[Dict[str, Any]], 
                           image_width: int, image_height: int,
                           grid_size: int = 50) -> np.ndarray:
    """
    Create a heatmap of detection locations.
    
    Args:
        detections (List[Dict[str, Any]]): List of detections
        image_width (int): Image width
        image_height (int): Image height
        grid_size (int): Size of grid cells
        
    Returns:
        np.ndarray: Heatmap image
    """
    # Create grid
    grid_width = image_width // grid_size
    grid_height = image_height // grid_size
    heatmap = np.zeros((grid_height, grid_width), dtype=np.float32)
    
    # Count detections in each grid cell
    for detection in detections:
        center_x, center_y = get_detection_center(detection['bbox'])
        grid_x = min(center_x // grid_size, grid_width - 1)
        grid_y = min(center_y // grid_size, grid_height - 1)
        heatmap[grid_y, grid_x] += 1
    
    # Normalize and convert to color
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Resize to original image size
    heatmap_resized = cv2.resize(heatmap, (image_width, image_height))
    
    # Convert to color heatmap
    heatmap_color = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    
    return heatmap_color

def log_detection_summary(detections: List[Dict[str, Any]], fps: float = None):
    """
    Log a summary of detections.
    
    Args:
        detections (List[Dict[str, Any]]): List of detections
        fps (float): Current FPS (optional)
    """
    if not detections:
        logger.info("No objects detected")
        return
    
    class_counts = {}
    total_confidence = 0
    
    for detection in detections:
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        total_confidence += confidence
    
    avg_confidence = total_confidence / len(detections)
    
    summary = f"Detected {len(detections)} objects (avg confidence: {avg_confidence:.2f})"
    if fps:
        summary += f" at {fps:.1f} FPS"
    
    summary += f" - Classes: {dict(class_counts)}"
    
    logger.info(summary)
