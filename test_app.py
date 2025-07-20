#!/usr/bin/env python3
"""
Test script for the YOLO Object Detection application.
This script tests the core functionality without requiring a webcam.
"""

import sys
import os
import numpy as np
import cv2

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from detection.yolo_detector import YOLODetector
        print("✅ YOLODetector imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import YOLODetector: {e}")
        return False
    
    try:
        from detection.webcam_capture import WebcamCapture, get_available_cameras
        print("✅ WebcamCapture imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import WebcamCapture: {e}")
        return False
    
    try:
        from utils.config import app_config
        print("✅ Configuration imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import configuration: {e}")
        return False
    
    try:
        from utils.helpers import resize_image, format_detection_info
        print("✅ Helper functions imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import helpers: {e}")
        return False
    
    try:
        from utils.error_handler import ErrorHandler
        print("✅ Error handler imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import error handler: {e}")
        return False
    
    return True

def test_detector():
    """Test YOLO detector with a dummy image."""
    print("\nTesting YOLO detector...")
    
    try:
        from detection.yolo_detector import YOLODetector
        
        # Create a dummy image (640x480 RGB)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Initialize detector
        print("Initializing YOLOv8n detector...")
        detector = YOLODetector("yolov8n.pt", 0.5)
        
        if detector.model is None:
            print("❌ Failed to load YOLO model")
            return False
        
        print("✅ YOLO model loaded successfully")
        
        # Test detection on dummy image
        print("Running detection on dummy image...")
        detections = detector.detect_objects(dummy_image)
        print(f"✅ Detection completed. Found {len(detections)} objects")
        
        # Test drawing detections
        if detections:
            annotated_image = detector.draw_detections(dummy_image, detections)
            print("✅ Detection drawing completed")
        
        # Test statistics
        stats = detector.get_detection_stats(detections)
        print(f"✅ Statistics generated: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detector test failed: {e}")
        return False

def test_webcam_availability():
    """Test webcam availability."""
    print("\nTesting webcam availability...")
    
    try:
        from detection.webcam_capture import get_available_cameras, test_camera_availability
        
        available_cameras = get_available_cameras()
        print(f"Available cameras: {available_cameras}")
        
        if available_cameras:
            print("✅ At least one camera is available")
            
            # Test first available camera
            camera_index = available_cameras[0]
            if test_camera_availability(camera_index):
                print(f"✅ Camera {camera_index} is working")
            else:
                print(f"⚠️ Camera {camera_index} detected but not working properly")
        else:
            print("⚠️ No cameras detected (this is normal in some environments)")
        
        return True
        
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from utils.config import app_config
        
        # Test configuration loading
        print(f"✅ Configuration loaded")
        print(f"  - Default model: {app_config.detection.model_name}")
        print(f"  - Confidence threshold: {app_config.detection.confidence_threshold}")
        print(f"  - Available models: {len(app_config.available_models)}")
        
        # Test validation
        errors = app_config.validate_config()
        if not errors:
            print("✅ Configuration validation passed")
        else:
            print(f"⚠️ Configuration validation issues: {errors}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_helpers():
    """Test helper functions."""
    print("\nTesting helper functions...")
    
    try:
        from utils.helpers import resize_image, calculate_iou, create_color_palette
        
        # Test image resizing
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        resized = resize_image(dummy_image, 320, 240)
        print(f"✅ Image resize: {dummy_image.shape} -> {resized.shape}")
        
        # Test IoU calculation
        box1 = [10, 10, 50, 50]
        box2 = [30, 30, 70, 70]
        iou = calculate_iou(box1, box2)
        print(f"✅ IoU calculation: {iou:.3f}")
        
        # Test color palette
        colors = create_color_palette(10)
        print(f"✅ Color palette created: {len(colors)} colors")
        
        return True
        
    except Exception as e:
        print(f"❌ Helper functions test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 YOLO Object Detection - Application Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Helper Functions Test", test_helpers),
        ("Webcam Availability Test", test_webcam_availability),
        ("YOLO Detector Test", test_detector),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to use.")
        print("\nTo run the application:")
        print("  streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
