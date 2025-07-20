"""
Real-time Object Detection with YOLO - Streamlit Application
Main application file for the web interface.
"""

import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detection.yolo_detector import YOLODetector
from detection.webcam_capture import WebcamCapture, get_available_cameras
from utils.config import app_config
from utils.helpers import format_detection_info, log_detection_summary

# Page configuration
st.set_page_config(
    page_title=app_config.ui.page_title,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'webcam' not in st.session_state:
    st.session_state.webcam = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'detection_stats' not in st.session_state:
    st.session_state.detection_stats = {}

def initialize_detector(model_name: str, confidence_threshold: float):
    """Initialize the YOLO detector."""
    try:
        with st.spinner(f"Loading {model_name} model..."):
            detector = YOLODetector(model_name, confidence_threshold)
            if detector.model is not None:
                st.session_state.detector = detector
                st.success(f"✅ Model {model_name} loaded successfully!")
                return True
            else:
                st.error(f"❌ Failed to load model {model_name}")
                return False
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return False

def initialize_webcam(camera_index: int):
    """Initialize the webcam."""
    try:
        webcam = WebcamCapture(
            camera_index=camera_index,
            width=app_config.camera.frame_width,
            height=app_config.camera.frame_height
        )
        if webcam.start_capture():
            st.session_state.webcam = webcam
            st.success(f"✅ Camera {camera_index} initialized successfully!")
            return True
        else:
            st.error(f"❌ Failed to initialize camera {camera_index}")
            return False
    except Exception as e:
        st.error(f"❌ Error initializing camera: {str(e)}")
        return False

def main():
    """Main application function."""
    
    # Title and description
    st.title("🎯 Real-time Object Detection with YOLO")
    st.markdown("Detect objects in real-time using your webcam and YOLO models.")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("🤖 Model Configuration")
        selected_model = st.selectbox(
            "Select YOLO Model",
            app_config.available_models,
            index=app_config.available_models.index(app_config.detection.model_name),
            help="Choose the YOLO model variant. Nano is fastest, X is most accurate."
        )
        
        # Show model description
        model_info = app_config.get_model_info(selected_model)
        st.info(model_info['description'])
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=app_config.detection.confidence_threshold,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        # Camera selection
        st.subheader("📹 Camera Configuration")
        available_cameras = get_available_cameras()

        if available_cameras:
            camera_index = st.selectbox(
                "Select Camera",
                available_cameras,
                index=0,
                help="Choose the camera to use for detection"
            )
        else:
            st.error("❌ No cameras detected!")
            st.error("Please connect a camera to use this application.")
            camera_index = 0
        
        # Performance settings
        st.subheader("⚡ Performance")
        detection_interval = st.slider(
            "Detection Interval",
            min_value=1,
            max_value=10,
            value=app_config.performance.detection_interval,
            help="Process every N frames (higher = faster but less frequent detection)"
        )
        
        # Display settings
        st.subheader("🎨 Display Options")
        show_fps = st.checkbox("Show FPS", value=app_config.ui.show_fps)
        show_stats = st.checkbox("Show Statistics", value=app_config.ui.show_stats)
        show_confidence = st.checkbox("Show Confidence Scores", value=app_config.ui.show_confidence)
        
        # Control buttons
        st.subheader("🎮 Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Detection", disabled=st.session_state.is_running or not available_cameras):
                # Initialize detector if needed
                if (st.session_state.detector is None or
                    st.session_state.detector.model_name != selected_model or
                    st.session_state.detector.confidence_threshold != confidence_threshold):

                    if initialize_detector(selected_model, confidence_threshold):
                        st.session_state.detector.update_confidence_threshold(confidence_threshold)
                    else:
                        st.stop()

                # Initialize webcam if needed
                if (st.session_state.webcam is None or
                    not st.session_state.webcam.is_camera_available()):

                    if not initialize_webcam(camera_index):
                        st.stop()

                st.session_state.is_running = True
                st.session_state.frame_count = 0
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Detection", disabled=not st.session_state.is_running):
                st.session_state.is_running = False
                if st.session_state.webcam:
                    st.session_state.webcam.stop_capture()
                    st.session_state.webcam = None
                st.rerun()
    
    # Main content area
    if st.session_state.is_running and st.session_state.webcam and st.session_state.detector:
        # Create columns for video and stats
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📹 Live Video Feed")
            video_placeholder = st.empty()
        
        with col2:
            if show_stats:
                st.subheader("📊 Statistics")
                stats_placeholder = st.empty()
        
        # Performance metrics placeholders
        if show_fps:
            fps_placeholder = st.empty()
        
        # Detection loop
        while st.session_state.is_running:
            frame = st.session_state.webcam.get_frame()

            if frame is not None:
                st.session_state.frame_count += 1
                
                # Process frame for detection
                if st.session_state.frame_count % detection_interval == 0:
                    detections = st.session_state.detector.detect_objects(frame)
                    annotated_frame = st.session_state.detector.draw_detections(frame, detections)
                    
                    # Update detection stats
                    st.session_state.detection_stats = st.session_state.detector.get_detection_stats(detections)
                else:
                    annotated_frame = frame
                    detections = []
                
                # Convert BGR to RGB for display
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display video
                with video_placeholder.container():
                    st.image(display_frame, channels="RGB", use_container_width=True)
                
                # Display FPS
                if show_fps:
                    current_fps = st.session_state.webcam.get_fps()
                    fps_placeholder.metric("🎯 FPS", f"{current_fps:.1f}")
                
                # Display statistics
                if show_stats and st.session_state.detection_stats:
                    with stats_placeholder.container():
                        stats = st.session_state.detection_stats
                        
                        st.metric("🎯 Objects Detected", stats['total_objects'])
                        
                        if stats['total_objects'] > 0:
                            st.metric("📈 Avg Confidence", f"{stats['avg_confidence']:.2f}")
                            st.metric("🔝 Max Confidence", f"{stats['max_confidence']:.2f}")
                            
                            # Class breakdown
                            st.write("**Object Classes:**")
                            for class_name, count in stats['class_counts'].items():
                                st.write(f"• {class_name}: {count}")
                
                # Small delay to prevent overwhelming the interface
                time.sleep(0.01)
            
            else:
                st.warning("⚠️ No frame received from camera")
                time.sleep(0.1)
    
    else:
        # Show instructions when not running
        if available_cameras:
            st.info("👆 Configure your settings in the sidebar and click 'Start Detection' to begin!")
        else:
            st.error("❌ No cameras detected! Please connect a camera to use this application.")

        # Show system information
        st.subheader("ℹ️ System Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Available Cameras:**")
            cameras = get_available_cameras()
            if cameras:
                for cam in cameras:
                    st.write(f"• Camera {cam}")
            else:
                st.write("• No cameras detected")

        with col2:
            st.write("**Available Models:**")
            for model in app_config.available_models:
                st.write(f"• {model}")

        with col3:
            st.write("**Current Configuration:**")
            st.write(f"• Model: {app_config.detection.model_name}")
            st.write(f"• Confidence: {app_config.detection.confidence_threshold}")
            st.write(f"• Camera: {app_config.camera.camera_index}")

if __name__ == "__main__":
    main()
