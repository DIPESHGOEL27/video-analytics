# main.py
import cv2
import os
import yaml
import streamlit as st
import tempfile
import time

from app.utils.config_loader import load_config
from app.detection.yolo_detector import YoloDetector
from app.tracking.deep_sort.deep_sort_tracker import DeepSortTracker
from app.analytics.frame_processor import FrameProcessor

def process_video(input_path, output_path, detector, tracker, processor, fps):
    
    if isinstance(input_path, str):
        cap = cv2.VideoCapture(input_path)
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(input_path.read())
        cap = cv2.VideoCapture(tfile.name)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_counter = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = processor.process_frame(frame)
        out.write(processed_frame)

        frame_counter += 1
        elapsed = time.time() - start_time
        fps_now = frame_counter / elapsed
        remaining = (total_frames - frame_counter) / (fps_now + 1e-5)

        progress = frame_counter / total_frames
        progress_bar.progress(min(int(progress * 100), 100))
        status_text.text(f"Processing frame {frame_counter}/{total_frames} - "
                         f"ETA: {int(remaining)} sec")

    cap.release()
    out.release()
    status_text.text("✅ Video processing complete.")
    st.session_state['output_path'] = output_path
    st.session_state['video_processed'] = True
    

def main():
    st.title("🎥 AI-Powered Video Analytics Dashboard")
    st.markdown("Upload a video and watch AI-driven tracking and analytics in action!")

    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if 'video_processed' not in st.session_state:
        st.session_state['video_processed'] = False

    if uploaded_video is not None and not st.session_state['video_processed']:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
            tmp_input.write(uploaded_video.read())
            input_path = tmp_input.name

        output_path = os.path.join("output", "processed_video.mp4")

        config = load_config("config.yaml")
        config['video_input'] = input_path
        config['video_output'] = output_path
        config['fps'] = 30.0  # default fps

        detector = YoloDetector(config)
        tracker = DeepSortTracker(config)
        processor = FrameProcessor(detector, tracker, config)

        st.info("Processing video... Please wait.")
        process_video(input_path, output_path, detector, tracker, processor, config['fps'])

        if st.session_state['video_processed']:
         
            st.download_button(
                label="Download Processed Video",
                data=open(output_path, "rb"),
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

        else:
            st.error("Failed to load processed video.")

        

if __name__ == "__main__":
    main()