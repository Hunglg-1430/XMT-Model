
import video_prediction_2
import streamlit as st
import cv2
import tempfile
import shutil

def app():
    st.write("Upload a Video to see if it contains a fake or real face.")
    file_uploaded = st.file_uploader("Choose the Video File", type=["mp4", "avi", "mov"])

    if file_uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            # Write the uploaded file to a temporary file
            shutil.copyfileobj(file_uploaded, tmp_file)
            tmp_file_path = tmp_file.name

            # Placeholder for displaying processed frames
            frame_placeholder = st.empty()

            # Process and display frames using your custom extract_frames function
            processed_frames = (video_prediction_2.process_frame(frame) for frame in video_prediction_2.extract_frames(tmp_file_path))
            for frame in processed_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, width=800)

                # Delay between frames (if necessary)
                cv2.waitKey(1000 // 30)  # Approximate delay for 30 fps
