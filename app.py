import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Transformer for real-time processing
def create_transformer():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.face_mesh = face_mesh

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            h, w, _ = img.shape
            results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # Default states
            text_lines = []
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]
                coords = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                # Draw bounding box
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # Face center (normalized)
                x_center = (x_min + x_max) / 2 / w
                y_center = (y_min + y_max) / 2 / h
                # Centered if within 20% of frame center
                is_centered = (abs(x_center - 0.5) < 0.2) and (abs(y_center - 0.5) < 0.2)
                text_lines.append(f"Position: ({x_min},{y_min})")
                text_lines.append(f"Centered: {is_centered}")

                # Head pose estimation (yaw) using landmarks
                # Using landmarks 33 (left eye outer) and 263 (right eye outer)
                left = np.array(coords[33])
                right = np.array(coords[263])
                dx = right[0] - left[0]
                is_looking = abs(dx) < (x_max - x_min) * 0.3
                text_lines.append(f"Looking at camera: {is_looking}")
            else:
                text_lines.append("No face detected")

            # Overlay text
            y0, dy = 30, 25
            for i, line in enumerate(text_lines):
                y = y0 + i * dy
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            return img

    return VideoTransformer

# Streamlit UI
st.set_page_config(page_title="Webcam Analytics Dashboard", layout="wide")
st.title("📹 Live Webcam Analytics")
st.write("Detects face, gaze direction, and posture centering in real time.")

webrtc_streamer(
    key="webcam-analytics",
    mode="VIDEO",
    video_transformer_factory=create_transformer,
    media_stream_constraints={"video": True, "audio": False},
)
