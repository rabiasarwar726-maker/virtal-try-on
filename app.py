# =========================================
# Virtual Try-On System - Streamlit App
# Compatible with Mediapipe 0.10.30 and Python 3.13
# =========================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Virtual Try-On", page_icon="👗", layout="centered")
st.title("👗 Virtual Try-On System")

# -------------------------------
# Sidebar Upload
# -------------------------------
st.sidebar.header("Upload Images")
person_file = st.sidebar.file_uploader("Upload Person Image", type=["jpg", "jpeg", "png"])
garment_file = st.sidebar.file_uploader("Upload Garment PNG (transparent)", type=["png"])

# -------------------------------
# Helper Functions
# -------------------------------
def load_image(image_file):
    """Load image as OpenCV array (RGB)."""
    image = Image.open(image_file).convert("RGB")
    return np.array(image)

def overlay_transparent(background, overlay, x, y):
    """
    Overlay PNG with alpha channel on background image at position x, y.
    """
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]

    # Resize overlay if it goes beyond background
    if x + ol_w > bg_w or y + ol_h > bg_h:
        scale_w = min(ol_w, bg_w - x)
        scale_h = min(ol_h, bg_h - y)
        overlay = cv2.resize(overlay, (scale_w, scale_h))

    # Split channels
    b, g, r, a = cv2.split(overlay)
    overlay_rgb = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a))

    # Overlay
    bg_region = background[y:y+overlay.shape[0], x:x+overlay.shape[1]]
    bg_region = cv2.addWeighted(bg_region, 1, overlay_rgb, 1, 0, mask=mask)
    background[y:y+overlay.shape[0], x:x+overlay.shape[1]] = bg_region
    return background

# -------------------------------
# Main Logic
# -------------------------------
if person_file and garment_file:
    # Load images
    person_img = load_image(person_file)
    garment_img = load_image(garment_file)

    # Convert person image to BGR for Mediapipe
    rgb_image = cv2.cvtColor(person_img, cv2.COLOR_RGB2BGR)

    # Initialize Mediapipe Pose Detector
    with mp.solutions.pose.Pose(static_image_mode=True) as pose_detector:
        results = pose_detector.process(rgb_image)

        if results.pose_landmarks:
            # Example: Get shoulders coordinates to position garment
            h, w, _ = person_img.shape
            left_shoulder = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

            x1 = int(left_shoulder.x * w)
            y1 = int(left_shoulder.y * h)
            x2 = int(right_shoulder.x * w)
            y2 = int(right_shoulder.y * h)

            # Simple positioning: top-left of garment at left shoulder
            x = x1
            y = y1
            person_img = overlay_transparent(person_img, garment_img, x, y)

    # Display final image
    st.image(person_img, channels="RGB", caption="Virtual Try-On Result")
