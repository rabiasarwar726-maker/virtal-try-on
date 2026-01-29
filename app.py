import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Virtual Try-On System", layout="centered")
st.title("👕 Virtual Try-On System (Image-Based)")
st.write("Upload a person image and a transparent garment PNG to visualize try-on.")

# ---------------- MEDIAPIPE ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
mp_draw = mp.solutions.drawing_utils

# ---------------- FILE UPLOAD ----------------
person_file = st.file_uploader("Upload Person Image", type=["jpg", "jpeg", "png"])
garment_file = st.file_uploader("Upload Garment PNG (Transparent)", type=["png"])

garment_type = st.selectbox("Garment Type", ["Shirt", "Pant"])
show_pose = st.checkbox("Show Pose Landmarks", value=False)

# ---------------- FIT FUNCTION ----------------
def get_fit_status(body_width, garment_width):
    ratio = garment_width / body_width
    if ratio < 0.9:
        return "Tight ❌"
    elif ratio > 1.15:
        return "Loose ⚠️"
    else:
        return "Good Fit ✅"

# ---------------- OVERLAY FUNCTION ----------------
def overlay_image(base, overlay, x, y):
    if overlay.shape[2] == 3:
        alpha = np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
        overlay = np.concatenate([overlay, alpha], axis=2)

    h, w = overlay.shape[:2]
    for c in range(3):
        base[y:y+h, x:x+w, c] = (
            overlay[:, :, c] * (overlay[:, :, 3] / 255.0) +
            base[y:y+h, x:x+w, c] * (1.0 - overlay[:, :, 3] / 255.0)
        )
    return base

# ---------------- MAIN LOGIC ----------------
if person_file and garment_file:
    person_img = np.array(Image.open(person_file).convert("RGB"))
    h, w, _ = person_img.shape
    person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

    results = pose.process(person_rgb)
    if not results.pose_landmarks:
        st.error("Pose not detected. Upload a clear front-facing image.")
        st.stop()

    st.success("Pose detected successfully ✔")

    lm = results.pose_landmarks.landmark

    # Landmarks
    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
    rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]
    lk = lm[mp_pose.PoseLandmark.LEFT_KNEE]
    rk = lm[mp_pose.PoseLandmark.RIGHT_KNEE]

    x_ls, y_ls = int(ls.x * w), int(ls.y * h)
    x_rs, y_rs = int(rs.x * w), int(rs.y * h)
    x_lh, y_lh = int(lh.x * w), int(lh.y * h)
    x_rh, y_rh = int(rh.x * w), int(rh.y * h)
    x_lk, y_lk = int(lk.x * w), int(lk.y * h)
    x_rk, y_rk = int(rk.x * w), int(rk.y * h)

    garment_img = cv2.imdecode(
        np.frombuffer(garment_file.read(), np.uint8),
        cv2.IMREAD_UNCHANGED
    )

    output = person_img.copy()

    if garment_type == "Shirt":
        body_width = abs(x_rs - x_ls)
        garment_width = body_width
        garment_height = int(abs(y_lh - y_ls) * 1.1)

        garment_img = cv2.resize(garment_img, (garment_width, garment_height))
        center_x = (x_ls + x_rs) // 2
        top_left_x = center_x - garment_width // 2
        top_left_y = y_ls

        fit_status = get_fit_status(body_width, garment_width)
        output = overlay_image(output, garment_img, top_left_x, top_left_y)

    else:  # Pant
        body_width = abs(x_rh - x_lh)
        garment_width = body_width
        garment_height = int(abs(y_lk - y_lh) * 1.1)

        garment_img = cv2.resize(garment_img, (garment_width, garment_height))
        center_x = (x_lh + x_rh) // 2
        top_left_x = center_x - garment_width // 2
        top_left_y = y_lh

        fit_status = get_fit_status(body_width, garment_width)
        output = overlay_image(output, garment_img, top_left_x, top_left_y)

    if show_pose:
        mp_draw.draw_landmarks(output, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    st.subheader("Try-On Result")
    st.image(output, use_container_width=True)
    st.markdown(f"### 👔 Fit Status: **{fit_status}**")

    _, buffer = cv2.imencode(".png", output)
    st.download_button(
        "⬇ Download Result",
        buffer.tobytes(),
        "virtual_tryon_result.png",
        "image/png"
    )
