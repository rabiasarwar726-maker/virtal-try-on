 
# **Virtual Try-On System (Image-Based)**

## **Description**
The Virtual Try-On System allows users to upload a person image and a transparent garment PNG to visualize how the garment looks on the body. The system uses pose detection to automatically position shirts or pants using a simple 2D image overlay approach.

## **Features**
- Upload person and garment images (front-facing)  
- Automatic garment positioning using pose landmarks  
- Supports transparent PNG shirts and pants  
- Fit status detection (Good / Tight / Loose)  
- Simple Streamlit-based interface  

## **Tech Stack **
- Python 3  
- Streamlit  
- OpenCV  
- MediaPipe  
- Pillow  
- NumPy  

## **How to Run **
```bash
git clone https://github.com/rabiasarwar726-maker/virtual-try-on.git
cd virtual-try-on
pip install -r requirements.txt
streamlit run app.py


Notes :

- Garments must be transparent PNGs
- Only front-facing images are supported
- This is a 2D virtual try-on system (no cloth physics)

Screenshot
 
<img width="1024" height="1536" alt="virtual_tryon_result (1)" src="https://github.com/user-attachments/assets/f53c2541-e4c6-4b96-b0f2-c86a4e36d90f" />
<img width="1024" height="1536" alt="virtual_tryon_result (2)" src="https://github.com/user-attachments/assets/8cfa988a-42ec-486e-b175-ac3b30384a54" />


## **Purpose** :

This project is developed for educational and academic purposes, focusing on pose detection and image-based virtual try-on using computer vision techniques.
Author: 
           Rabia Sarwar





