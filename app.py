import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
import os
import requests

st.set_page_config(page_title="Offroad Terrain Intelligence", layout="wide")

NUM_CLASSES = 10
MODEL_PATH = "best_model.pth"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1JdZ7XI80wOZ2llke0kqpgqeWNmUY8WJF"

class_names = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks",
    "Landscape", "Sky"
]

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading trained model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("Model downloaded successfully.")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, device

download_model()
model, device = load_model()

st.title("🏜️ Offroad Terrain Intelligence")
st.write("Upload an image to perform semantic segmentation.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, (512, 512))

    input_tensor = torch.from_numpy(image_resized).permute(2,0,1).float()/255.0
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze()

    pred_np = pred.cpu().numpy()

    unique, counts = np.unique(pred_np, return_counts=True)
    total_pixels = pred_np.size

    class_percentages = {}
    for cls, cnt in zip(unique, counts):
        percent = (cnt / total_pixels) * 100
        class_percentages[class_names[cls]] = percent

    colors = np.random.randint(0,255,(NUM_CLASSES,3))
    colored_mask = colors[pred_np]
    overlay = (0.6 * image_resized + 0.4 * colored_mask).astype(np.uint8)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image_resized, caption="Original Image")

    with col2:
        st.image(colored_mask, caption="Segmentation Mask")

    with col3:
        st.image(overlay, caption="Overlay")

    st.subheader("📊 Terrain Distribution")

    for name, percent in class_percentages.items():
        st.write(f"{name}: {percent:.2f}%")

    confidence = torch.mean(torch.max(probs, dim=1)[0]).item()

    st.subheader("🔎 Model Confidence")
    st.write(f"{confidence:.4f}")
