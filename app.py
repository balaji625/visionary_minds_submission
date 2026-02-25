import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
import os
import gdown
import matplotlib.pyplot as plt
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassAccuracy,
    MulticlassConfusionMatrix
)

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Offroad Terrain Intelligence",
    layout="wide"
)

# ---------------- CONFIG ---------------- #

NUM_CLASSES = 10
MODEL_PATH = "best_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1JdZ7XI80wOZ2llke0kqpgqeWNmUY8WJF"

class_names = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks",
    "Landscape", "Sky"
]

class_map = {
    100:0, 200:1, 300:2, 500:3, 550:4,
    600:5, 700:6, 800:7, 7100:8, 10000:9
}

# ---------------- DOWNLOAD MODEL ---------------- #

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name="resnet50",   # MUST match training
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES
    )

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device

# ---------------- INITIALIZE ---------------- #

download_model()
model, device = load_model()

# ---------------- UI ---------------- #

st.title("🏜️ Offroad Terrain Intelligence")
st.write("Upload an image to perform semantic segmentation with full evaluation metrics.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:

    # -------- Image Processing -------- #

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, (512,512))

    input_tensor = torch.from_numpy(image_resized).permute(2,0,1).float()/255.0
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # -------- Prediction -------- #

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze()

    pred_np = pred.cpu().numpy()

    # -------- Confidence & Entropy -------- #

    max_probs = torch.max(probs, dim=1)[0]
    confidence = torch.mean(max_probs).item()

    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    mean_entropy = torch.mean(entropy).item()

    # -------- Class Distribution -------- #

    unique, counts = np.unique(pred_np, return_counts=True)
    total_pixels = pred_np.size

    class_percentages = {}
    for cls, cnt in zip(unique, counts):
        percent = (cnt / total_pixels) * 100
        class_percentages[class_names[cls]] = percent

    # -------- Visualization -------- #

    np.random.seed(42)
    colors = np.random.randint(0,255,(NUM_CLASSES,3))
    colored_mask = colors[pred_np]
    overlay = (0.6 * image_resized + 0.4 * colored_mask).astype(np.uint8)

    col1, col2, col3 = st.columns(3)
    col1.image(image_resized, caption="Original Image")
    col2.image(colored_mask, caption="Segmentation Mask")
    col3.image(overlay, caption="Overlay")

    # -------- Distribution Chart -------- #

    st.subheader("📊 Terrain Distribution")

    fig_bar = plt.figure()
    plt.bar(class_percentages.keys(), class_percentages.values())
    plt.xticks(rotation=45)
    plt.ylabel("Area Percentage (%)")
    plt.tight_layout()
    st.pyplot(fig_bar)

    # -------- Confidence -------- #

    st.subheader("🔎 Model Confidence")
    st.write(f"Confidence Score: {confidence:.4f}")
    st.write(f"Entropy: {mean_entropy:.4f}")

    # -------- Optional Ground Truth -------- #

    st.markdown("---")
    st.subheader("Upload Ground Truth Mask (Optional for Metrics)")

    gt_file = st.file_uploader("Upload Ground Truth Mask", type=["png"])

    if gt_file:

        mask = Image.open(gt_file)
        mask = np.array(mask)

        if len(mask.shape)==3:
            mask = mask[:,:,0]

        mapped_mask = np.zeros_like(mask)
        for k,v in class_map.items():
            mapped_mask[mask==k] = v

        mapped_mask = cv2.resize(mapped_mask,(512,512),interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mapped_mask).long().to(device)

        # -------- Metrics -------- #

        iou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES).to(device)
        iou_per_class_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average=None).to(device)
        acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES).to(device)
        conf_metric = MulticlassConfusionMatrix(num_classes=NUM_CLASSES).to(device)

        iou_metric.update(pred, mask_tensor)
        iou_per_class_metric.update(pred, mask_tensor)
        acc_metric.update(pred, mask_tensor)
        conf_metric.update(pred, mask_tensor)

        mean_iou = iou_metric.compute().item()
        per_class_iou = iou_per_class_metric.compute().cpu().numpy()
        accuracy = acc_metric.compute().item()
        conf_matrix = conf_metric.compute().cpu().numpy()
        map50 = np.mean(per_class_iou >= 0.5)

        st.subheader("📈 Performance Metrics")
        st.write(f"Mean IoU: {mean_iou:.4f}")
        st.write(f"Pixel Accuracy: {accuracy:.4f}")
        st.write(f"mAP@50: {map50:.4f}")

        # -------- Per-Class IoU -------- #

        fig_iou = plt.figure()
        plt.bar(class_names, per_class_iou)
        plt.xticks(rotation=45)
        plt.ylabel("IoU")
        plt.tight_layout()
        st.pyplot(fig_iou)

        # -------- Confusion Matrix -------- #

        fig_cm = plt.figure()
        plt.imshow(conf_matrix)
        plt.title("Confusion Matrix")
        plt.colorbar()
        st.pyplot(fig_cm)
