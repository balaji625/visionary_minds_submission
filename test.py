import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassAccuracy,
    MulticlassConfusionMatrix
)
from google.colab import files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

NUM_CLASSES = 10

class_names = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks",
    "Landscape", "Sky"
]

class_map = {
    100:0, 200:1, 300:2, 500:3, 550:4,
    600:5, 700:6, 800:7, 7100:8, 10000:9
}

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES
)

model.load_state_dict(torch.load("runs/best_model.pth", map_location=device))
model = model.to(device)
model.eval()

print("Model loaded successfully.")

print("\nUpload IMAGE:")
uploaded_img = files.upload()
img_name = list(uploaded_img.keys())[0]

image = cv2.imread(img_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (512,512))

input_tensor = torch.from_numpy(image).permute(2,0,1).float()/255.0
input_tensor = input_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).squeeze()

pred_np = pred.cpu().numpy()

print("Prediction complete.")

max_probs = torch.max(probs, dim=1)[0]
confidence = torch.mean(max_probs).item()

entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
mean_entropy = torch.mean(entropy).item()

unique, counts = np.unique(pred_np, return_counts=True)
total_pixels = pred_np.size

print("\n===== CLASS DISTRIBUTION =====")
class_percentages = {}
for cls, cnt in zip(unique, counts):
    percent = (cnt / total_pixels) * 100
    class_percentages[class_names[cls]] = percent
    print(f"{class_names[cls]}: {percent:.2f}%")

colors = np.random.randint(0,255,(NUM_CLASSES,3))
colored_mask = colors[pred_np]
overlay = (0.6 * image + 0.4 * colored_mask).astype(np.uint8)

plt.figure(figsize=(18,6))
plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(colored_mask)
plt.title("Prediction")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(overlay)
plt.title("Overlay")
plt.axis("off")
plt.show()

plt.figure(figsize=(10,5))
plt.bar(class_percentages.keys(), class_percentages.values())
plt.xticks(rotation=45)
plt.ylabel("Area Percentage (%)")
plt.title("Predicted Class Distribution")
plt.show()

print("\nIf ground truth mask exists, upload it now (optional).")
uploaded_mask = files.upload()

if len(uploaded_mask) > 0:

    mask_name = list(uploaded_mask.keys())[0]
    mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)

    if len(mask.shape) == 3:
        mask = mask[:,:,0]

    mapped_mask = np.zeros_like(mask)
    for k, v in class_map.items():
        mapped_mask[mask == k] = v

    mapped_mask = cv2.resize(mapped_mask, (512,512), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.from_numpy(mapped_mask).long().to(device)

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

    print("\n===== REAL PERFORMANCE METRICS =====")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Pixel Accuracy: {accuracy:.4f}")
    print(f"mAP@50: {map50:.4f}")

    print("\nPer-Class IoU:")
    for i, score in enumerate(per_class_iou):
        print(f"{class_names[i]}: {score:.4f}")

    plt.figure(figsize=(6,6))
    plt.imshow(conf_matrix)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(10,5))
    plt.bar(class_names, per_class_iou)
    plt.xticks(rotation=45)
    plt.ylabel("IoU")
    plt.title("Per-Class IoU")
    plt.show()

else:
    print("\nNo ground truth provided → Inference mode only.")

print("\nConfidence:", round(confidence,4))
print("Entropy:", round(mean_entropy,4))
print("\nTesting completed successfully.")