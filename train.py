import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex

TRAIN_IMG_DIR = "hackathon/Offroad_Segmentation_Training_Dataset/train/Color_Images"
TRAIN_MASK_DIR = "hackathon/Offroad_Segmentation_Training_Dataset/train/Segmentation"
VAL_IMG_DIR = "hackathon/Offroad_Segmentation_Training_Dataset/val/Color_Images"
VAL_MASK_DIR = "hackathon/Offroad_Segmentation_Training_Dataset/val/Segmentation"

BATCH_SIZE = 4
LR = 3e-4
EPOCHS = 50
NUM_CLASSES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OffroadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.class_map = {
            100:0, 200:1, 300:2, 500:3, 550:4,
            600:5, 700:6, 800:7, 7100:8, 10000:9
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        mapped = np.zeros_like(mask)
        for k, v in self.class_map.items():
            mapped[mask == k] = v

        if self.transform:
            augmented = self.transform(image=image, mask=mapped)
            image = augmented["image"]
            mapped = augmented["mask"]

        image = image.float() / 255.0
        return image, mapped.long()

train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(512, 512),
    ToTensorV2()
])

train_dataset = OffroadDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_transform)
val_dataset = OffroadDataset(VAL_IMG_DIR, VAL_MASK_DIR, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES
).to(device)

class_weights = torch.tensor([0.5,1.0,1.0,1.0,1.2,2.5,2.5,1.3,0.3,0.3]).to(device)
ce_loss = nn.CrossEntropyLoss(weight=class_weights)
dice_loss = smp.losses.DiceLoss(mode="multiclass")

def loss_fn(pred, target):
    return ce_loss(pred, target) + dice_loss(pred, target)

optimizer = optim.AdamW(model.parameters(), lr=LR)
iou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES).to(device)

best_iou = 0
os.makedirs("runs", exist_ok=True)

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    iou_metric.reset()

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            iou_metric.update(preds, masks)

    val_loss /= len(val_loader)
    val_iou = iou_metric.compute().item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "runs/best_model.pth")

print("Training Complete")
print("Best IoU:", best_iou)