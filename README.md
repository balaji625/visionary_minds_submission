# 🏜️ Offroad Terrain Intelligence

### Duality AI Hackathon Submission

**Semantic Segmentation for Desert Navigation**


🔗 Deployed Application:
👉 https://visionarymindssubmission-6lubwyt7kvbwlbwitzfonh.streamlit.app

---
## Team Name : Visionary Minds

## 👤 Participants

**KONDREDDY VIJAYA **
**BATHALA BALAJI **
**BALA NEERAJA **

---

## 🎯 Final Validation Performance

* **Mean IoU (mIoU):** *0.6982+*
* **Pixel Accuracy:** *0.79+*
* **Epochs:** 50
* **Input Resolution:** 512 × 512
* **Classes:** 10 Terrain Categories

IoU is computed as:

IoU = TP / (TP + FP + FN)

Mean IoU:0.6982+

mIoU = (1/N) \sum IoU

---

# 🌵 Project Overview

Desert terrain segmentation presents unique challenges:

* Extreme lighting variability
* Texture similarity (sand vs dry grass)
* Severe class imbalance
* Sparse object boundaries

This project delivers a **robust semantic segmentation pipeline** trained for high accuracy and stability under these conditions.

The model segments:

* 🌳 Trees
* 🌿 Lush Bushes
* 🌾 Dry Grass
* 🍂 Dry Bushes
* 🧱 Ground Clutter
* 🌸 Flowers
* 🪵 Logs
* 💎 Rocks
* 🏜️ Landscape (Sand/Dunes)
* ☁️ Sky

---

# 🧠 Model Architecture

| Component            | Specification                     |
| -------------------- | --------------------------------- |
| Architecture         | UNet                              |
| Backbone             | ResNet50 (ImageNet pretrained)    |
| Framework            | PyTorch                           |
| Segmentation Library | segmentation_models_pytorch       |
| Augmentation         | Albumentations                    |
| Loss                 | Weighted CrossEntropy + Dice Loss |
| Optimizer            | AdamW                             |
| Scheduler            | Stable constant LR                |

---

# 🏗️ Architecture Flow

```
Input Image (512x512)
        ↓
ResNet50 Encoder (Pretrained)
        ↓
UNet Decoder
        ↓
Pixel-wise 10-Class Mask
```

---

# ⚙️ Training Strategy

* 50 Epochs
* Mixed loss: CE + Dice
* Class-weight balancing
* Validation IoU tracking
* Best model saved automatically

The best model is selected based on validation mIoU improvement.

---

# 📂 Project Structure

```
submission/
│
├── train.py
├── test.py
├── requirements.txt
├── report.pdf
│
└── runs/
    └── best_model.pth
```

---

# 🚀 How To Run

## 1️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Train Model

```bash
python train.py
```

Best model will be saved at:

```
runs/best_model.pth
```

---

## 3️⃣ Run Inference

```bash
python test.py --model_path runs/best_model.pth --image_dir path_to_test_images
```

Predicted masks will be saved in:

```
predictions/
```

---

# 📊 Performance Characteristics

* Stable convergence over 50 epochs
* Improved rare-class detection using weighted loss
* Balanced precision across terrain types
* GPU accelerated training

---

# 🔬 Evaluation Methodology

Validation metrics computed using:

* Mean IoU
* Per-class IoU
* Pixel Accuracy
* Confusion Matrix

All metrics are calculated using ground-truth masks from the validation set.

No artificial benchmarking values are used.

---

# 💡 Key Strengths

✔ Proper label remapping
✔ Robust class imbalance handling
✔ Clean training pipeline
✔ Reproducible results
✔ Best-model checkpointing
✔ Submission-ready structure

---

# 🔮 Future Improvements

* ONNX export for deployment
* TensorRT acceleration
* Real-time Streamlit demo
* Edge device optimization
* Multi-scale inference

---

# 📜 License

This project is created for the **Duality AI Hackathon** submission.

---

#  Final Note

Designed for robustness.
Engineered for clarity.
Built for real-world desert intelligence.





