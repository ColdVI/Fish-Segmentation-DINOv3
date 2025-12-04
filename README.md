

```md
# 🎣 Fish-Segmentation-DINOv3

This repository contains a complete semantic segmentation pipeline using **Meta AI’s DINOv3 Vision Transformer** on the **“A Large-Scale Fish Dataset”** from Kaggle.  
The project demonstrates how a frozen self-supervised backbone combined with a lightweight decoder can achieve **95%+ IoU** on a real-world dataset containing diverse fish species and challenging imaging conditions.

---

# 🐟 Dataset

This project uses the **A Large-Scale Fish Dataset** from Kaggle:  
https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset

### Dataset Properties
- ~9,000 labeled fish images  
- Pixel-accurate segmentation masks (`ClassName GT`)  
- Large variations in:
  - species  
  - fish shape/thickness  
  - rotation orientation  
  - lighting and glare  
  - background plates  
  - color variations  

These characteristics make the dataset ideal for evaluating general-purpose segmentation models.

---

# 🤖 Why DINOv3 Works Extremely Well

Although DINOv3 is **self-supervised** and not explicitly trained for segmentation, it performs exceptionally due to:

### 🔹 1. Strong global feature representation  
Vision Transformers capture long-range dependencies → ideal for elongated fish bodies.

### 🔹 2. High semantic separation  
The model’s learned representations naturally separate foreground (fish) from background.

### 🔹 3. Patch-level structure preservation  
ViT patch embeddings preserve body contours, improving mask sharpness.

### 🔹 4. Frozen backbone → stable training  
Only a small CNN decoder is trained.  
No risk of overfitting.  
Fast, consistent convergence.

---

# ⚙️ Model Architecture

```

Input Image
↓
DINOv3 ViT Backbone (Frozen)
↓ patch embeddings
Reshaped into a 2D grid (H/16 × W/16)
↓
Lightweight 3-layer CNN Decoder
↓
Upsampled segmentation mask (1 × H × W)

````

---

# 📈 Training Configuration

- Backbone: **facebook/dinov3-vits16-pretrain-lvd1689m**  
- Decoder: **3-layer CNN**  
- Loss: **BCEWithLogitsLoss**  
- Optimizer: **AdamW**  
- Image size: **448×448**  
- Epochs: **20**  
- Batch size: **8**  
- Metric: **IoU (Intersection-over-Union)**  

---

# 📊 Training Curves

## 🔹 Train Iteration Loss
![Train Iteration Loss](plots/1_Train_Iteration_Loss.png)

## 🔹 Train Iteration IoU
![Train Iteration IoU](plots/2_Train_Iteration_IoU.png)

## 🔹 Train vs Validation Loss (Mean ± Std)
![Train vs Validation Loss](plots/3_Train_Val_Loss_MeanStd.png)

## 🔹 Validation Iteration Loss
![Val Iteration Loss](plots/4_Val_Iteration_Loss.png)

## 🔹 Train vs Validation IoU (Mean ± Std)
![Train vs Validation IoU](plots/5_Train_Val_IoU_MeanStd.png)

## 🔹 Validation Iteration IoU
![Val Iteration IoU](plots/6_Val_Iteration_IoU.png)

---

# 📜 Epoch-by-Epoch IoU Log (Screenshot)

This screenshot shows the steady increase in IoU over epochs.

![Epoch Log](plots/EkranResmi.png)

*(Rename your file to `EkranResmi.png` inside `plots/` before pushing.)*

---

# 🎲 Qualitative Results

## 🔹 20 Random Predictions
Demonstrates strong generalization across species, lighting, and rotation.

![Random Predictions](plots/Unknown-8.png)

## 🥇 Best 5 Predictions (IoU ~98.3–98.8%)
Nearly perfect overlaps between predicted and ground truth masks.

![Best Predictions](plots/Unknown-10.png)

## ⚠️ Worst 5 Predictions (IoU ~56–79%)
Performance drops for:
- extreme rotations  
- very thin fish  
- heavy reflections  
- occlusions  

![Worst Predictions](plots/Unknown-9.png)

---

# 🚀 Installation

```bash
git clone https://github.com/PEPEZHK/Fish-Segmentation-DINOv3
cd Fish-Segmentation-DINOv3
pip install -r requirements.txt
````

---

# 🏋️‍♂️ Training

```bash
python src/train.py \
    --data-root "/path/to/fish_dataset/Fish_Dataset/Fish_Dataset" \
    --epochs 20 \
    --batch-size 8 \
    --image-size 448
```

---

# 🔍 Evaluation Only

```bash
python src/train.py --eval-only
```

---

# 📦 Repository Structure

```
Fish-Segmentation-DINOv3/
│
├── notebooks/
│   └── DINOv3_Fish_Segmentation_Final.ipynb
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│
├── plots/
│   ├── 1_Train_Iteration_Loss.png
│   ├── 2_Train_Iteration_IoU.png
│   ├── 3_Train_Val_Loss_MeanStd.png
│   ├── 4_Val_Iteration_Loss.png
│   ├── 5_Train_Val_IoU_MeanStd.png
│   ├── 6_Val_Iteration_IoU.png
│   ├── Unknown-8.png
│   ├── Unknown-9.png
│   ├── Unknown-10.png
│   └── EkranResmi.png
│
├── outputs/
│   └── checkpoints/
│
└── README.md
```

---

# 🏁 Conclusion

This project shows how a **frozen DINOv3 backbone + small decoder** can deliver high-quality segmentation masks with:

* **95%+ IoU**
* fast training
* strong robustness to rotation and lighting
* minimal overfitting

DINOv3 proves to be a powerful universal feature extractor for downstream vision tasks.

---

```

---

# ✅ Done — This is the *complete* README in one block.  
If you want:

📌 badges  
📌 inference script section  
📌 architecture diagram  
📌 citation section  

—I can add them too.
```
