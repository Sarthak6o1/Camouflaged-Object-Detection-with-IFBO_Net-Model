# Camouflaged Object Detection using SegNet, U-Net, and IFBONet

This project implements Camouflaged Object Detection (COD) using three deep learning architectures: **SegNet**, **U-Net**, and **IFBONet**. The goal is to segment objects that are visually camouflaged in complex scenes using supervised semantic segmentation.

---

## 📁 Dataset: COD10K

Dataset used: [COD10K on Kaggle](https://www.kaggle.com/datasets/getcam/cod10k)

### Structure:

- 10,000 camouflaged object images with annotations.
- Divided into **Train**, **Validation**, and **Test** sets.
- Each sample includes:
  - RGB image
  - GT\_Object (segmentation mask)
  - GT\_Edge (object edges)
  - GT\_Instance (instance masks)

```bash
!unzip "/content/drive/MyDrive/archive.zip" -d /content/
```

---

## 🧠 Architectures Used

### 🔹 U-Net

- Encoder-decoder network with skip connections.
- Captures context and enables precise localization.

### 🔹 SegNet

- Encoder-decoder based on VGG16.
- Upsampling uses pooling indices from encoder.

### 🔹 IFBONet

- A custom architecture for COD based on **Integrate Fusion and Boundary Optimization**.
- Performed with:
  - **SGD optimizer**
  - **Learning Rate Scheduler**
  - **Gradient Clipping**
  - **CrossEntropy + Dice Loss**

---

## ⚙️ Training Setup

- Platform: Google Colab
- GPU: Tesla T4
- Framework: PyTorch
- Loss: Binary Cross-Entropy + Dice
- Optimizer: SGD with momentum
- Scheduler: StepLR/ReduceLROnPlateau
- Augmentations: RandomFlip, Normalize, Resize

---

## 📊 PyTorch Training Pipeline

```python
def train(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)
```

---

## 📊 PyTorch Evaluation Pipeline

```python
def evaluate(model, loader, criterion, device):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            eval_loss += loss.item()
    return eval_loss / len(loader)
```

---

## 📂 Folder Structure

```
Camouflaged-Object-Detection/
|
├── unet.py
├── segnet_model.py
├── ifbonet.py
└── README.md
```
## Use the s_measure,fbw,mae,ephi fucntions from the ifbonet.py

---

## 🙌 Credits

- [COD10K Dataset on Kaggle](https://www.kaggle.com/datasets/getcam/cod10k)

- 


