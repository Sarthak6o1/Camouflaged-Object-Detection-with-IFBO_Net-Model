# Camouflaged Object Detection with IFBO-Net, SegNet, and UNet

This repository provides state-of-the-art PyTorch implementations for camouflaged object detection using three architectures: **IFBO-Net**, **SegNet**, and **UNet**.

-
Adjust the num_epochs Accordingly

**Lets First of All clone the repository:**
git clone https://github.com/Sarthak6o1/Camouflaged-Object-Detection-with-IFBO_Net-Model.git

## 🚀 Quick Start

1. **Mount Google Drive (for Colab):**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Prepare COD10K dataset:**  
   - Download from Kaggle: `https://www.kaggle.com/datasets/getcam/cod10k`  
   - Unzip into: `/content/drive/MyDrive/COD10K/` (or adjust paths in `dataset_loader_cod10k.py`)

3. **Run training and testing:**  
   ```bash
   # For IFBO-Net
   python IFBO_Net_model_train.py
   python IFBO_Net_model_test.py

   # For SegNet
   python SegNet_model_train.py
   python SegNet_model_test.py

   # For UNet
   python UNet_model_train.py
   python UNet_model_test.py
   ```

---

## 🔍 Project Structure

```
├── IFBO_Net_architecture.py       # IFBO-Net model definition
├── IFBO_Net_metrics.py            # Metrics for IFBO-Net
├── IFBO_Net_model_train.py        # Training pipeline (Adam optimizer)
├── IFBO_Net_model_test.py         # Evaluation pipeline
├── IFBO_Net_original.py           # Initial implementation

├── SegNet_architecture.py         # SegNet definition
├── SegNet_metrics.py              # SegNet metrics
├── SegNet_model_train.py          # Training pipeline (SGD optimizer)
├── SegNet_model_test.py           # Evaluation pipeline
├── SegNet_Original.py             # Initial Implementation

├── UNet_architecture.py           # UNet definition
├── UNet_metrics.py                # UNet metrics
├── UNet_model_train.py            # Training pipeline (SGD + scheduler)
├── UNet_model_test.py             # Evaluation pipeline
├── UNet_original.py               # Initial Implementation

├── dataset_loader_cod10k.py       # DataLoader & transforms (Resize 256×256)
├── cod10k_visualizer.py           # Initial image visualizer (raw images & masks)
├── Mounting_drive.py              # Utility for Google Drive mounting
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🛠️ Data Preparation & Loading

- **Transforms**: All images and masks are resized to `(256, 256)` and converted to tensors.
- **Datasets**: `COD10KDataset` handles reading image-mask pairs from `Train` and `Test` folders.
- **DataLoader**: Batches data (`batch_size=8`), shuffles train set, uses `num_workers=2`.

```python
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_loader = DataLoader(...)
test_loader  = DataLoader(...)
```

---

## 📈 Visualization

- **cod10k_visualizer.py** provides an **initial image visualization** of raw images and corresponding ground-truth masks using matplotlib.  
- For prediction heatmaps and advanced plotting, refer to your custom evaluation scripts.

---

## 🏋️‍♂️ Training Pipeline

Each training script follows this workflow:

1. **Setup**: Check for CUDA, instantiate model on `device`.
2. **Optimizer & Scheduler** (_UNet, SegNet_):  
   - SGD with momentum (0.9), weight decay (1e-4).  
   - `StepLR` scheduler (step_size=10, gamma=0.1).  
   - IFBO-Net uses Adam (lr=0.001).
3. **Loss Function**:  
   - IFBO-Net: `BCE` loss.  
   - SegNet & UNet: `BCEDiceLoss` (combines BCE + Dice).
4. **Training Loop**:  
   ```python
   for epoch in range(num_epochs):
       model.train()
       for images, masks in train_loader:
           outputs = model(images)
           loss = criterion(outputs, masks)
           optimizer.zero_grad()
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
           optimizer.step()
       scheduler.step()  # if using scheduler
   ```

---

## 📊 Evaluation Metrics

- **Loss**: BCE, Dice
- **Accuracy**: Pixel-level accuracy
- **MAE**: Mean absolute error
- **S-measure (Sα)**: Structural similarity
- **E-measure (Eϕ)**: Enhanced alignment
- **Fβw**: Weighted F-measure
- **Dice Coefficient**: Overlap measure
- **IoU**: Intersection over Union

---

## 📚 References

- [COD10K Dataset](https://www.kaggle.com/datasets/getcam/cod10k)
- Zhao et al., *EGNet: Edge Guidance Network for Salient Object Detection*, ICCV 2019.
- Long et al., *Fully Convolutional Networks*, CVPR 2015.
- Camouflaged object detection with integrated feature fusion
  and boundary optimization
  Bin Ge1 · Xiaolong Peng1 · Chenxing Xia1

· Hailong Chen1

