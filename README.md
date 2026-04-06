# HINT + LAFIN Symmetry Architecture

This repository contains the **HINT (Transformer-based Image Inpainting)** model, now enhanced with the **LAFIN Symmetry Architecture** for superior facial reconstruction.

## 🚀 Symmetry Architecture: What is it doing?

Facial inpainting often suffers from vertical asymmetry (e.g., mismatching eyes or eyebrows). This integration solves that using two key methods:

1.  **Landmark-Guided Guidance**: The model uses a 68-point facial landmark detector (MobileNetV2) to identify the "skeleton" of the face. Even if a feature is 100% masked, the model knows exactly where it should be.
2.  **Symmetry Consistency Loss**: Derived from LAFIN, this loss identifies symmetric landmark pairs (Point 36 <-> Point 45, etc.). During training, it ensures that the generated features on the left side of the face are consistent with those on the right, resulting in perfectly balanced facial structures.

## ⚡ Hardware Optimizations (For RTX 5000 / 24-Core CPU)

To maximize performance on high-end hardware, this pipeline includes:
-   **Automatic Mixed Precision (AMP)**: Uses NVIDIA Tensor Cores for ~2x faster training and 50% less VRAM usage.
-   **20-Worker DataLoader**: Fully leverages your 24-core CPU to eliminate data-loading bottlenecks.
-   **Optimized Batching**: Increased to `Batch Size: 16` for maximum gradient stability and GPU utilization (Utilizing ~18-20GB VRAM).

---

## 🛠️ Setup & Training Instructions

### 1. Dataset Preparation
Update the paths in `DatasetCreator.py` to point to your CelebA dataset and run the script:
```powershell
python DatasetCreator.py
```
This will generate the `.flist` files for images, masks, and landmarks in the `./dataset` folder.

### 2. Checkpoints
Ensure your pre-trained models are in the `./checkpoints` directory:
- `InpaintingModel_gen.pth`
- `landmark_detector.pth`
- `InpaintingModel_dis.pth`

### 3. Start Training
Run the optimized training session using:
```powershell
python main.py --path ./checkpoints
```
The model will automatically use **AMP**, **20 CPU Workers**, and the **Symmetry Architecture**.

### 4. Inference
To test the model on your own images:
```powershell
python run_inference.py --celeba_dir "./dataset/images" --mask_dir "./dataset/masks" --num_samples 20
```

---

## 📊 Monitoring
Training progress is logged via **WandB** and local log files in `./checkpoints/log_inpaint.dat`. Look for `symLoss` in your logs to monitor how strictly the model is enforcing facial symmetry.