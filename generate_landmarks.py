import os
import cv2
import torch
import numpy as np
import face_alignment
from tqdm import tqdm

# === CONFIGURATION ===
ROOT = "/home/snuc/Desktop/HINTwithSymmetry/HINT"
IMAGE_FLISTS = {
    'train': os.path.join(ROOT, 'dataset/celeba_train.flist'),
    'val':   os.path.join(ROOT, 'dataset/celeba_val.flist'),
    'test':  os.path.join(ROOT, 'dataset/celeba_test.flist')
}

# RTX A5000 has 24GB. 64 is safe for 256px images; we'll auto-scale if needed.
INITIAL_BATCH_SIZE = 64 

def run_landmark_generation():
    print(f"🚀 Initializing FAN on GPU (RTX A5000 Detected)")
    # Using 2D 68-point landmarks (Standard for HINT/LaFIn)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, 
        flip_input=False, 
        device='cuda'
    )

    for split, flist_path in IMAGE_FLISTS.items():
        if not os.path.exists(flist_path):
            print(f"⚠️  Skipping {split}: Flist not found at {flist_path}")
            continue

        print(f"\n📂 Processing {split} set...")
        
        # Segregate output folders
        out_dir = os.path.join(ROOT, "dataset", f"{split}_landmarks")
        os.makedirs(out_dir, exist_ok=True)

        with open(flist_path, 'r') as f:
            img_paths = [line.strip() for line in f.readlines() if line.strip()]

        landmark_paths = []
        batch_size = INITIAL_BATCH_SIZE

        # Progress bar
        pbar = tqdm(total=len(img_paths), desc=f"Generating {split}")
        i = 0
        while i < len(img_paths):
            batch_paths = img_paths[i : i + batch_size]
            batch_imgs = []
            
            for p in batch_paths:
                img = cv2.imread(p)
                if img is not None:
                    batch_imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if not batch_imgs:
                i += batch_size
                pbar.update(len(batch_paths))
                continue

            try:
                # Batch Inference
                # input: [B, H, W, 3] numpy array
                preds = fa.get_landmarks_from_batch(np.stack(batch_imgs))
                
                for idx, (img_path, face_preds) in enumerate(zip(batch_paths, preds)):
                    fname = os.path.basename(img_path).split('.')[0]
                    save_path = os.path.join(out_dir, f"{fname}.npy")
                    
                    # If no face is detected, save zeros (68, 2) to prevent model crash
                    if face_preds is None or len(face_preds) == 0:
                        lm = np.zeros((68, 2))
                    else:
                        # Take the first (most confident) face detected
                        lm = face_preds[0] 
                    
                    np.save(save_path, lm.astype(np.float32))
                    landmark_paths.append(save_path)
                
                i += batch_size
                pbar.update(len(batch_paths))

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ OOM: Reducing batch size from {batch_size} to {batch_size//2}")
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    if batch_size < 1: raise e
                else:
                    raise e

        pbar.close()

        # Generate the .flist file HINT is looking for
        flist_out = os.path.join(ROOT, "dataset", f"celeba_{split}_landmarks.flist")
        with open(flist_out, 'w') as f:
            for lp in landmark_paths:
                f.write(lp + '\n')
        
        print(f"✅ Created {len(landmark_paths)} .npy files and {flist_out}")

if __name__ == "__main__":
    run_landmark_generation()