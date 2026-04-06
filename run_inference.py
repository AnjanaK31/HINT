import os
import random
import glob
import shutil
import numpy as np
from PIL import Image
import subprocess
import argparse

def run():
    parser = argparse.ArgumentParser(description="Run HINT inference with custom paths and GPU support.")
    parser.add_argument("--celeba_dir", type=str, required=True, help="Path to the CelebA dataset (images).")
    parser.add_argument("--mask_dir", type=str, default=None, help="Path to the irregular mask dataset.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of images to sample.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--celeba_test_split", type=int, default=182638, help="Starting ID for the official CelebA test split.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID(s) to use (e.g., '0' or '0,1').")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode.")
    parser.add_argument("--output_inputs", type=str, default="./test_inputs", help="Temp input directory.")
    parser.add_argument("--output_masks", type=str, default="./test_masks", help="Temp mask directory.")
    parser.add_argument("--results_dir", type=str, default="./test_outputs", help="Results directory.")

    args = parser.parse_args()

    os.makedirs(args.output_inputs, exist_ok=True)
    os.makedirs(args.output_masks, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Clean existing temp files if any
    for d in [args.output_inputs, args.output_masks, args.results_dir]:
        for f in glob.glob(os.path.join(d, "*")):
            if os.path.isfile(f):
                os.remove(f)

    all_images = glob.glob(os.path.join(args.celeba_dir, "*.jpg")) + glob.glob(os.path.join(args.celeba_dir, "*.png"))
    
    # Optional filtering for CelebA test split (assuming numbered filenames)
    if args.celeba_test_split > 0:
        def get_id(path):
            try: return int(os.path.basename(path).split(".")[0])
            except: return -1
        all_images = [f for f in all_images if get_id(f) >= args.celeba_test_split]

    if not all_images:
        print(f"Error: No images found in {args.celeba_dir}.")
        return

    random.shuffle(all_images)
    if args.num_samples == -1:
        num_samples = len(all_images)
    else:
        num_samples = min(args.num_samples, len(all_images))
    sampled_images = all_images[:num_samples]

    print(f"Sampled {len(sampled_images)} images. Preparing masks...")

    mask_files = []
    if args.mask_dir and os.path.isdir(args.mask_dir):
        mask_files = glob.glob(os.path.join(args.mask_dir, "*.jpg")) + glob.glob(os.path.join(args.mask_dir, "*.png"))
        if not mask_files:
            print(f"Warning: No masks found in {args.mask_dir}. Falling back to random rectangular masks.")

    for i, img_path in enumerate(sampled_images):
        # copy image
        base_name = os.path.basename(img_path)
        dest_img_path = os.path.join(args.output_inputs, f"{i:05d}_{base_name}")
        shutil.copyfile(img_path, dest_img_path)

        # generate or copy mask
        img = Image.open(img_path)
        w, h = img.size
        
        if mask_files:
            # Pick a random mask
            mask_path = random.choice(mask_files)
            mask_img = Image.open(mask_path).convert('L')
            # Resize mask to match image
            mask_img = mask_img.resize((w, h), Image.NEAREST)
            dest_mask_path = os.path.join(args.output_masks, f"{i:05d}_{base_name}")
            mask_img.save(dest_mask_path)
        else:
            # generate random rectangular mask
            mask = np.zeros((h, w), dtype=np.uint8)
            mask_w = random.randint(w // 4, w // 2)
            mask_h = random.randint(h // 4, h // 2)
            mask_x = random.randint(0, max(1, w - mask_w))
            mask_y = random.randint(0, max(1, h - mask_h))
            mask[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = 255
            mask_img = Image.fromarray(mask)
            dest_mask_path = os.path.join(args.output_masks, f"{i:05d}_{base_name}")
            mask_img.save(dest_mask_path)
    
    print("Executing HINT test.py...")
    # current_dir = os.getcwd()
    hint_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(hint_dir)
    os.environ["WANDB_MODE"] = "disabled"
    
    # We call python test.py --input test_inputs --mask test_masks --output test_outputs
    cmd = ["python", "test.py", "--path", "./checkpoints", "--input", args.output_inputs, "--mask", args.output_masks, "--output", args.results_dir]
    
    if args.cpu:
        cmd.append("--cpu")
    else:
        cmd.extend(["--gpu", args.gpu, "--batch_size", str(args.batch_size)])

    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("Inference completed successfully!")
    except subprocess.CalledProcessError as e:
        print("Error running inference:", e)

if __name__ == "__main__":
    run()
