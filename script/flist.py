import os
import argparse
import numpy as np
import glob

def generate_celeba_flists(eval_file, images_dir, out_dir):
    train_list = []
    val_list   = []
    test_list  = []
    
    print(f"Reading CelebA partitions from {eval_file}...")
    with open(eval_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
            
        filename = parts[0]
        partition = int(parts[1])
        
        full_path = os.path.join(images_dir, filename)
        
        if partition == 0:
            train_list.append(full_path)
        elif partition == 1:
            val_list.append(full_path)
        elif partition == 2:
            test_list.append(full_path)

    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'celeba_train.flist'), 'w') as f:
        for p in train_list:
            f.write(p + '\n')
    
    with open(os.path.join(out_dir, 'celeba_val.flist'), 'w') as f:
        for p in val_list:
            f.write(p + '\n')
            
    with open(os.path.join(out_dir, 'celeba_test.flist'), 'w') as f:
        for p in test_list:
            f.write(p + '\n')
            
    print(f"CelebA Flists generated: {len(train_list)} train, {len(val_list)} val, {len(test_list)} test.")

def generate_mask_flists(test_mask_dir, out_dir):
    import random
    print("Generating mask flists strictly from testing_mask_dataset, splitting 50/50...")
    
    masks = []
    for ext in ['*.jpg', '*.png']:
        masks.extend(glob.glob(os.path.join(test_mask_dir, '**', ext), recursive=True))
        
    # Sort for determinism, then shuffle
    masks = sorted(masks)
    random.seed(1337)
    random.shuffle(masks)
    
    split_idx = len(masks) // 2
    train_masks = masks[:split_idx]
    test_masks = masks[split_idx:]
        
    with open(os.path.join(out_dir, 'masks_train.flist'), 'w') as f:
        for p in train_masks:
            f.write(p + '\n')
            
    with open(os.path.join(out_dir, 'masks_test.flist'), 'w') as f:
        for p in test_masks:
            f.write(p + '\n')

    print(f"Mask Flists generated: {len(train_masks)} train, {len(test_masks)} test.")

def main():
    parser = argparse.ArgumentParser()
    # Paths according to the user's setup
    parser.add_argument('--eval_file', type=str, default='/mnt/datadrive/inpaint/CelebA/Eval/list_eval_partition.txt')
    parser.add_argument('--images_dir', type=str, default='/mnt/datadrive/inpaint/CelebA/test/')
    
    # We ignore the --train_mask_dir argument now based on user instruction
    parser.add_argument('--test_mask_dir', type=str, default='/mnt/datadrive/inpaint/iregularmask/test_mask/mask/testing_mask_dataset')
    
    parser.add_argument('--out_dir', type=str, default='../flists/')
    
    args = parser.parse_args()
    
    # Ensure relative out_dir resolves from script location to root/flists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    final_out_dir = os.path.normpath(os.path.join(script_dir, args.out_dir))
    
    generate_celeba_flists(args.eval_file, args.images_dir, final_out_dir)
    generate_mask_flists(args.test_mask_dir, final_out_dir)
    
if __name__ == '__main__':
    main()
