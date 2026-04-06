import os

# Update these paths to your actual locations
# Update these paths to your actual locations
# For Windows, ensure your dataset is reachable
partition_file = './dataset/list_eval_partition.txt'
image_dir = './dataset/images'
output_dir = './dataset'
test_mask_dir = './dataset/masks/test'
train_mask_dir = './dataset/masks/train'
landmark_dir = './dataset/landmarks' # Path to CelebA landmarks if available

# 1. Processing Image & Landmark Partitions
train_list, val_list, test_list = [], [], []
train_landmarks, val_landmarks, test_landmarks = [], [], []

print("Reading partition file and matching landmarks...")
with open(partition_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2: continue
        
        filename, split_id = parts[0], parts[1]
        full_path = os.path.abspath(os.path.join(image_dir, filename))
        
        # Match landmark file (assuming filename.txt or filename.jpg.txt)
        lmk_filename = filename.split('.')[0] + '.txt'
        lmk_path = os.path.abspath(os.path.join(landmark_dir, lmk_filename))
        
        if split_id == '0':
            train_list.append(full_path)
            if os.path.exists(lmk_path): train_landmarks.append(lmk_path)
        elif split_id == '1':
            val_list.append(full_path)
            if os.path.exists(lmk_path): val_landmarks.append(lmk_path)
        elif split_id == '2':
            test_list.append(full_path)
            if os.path.exists(lmk_path): test_landmarks.append(lmk_path)

# 2. Scanning Mask Folders
print("Generating mask lists from folders...")
def get_all_mask_paths(mask_dir):
    paths = []
    if os.path.exists(mask_dir):
        for f in os.listdir(mask_dir):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                paths.append(os.path.abspath(os.path.join(mask_dir, f)))
    return sorted(paths)

train_masks = get_all_mask_paths(train_mask_dir)
test_masks  = get_all_mask_paths(test_mask_dir)

# 3. Saving all flist files
def save_flist(flist, name):
    if not flist: 
        print(f"Skipping empty list: {name}")
        return
    save_path = os.path.join(output_dir, name)
    with open(save_path, 'w') as f:
        f.write('\n'.join(flist))
    print(f"Saved: {save_path} ({len(flist)} entries)")

save_flist(train_list, 'celeba_train.flist')
save_flist(train_landmarks, 'celeba_train_landmarks.flist')
save_flist(val_list, 'celeba_val.flist')
save_flist(val_landmarks, 'celeba_val_landmarks.flist')
save_flist(test_list, 'celeba_test.flist')
save_flist(test_landmarks, 'celeba_test_landmarks.flist')
save_flist(train_masks, 'masks_train.flist')
save_flist(test_masks, 'masks_test.flist')

print("\nAll flist files generated successfully!")

