import os
import argparse
import numpy as np
import collections
from skimage import io, color
import face_alignment
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, flist_path, lmk_dir):
        with open(flist_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        print(f"Total entries in {os.path.basename(flist_path)}: {len(lines)}")
        
        self.flist_path = flist_path
        self.lmk_dir = lmk_dir
        self.original_lines = lines
        
        # Determine remaining
        self.to_process = []
        for img_path in lines:
            basename = os.path.basename(img_path).split('.')[0]
            lmk_path = os.path.join(lmk_dir, f"{basename}.txt")
            if not (os.path.exists(lmk_path) and os.path.getsize(lmk_path) > 0):
                self.to_process.append((img_path, lmk_path))
                
        print(f"Remaining to process: {len(self.to_process)}")
        
    def __len__(self):
        return len(self.to_process)
        
    def __getitem__(self, idx):
        img_path, lmk_path = self.to_process[idx]
        try:
            img = io.imread(img_path)
            if img.ndim == 2:
                img = color.gray2rgb(img)
            elif img.shape[2] == 4:
                img = img[..., :3]
            
            # Convert to PyTorch tensor format for FaceAlignment: C, H, W
            # FaceAlignment get_landmarks_from_batch expects tensor in (B, 3, H, W)
            # scaled 0-255
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            return img_tensor, img_path, lmk_path
        except Exception as e:
            return torch.zeros(3, 218, 178), img_path, lmk_path # return empty tensor as fallback

def custom_collate(batch):
    # Filter out empty paths or unreadable images if necessary
    # Batch could have varying sizes if CelebA isn't strictly unified.
    # We will try to stack them. If they fail to stack, we'll return a list.
    try:
        tensors = torch.stack([item[0] for item in batch])
    except RuntimeError:
        # If sizes mismatch, cannot batch
        tensors = [item[0] for item in batch]
        
    img_paths = [item[1] for item in batch]
    lmk_paths = [item[2] for item in batch]
    return tensors, img_paths, lmk_paths

def process_dataset(dataset, fan, batch_size=64, num_workers=4):
    if len(dataset) == 0:
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=custom_collate)
                            
    for batch_tensors, img_paths, lmk_paths in tqdm(dataloader):
        # We always process sequentially so that the face detector (which handles varying scales)
        # runs properly before the alignment network. Dataloader is strictly for fast async IO fetching.
        if torch.is_tensor(batch_tensors):
            for i in range(len(img_paths)):
                tensor = batch_tensors[i]
                img_path = img_paths[i]
                lmk_path = lmk_paths[i]
                
                try:
                    preds = fan.get_landmarks(tensor.permute(1,2,0).cpu().numpy())
                    if preds is not None and len(preds) > 0:
                        l_pos = preds[0] # first face
                        with open(lmk_path, 'w') as f:
                            for p in range(68):
                                f.write(str(l_pos[p, 0]) + ' ' + str(l_pos[p, 1]) + ' ')
                            f.write('\n')
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        else:
            # Different sizes
            for tensor, img_path, lmk_path in zip(batch_tensors, img_paths, lmk_paths):
                try:
                    preds = fan.get_landmarks(tensor.permute(1,2,0).cpu().numpy())
                    if preds is not None and len(preds) > 0:
                        l_pos = preds[0]
                        with open(lmk_path, 'w') as f:
                            for p in range(68):
                                f.write(str(l_pos[p, 0]) + ' ' + str(l_pos[p, 1]) + ' ')
                            f.write('\n')
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

def generate_outputs(dataset):
    flist_path = dataset.flist_path
    lmk_dir = dataset.lmk_dir
    original_lines = dataset.original_lines
    print(f"Updating flist to remove entries without valid landmarks for {flist_path}...")
    
    valid_lines = []
    for img_path in original_lines:
        basename = os.path.basename(img_path).split('.')[0]
        lmk_path = os.path.join(lmk_dir, f"{basename}.txt")
        if os.path.exists(lmk_path) and os.path.getsize(lmk_path) > 0:
            valid_lines.append(img_path)
            
    print(f"Original entries: {len(original_lines)}. Valid entries: {len(valid_lines)}. Dropped: {len(original_lines) - len(valid_lines)}")
    
    # Overwrite the original flist!
    with open(flist_path, 'w') as f:
        for line in valid_lines:
            f.write(line + '\n')
            
    # Generate the landmark flist directly matching the image flist.
    lmk_flist_path = flist_path.replace('celeba_', 'landmarks_')
    print(f"Generating landmark flist: {lmk_flist_path}")
    with open(lmk_flist_path, 'w') as f:
        for line in valid_lines:
            basename = os.path.basename(line).split('.')[0]
            # Write absolute paths pointing to the unified landmark folder
            abs_lmk_path = os.path.abspath(os.path.join(lmk_dir, f"{basename}.txt"))
            f.write(abs_lmk_path + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flists_dir', type=str, default='./flists', help='path to the directory containing flists')
    parser.add_argument('--device', type=str, default='cuda', help='device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for prediction')
    parser.add_argument('--num_workers', type=int, default=8, help='Dataloader num_workers')
    args = parser.parse_args()
    
    print("Initializing FaceAlignment model...")
    fan = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=args.device)
    
    # ALL targets will map to one single folder 'dataset/landmark'
    unified_landmark_dir = os.path.abspath(os.path.join(args.flists_dir, '../dataset/landmark'))
    os.makedirs(unified_landmark_dir, exist_ok=True)
    
    tasks = [
        'celeba_test.flist',
        'celeba_val.flist',
        'celeba_train.flist'
    ]
    
    for flist_name in tasks:
        flist_path = os.path.join(args.flists_dir, flist_name)
        if not os.path.exists(flist_path):
            print(f"Warning: {flist_path} not found. Skipping.")
            continue
            
        dataset = ImageDataset(flist_path, unified_landmark_dir)
        process_dataset(dataset, fan, batch_size=args.batch_size, num_workers=args.num_workers)
        # Ensure we write out the newly verified valid subsets afterwards
        generate_outputs(dataset)

if __name__ == '__main__':
    main()
