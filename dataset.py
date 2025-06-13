# file: dataset.py
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import scipy.io
import torchvision.transforms as transforms
import random

class ShanghaiTechDataset(Dataset):
    def __init__(self, root_path, part='part_A_final', mode='train', patch_size=64, transform=None):
        self.root = os.path.join(root_path, part, f'{mode}_data')
        self.img_dir = os.path.join(self.root, 'images')
        self.gt_dir = os.path.join(self.root, 'ground_truth')
        self.patch_size = patch_size

        self.img_files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
        # Default transforms if none are provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # --- Load Image and Ground Truth ---
        img_name = self.img_files[index]
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, f"GT_{img_name.replace('.jpg', '.mat')}")
        
        image = Image.open(img_path).convert('RGB')
        gt_data = scipy.io.loadmat(gt_path)
        # Coordinates are in shape [num_heads, 2]
        coords = gt_data['image_info'][0, 0][0, 0][0]

        # --- Random Cropping and Flipping ---
        w, h = image.size
        
        # Random crop coordinates
        crop_x = random.randint(0, w - self.patch_size)
        crop_y = random.randint(0, h - self.patch_size)
        
        # Crop the image
        img_patch = image.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))
        
        # Adjust ground truth coordinates to be relative to the patch
        patch_coords = []
        for x, y in coords:
            if crop_x <= x < crop_x + self.patch_size and crop_y <= y < crop_y + self.patch_size:
                patch_coords.append([x - crop_x, y - crop_y])
        patch_coords = np.array(patch_coords)
        
        # Random horizontal flip
        if random.random() > 0.5:
            img_patch = img_patch.transpose(Image.FLIP_LEFT_RIGHT)
            if len(patch_coords) > 0:
                # Flip the x-coordinates
                patch_coords[:, 0] = self.patch_size - 1 - patch_coords[:, 0]

        # --- Create Target Dot Map ---
        dot_map = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        for x, y in patch_coords:
            # Ensure coordinates are within bounds before placing dots
            if 0 <= x < self.patch_size and 0 <= y < self.patch_size:
                dot_map[int(y), int(x)] = 1.0

        # --- Convert to Tensors ---
        cond_image_tensor = self.transform(img_patch)
        dot_map_tensor = torch.from_numpy(dot_map).unsqueeze(0) # Add channel dimension
        
        return cond_image_tensor, dot_map_tensor

if __name__ == '__main__':
    # --- Test the dataset ---
    dataset = ShanghaiTechDataset(root_path=r"C:\Users\Mehmet_Postdoc\Desktop\ShanghaiTech_Crowd_Counting_Dataset", patch_size=64)
    print(f"Found {len(dataset)} images in the training set.")
    
    # Get a sample
    cond_img, dot_map = dataset[0]
    
    print(f"Condition image shape: {cond_img.shape}")
    print(f"Dot map shape: {dot_map.shape}")
    print(f"Number of dots in sample: {int(dot_map.sum())}")
    
    # Check if a DataLoader works
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    cond_batch, dots_batch = next(iter(dataloader))
    print(f"Batch of condition images shape: {cond_batch.shape}")
    print(f"Batch of dot maps shape: {dots_batch.shape}")
    print("Dataset and DataLoader test passed!")

    import matplotlib.pyplot as plt

    # Visualize a sample
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cond_img.permute(1, 2, 0).numpy())
    plt.title("Condition Image")
    plt.subplot(1, 2, 2)
    plt.imshow(dot_map.squeeze().numpy(), cmap='hot')
    plt.title("Dot Map")
    plt.show()