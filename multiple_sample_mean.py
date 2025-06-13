# file: sample.py
import torch
import os
import argparse
from PIL import Image
import numpy as np
import scipy.io
import scipy.ndimage as ndimage
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

from model import ConditionalDiffusionModel64
from diffusion import DiscreteDiffusion

def sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Load Model and Diffusion ---
    print("Loading model...")
    model = ConditionalDiffusionModel64(vgg_pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    diffusion = DiscreteDiffusion(timesteps=args.timesteps, num_classes=2).to(device)

    # --- Load and Preprocess Condition Image ---
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cond_image = Image.open(args.cond_img).convert("RGB")
    cond_tensor = transform(cond_image).unsqueeze(0).to(device)
    
    print(f"Generating {args.num_samples} samples for {args.cond_img} and averaging...")
    all_samples = []
    with torch.no_grad():
        for _ in tqdm(range(args.num_samples), desc="Generating Samples"):
            generated_sample = diffusion.sample(
                model,
                image_size=args.img_size,
                batch_size=1,
                condition_image=cond_tensor
            ).cpu()
            all_samples.append(generated_sample)

    # --- Average Samples and Post-process ---
    stacked_samples = torch.cat(all_samples, dim=0)
    mean_sample = torch.mean(stacked_samples, dim=0, keepdim=True)

    # --- MODIFICATION START: Adaptive Thresholding and Blob Center Detection ---
    # Find the maximum value in the averaged density map
    max_density_val = torch.max(mean_sample)
    
    # Calculate the dynamic threshold based on the max value and user-provided factor
    dynamic_threshold = max_density_val * args.threshold
    
    print(f"\nMax density value in averaged map: {max_density_val.item():.4f}")
    print(f"Applying relative threshold factor: {args.threshold}")
    print(f"Calculated dynamic threshold for mask: {dynamic_threshold.item():.4f}")

    # Create a binary mask by applying the dynamic threshold
    binary_mask = (mean_sample > dynamic_threshold).float()

    # Convert tensors to numpy for scipy processing
    mean_sample_np = mean_sample.squeeze().numpy()
    binary_mask_np = binary_mask.squeeze().numpy()

    # Find connected components (blobs) in the binary mask
    labeled_array, num_features = ndimage.label(binary_mask_np)
    print(f"Found {num_features} blobs (potential detections).")

    # Find the center of mass for each labeled blob
    centers = ndimage.center_of_mass(mean_sample_np, labeled_array, range(1, num_features + 1))
    
    # Create the final dot map
    final_dot_map_np = np.zeros((args.img_size, args.img_size), dtype=np.float32)
    for i, (y, x) in enumerate(centers):
        y_int, x_int = int(round(y)), int(round(x))
        if 0 <= y_int < args.img_size and 0 <= x_int < args.img_size:
            final_dot_map_np[y_int, x_int] = 1.0

    final_dot_map_tensor = torch.from_numpy(final_dot_map_np).unsqueeze(0).unsqueeze(0)
    # --- MODIFICATION END ---


    # --- Create Visualization Grid ---
    cond_image_vis = (cond_tensor.cpu() * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    images_to_show = [cond_image_vis]
    
    if args.gt_mat:
        gt_data = scipy.io.loadmat(args.gt_mat)
        coords = gt_data['image_info'][0, 0][0, 0][0]
        orig_w, orig_h = Image.open(args.cond_img).size
        scale_w = args.img_size / orig_w
        scale_h = args.img_size / orig_h
        true_dot_map = np.zeros((args.img_size, args.img_size), dtype=np.float32)
        for x, y in coords:
            nx, ny = int(x * scale_w), int(y * scale_h)
            if 0 <= nx < args.img_size and 0 <= ny < args.img_size:
                true_dot_map[ny, nx] = 1.0
        true_dot_map = torch.from_numpy(true_dot_map).unsqueeze(0).unsqueeze(0)
        images_to_show.append(true_dot_map.repeat(1, 3, 1, 1))

    images_to_show.extend([
        mean_sample.repeat(1, 3, 1, 1),
        binary_mask.repeat(1, 3, 1, 1),
        final_dot_map_tensor.repeat(1, 3, 1, 1)
    ])

    comparison_grid = torch.cat(images_to_show, dim=3)
    save_image(comparison_grid, args.out, normalize=False)
    print(f"\nSaved visualization grid to {args.out}")
    print(f"Final Count: {num_features}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate samples from a trained crowd counting model.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--cond_img', type=str, required=True, help="Path to the conditional image.")
    parser.add_argument('--gt_mat', type=str, default=None, help="(Optional) Path to the ground truth .mat file for comparison.")
    parser.add_argument('--out', type=str, default=r"C:\Users\Mehmet_Postdoc\Desktop\python_set_up_code\Discrete_Diffusion_Shanghai_Tech\generated_sample.png", help="Output filename.")
    parser.add_argument('--img_size', type=int, default=64, help='Image size (must match model).')
    parser.add_argument('--timesteps', type=int, default=200, help="Number of timesteps (must match training).")
    parser.add_argument('--num_samples', type=int, default=1000, help="Number of samples to generate and average.")
    # --- MODIFICATION: Updated help text for threshold argument ---
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help="Relative threshold factor (e.g., 0.5 for 50%%) to multiply by the max density value to create the binary mask.")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sample(args)