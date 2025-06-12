# file: train.py
import torch
import os
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import time

from model import ConditionalDiffusionModel64
from diffusion import DiscreteDiffusion
from dataset import ShanghaiTechDataset

def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    sample_dir = os.path.join(args.save_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset = ShanghaiTechDataset(root_path=args.dataset_path, patch_size=args.img_size, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = ConditionalDiffusionModel64(vgg_pretrained=True).to(device)
    
    # --- (MODIFIED) Instantiate Diffusion with Focal Loss parameters ---
    print(f"Using Focal Loss with gamma={args.gamma} and alpha={args.alpha}")
    diffusion = DiscreteDiffusion(
        timesteps=args.timesteps,
        num_classes=2,
        focal_loss_gamma=args.gamma,
        focal_loss_alpha=args.alpha
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Found {len(dataset)} training images. Starting training for {args.epochs} epochs...")
    
    # --- Get a fixed validation batch for consistent sampling ---
    val_dataset = ShanghaiTechDataset(root_path=args.dataset_path, patch_size=args.img_size, mode='test')
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    fixed_val_batch = next(iter(val_dataloader))
    fixed_cond_images, fixed_true_dots = fixed_val_batch[0].to(device), fixed_val_batch[1]

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        model.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (condition_image, x_start) in enumerate(pbar):
            optimizer.zero_grad()
            
            condition_image = condition_image.to(device)
            x_start = x_start.to(device)
            
            loss = diffusion.compute_loss(model, x_start, condition_image)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f} | Duration: {epoch_duration:.2f}s")

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            
            model.eval()
            with torch.no_grad():
                generated_samples = diffusion.sample(
                    model, 
                    image_size=args.img_size, 
                    batch_size=fixed_cond_images.shape[0], 
                    condition_image=fixed_cond_images
                ).cpu()
            
            cond_rgb = torch.stack([
                (fixed_cond_images[i].cpu() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                for i in range(fixed_cond_images.shape[0])
            ])
            true_dots_rgb = fixed_true_dots.repeat(1, 3, 1, 1)
            generated_rgb = generated_samples.repeat(1, 3, 1, 1)
            
            comparison_grid = torch.cat([cond_rgb, true_dots_rgb, generated_rgb], dim=2)
            sample_path = os.path.join(sample_dir, f"sample_epoch_{epoch+1}.png")
            save_image(comparison_grid, sample_path, nrow=1, normalize=False)
            print(f"Saved checkpoint and samples for epoch {epoch+1}")

    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model for crowd counting.")
    parser.add_argument('--dataset_path', type=str, default=r"C:\Users\Mehmet_Postdoc\Desktop\ShanghaiTech_Crowd_Counting_Dataset", help='Path to the ShanghaiTech dataset directory.')
    parser.add_argument('--img_size', type=int, default=64, help='Patch size for training.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--timesteps', type=int, default=200, help='Number of diffusion timesteps.')
    parser.add_argument('--save_dir', type=str, default='results_ddcc', help='Directory to save results.')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint and samples every N epochs.')
    # --- (NEW) Arguments for Focal Loss ---
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss focusing parameter (gamma).')
    parser.add_argument('--alpha', type=float, default=0.7, help='Focal loss alpha parameter (weight for the positive class).')
    args = parser.parse_args()
    
    train(args)