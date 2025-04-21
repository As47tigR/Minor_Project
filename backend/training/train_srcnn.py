import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import json
from datetime import datetime

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.srcnn import SRCNN

class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None, target_size=(128, 128), scale_factor=4):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.target_size = target_size
        self.scale_factor = scale_factor
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.lr_files)
    
    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Resize images maintaining the scale factor
        lr_size = self.target_size
        hr_size = (self.target_size[0] * self.scale_factor, 
                  self.target_size[1] * self.scale_factor)
        
        lr_img = lr_img.resize(lr_size, Image.BICUBIC)
        hr_img = hr_img.resize(hr_size, Image.BICUBIC)
        
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)
        
        return lr_img, hr_img

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 4  # Reduced from 16
    num_epochs = 50
    learning_rate = 1e-4
    scale_factor = 4
    target_size = (128, 128)  # Size for LR images
    gradient_accumulation_steps = 4  # Accumulate gradients for effective batch size of 16
    
    # Create model
    model = SRCNN(scale_factor=scale_factor).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    train_dataset = DIV2KDataset(
        lr_dir=os.path.join(parent_dir, 'Dataset/DIV2K_train_LR'),
        hr_dir=os.path.join(parent_dir, 'Dataset/DIV2K_train_HR'),
        transform=transform,
        target_size=target_size,
        scale_factor=scale_factor
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True  # Enable pin_memory for faster data transfer
    )
    
    # Training metrics
    metrics = {
        'epochs': [],
        'losses': [],
        'psnr_values': [],
        'training_times': [],
        'upscaling_times': []
    }
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        total_psnr = 0
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device, non_blocking=True)  # Enable non_blocking for faster data transfer
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Step optimizer only after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Calculate PSNR
            psnr = calculate_psnr(outputs, hr_imgs)
            
            total_loss += loss.item() * gradient_accumulation_steps  # Scale back the loss
            total_psnr += psnr.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item() * gradient_accumulation_steps:.4f}, PSNR: {psnr:.2f} dB')
        
        # Calculate average metrics for this epoch
        avg_loss = total_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        # Save metrics
        metrics['epochs'].append(epoch + 1)
        metrics['losses'].append(avg_loss)
        metrics['psnr_values'].append(avg_psnr)
        metrics['training_times'].append(epoch_time)
        
        # Test upscaling time with smaller image
        test_img = torch.randn(1, 3, 32, 32).to(device)  # Reduced from 64x64
        upscale_start = time.time()
        with torch.no_grad():
            model(test_img)
        upscale_time = time.time() - upscale_start
        metrics['upscaling_times'].append(upscale_time)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Average Loss: {avg_loss:.4f}, '
              f'Average PSNR: {avg_psnr:.2f} dB, '
              f'Epoch Time: {epoch_time:.2f}s, '
              f'Upscaling Time: {upscale_time:.4f}s')
        
        # Save model checkpoint
        model_path = os.path.join(parent_dir, f'models/UpscalingModel_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'psnr': avg_psnr
        }, model_path)
        
        # Save metrics
        metrics_path = os.path.join(parent_dir, f'metrics/training_metrics_{epoch+1}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    total_training_time = time.time() - start_time
    print(f"Training completed in {total_training_time:.2f} seconds")
    
    # Save final metrics
    final_metrics = {
        'total_training_time': total_training_time,
        'final_loss': metrics['losses'][-1],
        'final_psnr': metrics['psnr_values'][-1],
        'average_upscaling_time': np.mean(metrics['upscaling_times']),
        'device_used': str(device),
        'model_architecture': 'SRCNN',
        'scale_factor': scale_factor,
        'batch_size': batch_size * gradient_accumulation_steps,  # Report effective batch size
        'learning_rate': learning_rate,
        'num_epochs': num_epochs
    }
    
    with open(os.path.join(parent_dir, 'metrics/final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)

if __name__ == '__main__':
    train_model() 