import os
import sys
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.srcnn import SRCNN
from utils.ai_upscale import process

def test_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test image paths
    test_dir = os.path.join(parent_dir, 'test_images')
    os.makedirs(test_dir, exist_ok=True)
    
    # Test on sample images
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No test images found in test_images directory")
        return
    
    for img_name in tqdm(image_files, desc="Processing images"):
        try:
            img_path = os.path.join(test_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            
            print(f"\nProcessing {img_name} (Size: {img.size})")
            
            # Process image with patch-based upscaling
            upscaled_img = process(
                img, 
                scale_factor=4,
                patch_size=128,  # Process in 128x128 patches
                stride=64        # 50% overlap between patches
            )
            
            # Save results
            output_dir = os.path.join(test_dir, 'results')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save original and upscaled images
            img.save(os.path.join(output_dir, f'original_{img_name}'))
            upscaled_img.save(os.path.join(output_dir, f'upscaled_{img_name}'))
            
            print(f"Successfully processed {img_name}")
            print(f"Original size: {img.size}")
            print(f"Upscaled size: {upscaled_img.size}")
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue

if __name__ == '__main__':
    test_model() 