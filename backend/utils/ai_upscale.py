import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import glob
import sys
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("upscale_errors.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_upscale")

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.srcnn import SRCNN

def get_latest_model():
    """Get the path to the latest trained model"""
    try:
        model_path = os.path.join(parent_dir, 'models')
        logger.info(f"Looking for models in {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model directory does not exist: {model_path}")
            raise FileNotFoundError(f"Model directory does not exist: {model_path}")
            
        model_files = glob.glob(os.path.join(model_path, 'UpscalingModel_*.pth'))
        logger.info(f"Found {len(model_files)} model files: {model_files}")
        
        if not model_files:
            raise FileNotFoundError("No trained models found in directory")
        
        # Extract epoch numbers and find the latest
        latest_epoch = max(int(f.split('_')[-1].split('.')[0]) for f in model_files)
        return os.path.join(model_path, f'UpscalingModel_{latest_epoch}.pth')
    except Exception as e:
        logger.error(f"Error finding model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def extract_patches(image, patch_size=128, stride=64):
    """Extract overlapping patches from an image"""
    patches = []
    positions = []
    
    width, height = image.size
    
    # If image is smaller than patch size, resize it
    if width < patch_size or height < patch_size:
        logger.warning(f"Image size {width}x{height} is smaller than patch size {patch_size}. Resizing.")
        ratio = patch_size / min(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.BICUBIC)
        width, height = image.size
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            positions.append((x, y))
    
    # Handle edge cases by adding patches for right and bottom edges
    if (height - patch_size) % stride != 0:
        for x in range(0, width - patch_size + 1, stride):
            y = height - patch_size
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            positions.append((x, y))
    
    if (width - patch_size) % stride != 0:
        for y in range(0, height - patch_size + 1, stride):
            x = width - patch_size
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            positions.append((x, y))
    
    # Add the bottom-right corner patch
    if (width - patch_size) % stride != 0 and (height - patch_size) % stride != 0:
        x = width - patch_size
        y = height - patch_size
        patch = image.crop((x, y, x + patch_size, y + patch_size))
        patches.append(patch)
        positions.append((x, y))
    
    return patches, positions, image

def reconstruct_image(patches, positions, original_size, scale_factor):
    """Reconstruct an image from processed patches"""
    width, height = original_size
    new_width = width * scale_factor
    new_height = height * scale_factor
    
    # Create a new image with the upscaled size
    result = Image.new('RGB', (new_width, new_height))
    
    # Place each patch in its correct position
    for patch, (x, y) in zip(patches, positions):
        result.paste(patch, (x * scale_factor, y * scale_factor))
    
    return result

def process(image, scale_factor=2.0, patch_size=128, stride=64):
    """
    Upscale an image using the trained SRCNN model
    
    Args:
        image (PIL.Image): Input image
        scale_factor (float): Scale factor (2.0 or 4.0)
        patch_size (int): Size of patches to process
        stride (int): Stride between patches
        
    Returns:
        PIL.Image: Upscaled image
    """
    try:
        logger.info(f"Starting upscaling process. Image size: {image.size}, Scale factor: {scale_factor}")
        
        # If scale factor is a string, convert to float
        if isinstance(scale_factor, str):
            scale_factor = float(scale_factor)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load the latest model
        model_path = get_latest_model()
        logger.info(f"Loading model from: {model_path}")
        
        model = SRCNN(scale_factor=int(scale_factor)).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Transform for image normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Check image size and extract patches
        logger.info(f"Extracting patches with size {patch_size} and stride {stride}")
        patches, positions, resized_image = extract_patches(image, patch_size, stride)
        logger.info(f"Extracted {len(patches)} patches")
        
        # Process each patch
        processed_patches = []
        for i, patch in enumerate(patches):
            try:
                logger.debug(f"Processing patch {i+1}/{len(patches)}")
                
                # Convert patch to tensor
                img_tensor = transform(patch).unsqueeze(0).to(device)
                
                # Process patch
                with torch.no_grad():
                    output = model(img_tensor)
                
                # Convert back to PIL Image
                output = output.squeeze(0).cpu()
                output = output * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                output = output + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                output = torch.clamp(output, 0, 1)
                output = transforms.ToPILImage()(output)
                
                processed_patches.append(output)
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error processing patch {i+1}: {str(e)}")
                logger.error(traceback.format_exc())
                # For failed patches, use a bicubic upscaled version as fallback
                fallback = patch.resize((patch.width * int(scale_factor), patch.height * int(scale_factor)), Image.BICUBIC)
                processed_patches.append(fallback)
        
        # Reconstruct the image
        logger.info(f"Reconstructing image from {len(processed_patches)} processed patches")
        result = reconstruct_image(processed_patches, positions, resized_image.size, int(scale_factor))
        
        logger.info(f"Upscaling complete. Result size: {result.size}")
        return result
    
    except Exception as e:
        logger.error(f"Error in upscaling process: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Fallback to bicubic upscaling
        logger.info("Falling back to bicubic upscaling")
        width, height = image.size
        new_width = int(width * float(scale_factor))
        new_height = int(height * float(scale_factor))
        return image.resize((new_width, new_height), Image.BICUBIC) 