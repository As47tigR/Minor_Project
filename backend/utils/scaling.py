import numpy as np
from PIL import Image

def process(image, scale_factor=1.0, method='nearest'):
    """
    Resize an image (up or down) using normal methods
    
    Args:
        image (PIL.Image): Input image
        scale_factor (float): Scale factor (0.5 = half size, 2.0 = double size)
        method (str): Resample method ('nearest', 'bilinear', 'bicubic', 'lanczos')
        
    Returns:
        PIL.Image: Resized image
    """
    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive")
    
    # Get original dimensions
    width, height = image.size
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Select resampling method
    resampling_methods = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    
    resampling = resampling_methods.get(method.lower(), Image.BICUBIC)
    
    # Resize image
    return image.resize((new_width, new_height), resampling) 