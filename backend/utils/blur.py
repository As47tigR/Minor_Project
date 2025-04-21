import numpy as np
from PIL import Image, ImageFilter

def process(image, radius=2):
    """
    Apply Gaussian blur to an image
    
    Args:
        image (PIL.Image): Input image
        radius (int, optional): Blur radius. Defaults to 2.
        
    Returns:
        PIL.Image: Blurred image
    """
    return image.filter(ImageFilter.GaussianBlur(radius=radius)) 