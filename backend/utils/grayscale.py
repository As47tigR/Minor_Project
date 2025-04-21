import numpy as np
from PIL import Image

def process(image):
    """
    Convert an image to grayscale
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Grayscale image
    """
    return image.convert('L') 