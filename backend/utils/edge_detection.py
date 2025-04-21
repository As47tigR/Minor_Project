import numpy as np
from PIL import Image, ImageFilter

def process(image):
    """
    Apply edge detection to an image
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        PIL.Image: Edge detected image
    """
    return image.filter(ImageFilter.FIND_EDGES) 