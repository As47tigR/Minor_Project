import numpy as np
from PIL import Image

def process(image, threshold=128):
    """
    Apply binary thresholding to an image
    
    Args:
        image (PIL.Image): Input image
        threshold (int, optional): Threshold value. Defaults to 128.
        
    Returns:
        PIL.Image: Thresholded image
    """
    # Convert to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Apply thresholding
    return image.point(lambda p: 255 if p > threshold else 0) 