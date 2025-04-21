from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import os
import importlib
import logging
import traceback
from utils import grayscale, blur, edge_detection, threshold, scaling, ai_upscale

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("flask_app")

app = Flask(__name__)
CORS(app)

# Available processing functions
PROCESSORS = {
    'grayscale': grayscale.process,
    'blur': blur.process,
    'edge_detection': edge_detection.process,
    'threshold': threshold.process,
    'scaling': scaling.process,
    'ai_upscale': ai_upscale.process
}

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'ok'})

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.json or 'process_type' not in request.json:
            logger.warning("Missing image or process_type parameter")
            return jsonify({'error': 'Missing image or process_type parameter'}), 400
        
        # Get image and process type
        image_data = request.json['image']
        process_type = request.json['process_type']
        logger.info(f"Processing image with {process_type}")
        
        # Get additional parameters if available
        params = request.json.get('params', {})
        
        # Check if process type is valid
        if process_type not in PROCESSORS:
            logger.warning(f"Invalid process_type: {process_type}")
            return jsonify({'error': f'Invalid process_type. Available types: {list(PROCESSORS.keys())}'}), 400
        
        try:
            # Decode base64 image
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            logger.info(f"Image decoded successfully. Size: {image.size}")
            
            # Process the image with additional parameters
            if process_type == 'blur':
                # Handle blur strength parameter
                blur_radius = float(params.get('blur_strength', 2))
                processed_image = PROCESSORS[process_type](image, radius=blur_radius)
            elif process_type == 'scaling':
                # Handle scaling parameters
                scale_factor = float(params.get('scale_factor', 1.0))
                method = params.get('method', 'bicubic')
                processed_image = PROCESSORS[process_type](image, scale_factor=scale_factor, method=method)
            elif process_type == 'ai_upscale':
                try:
                    # Handle AI upscaling parameters
                    scale_factor = float(params.get('scale_factor', 2.0))
                    logger.info(f"Starting AI upscaling with scale factor {scale_factor}")
                    
                    # Use smaller patch size and stride for large images
                    width, height = image.size
                    if width * height > 1000000:  # For images larger than ~1MP
                        logger.info("Large image detected, using smaller patch size")
                        patch_size = 64
                        stride = 32
                    else:
                        patch_size = 128
                        stride = 64
                        
                    processed_image = PROCESSORS[process_type](image, 
                                                              scale_factor=scale_factor,
                                                              patch_size=patch_size,
                                                              stride=stride)
                    logger.info("AI upscaling completed successfully")
                except Exception as e:
                    logger.error(f"Error in AI upscaling: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Fallback to bicubic scaling if AI upscaling fails
                    logger.info("Falling back to bicubic scaling")
                    width, height = image.size
                    new_width = int(width * float(scale_factor))
                    new_height = int(height * float(scale_factor))
                    processed_image = image.resize((new_width, new_height), Image.BICUBIC)
            else:
                # Use default processing
                processed_image = PROCESSORS[process_type](image)
            
            # Convert processed image back to base64
            buffered = io.BytesIO()
            processed_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return jsonify({
                'processed_image': f'data:image/jpeg;base64,{img_str}',
                'process_type': process_type,
                'original_size': {'width': image.size[0], 'height': image.size[1]},
                'processed_size': {'width': processed_image.size[0], 'height': processed_image.size[1]}
            })
        
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Starting Flask server")
    app.run(host='0.0.0.0', port=5000, debug=True) 