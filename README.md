# AI Image Processing & Super-Resolution Project

A comprehensive image processing application with AI-powered super-resolution capabilities.

## Features

- **Multiple Image Processing Options**:
  - AI Image Upscaling using SRCNN (2x and 4x)
  - Traditional scaling methods (bicubic, bilinear, etc.)
  - Grayscale conversion
  - Blur effects
  - Edge detection
  - Thresholding

- **Interactive UI**:
  - Drag-and-drop image upload
  - Real-time parameter adjustment
  - Before/After comparison with interactive overlay slider
  - Image download functionality
  - Fullscreen view

## System Architecture

### Frontend
- Modern HTML5, CSS3, and JavaScript
- Responsive design for desktop and mobile devices
- Interactive image comparison tools

### Backend
- Node.js Express server for frontend hosting and API routes
- Python Flask server for image processing operations
- PyTorch-based SRCNN model for AI image upscaling

## Setup Instructions

### Prerequisites

- Node.js (v14+)
- Python (3.7+)
- PyTorch
- CUDA-compatible GPU (recommended for faster processing)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ai-image-processing.git
   cd ai-image-processing
   ```

2. **Install Node.js dependencies**:
   ```bash
   cd server
   npm install
   ```

3. **Install Python dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Create required directories**:
   ```bash
   mkdir -p backend/metrics
   mkdir -p backend/test_images
   ```

### Running the Application

1. **Start the Python backend server** (in one terminal):
   ```bash
   cd backend
   python main.py
   ```

2. **Start the Node.js frontend server** (in another terminal):
   ```bash
   cd server
   npm start
   ```

3. **Access the application**:
   Open your browser and navigate to `http://localhost:3000`

## Using the Model

### Training the AI Upscaling Model

1. **Prepare the dataset**:
   - Place your low-resolution images in `backend/Dataset/DIV2K_train_LR`
   - Place your high-resolution images in `backend/Dataset/DIV2K_train_HR`

2. **Start training**:
   ```bash
   cd backend
   python training/train_srcnn.py
   ```

3. **Monitor training**:
   - Checkpoints will be saved to `backend/models/UpscalingModel_*.pth`
   - Metrics will be saved to `backend/metrics/`

### Testing the Trained Model

1. **Place test images in** `backend/test_images/`

2. **Run the test script**:
   ```bash
   cd backend
   python test_model.py
   ```

3. **View results in** `backend/test_images/results/`

## Production Deployment

For production deployment, the following steps are recommended:

1. **Set up environment variables**:
   Create a `.env` file in both `server/` and `backend/` directories with appropriate settings for production.

2. **Add SSL certificates**:
   Configure SSL for secure connections.

3. **Set up a reverse proxy**:
   Use Nginx or similar to route traffic to both servers.

4. **Configure PM2 or similar**:
   For process management and auto-restart capabilities.

## Technical Implementation Details

### SRCNN Architecture

The super-resolution model uses SRCNN (Super-Resolution Convolutional Neural Network) architecture with three convolutional layers:

1. **Feature extraction layer**: 9x9 kernels, 64 filters
2. **Non-linear mapping layer**: 1x1 kernels, 32 filters
3. **Reconstruction layer**: 5x5 kernels, 3 filters

### Image Processing Pipeline

1. **Upload**: Image is uploaded via frontend and sent to Node.js server
2. **Processing Request**: Node.js forwards the request to Python server
3. **AI Processing**: Python server processes the image using the appropriate method
4. **Response**: Processed image is returned to frontend for display

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DIV2K dataset for training the super-resolution model
- SRCNN paper: "Image Super-Resolution Using Deep Convolutional Networks" 