document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const imageUpload = document.getElementById('image-upload');
  const uploadPlaceholder = document.getElementById('upload-placeholder');
  const previewImage = document.getElementById('preview-image');
  const processButton = document.getElementById('process-button');
  const originalContainer = document.getElementById('original-container');
  const processedContainer = document.getElementById('processed-container');
  const comparisonContainer = document.getElementById('comparison-container');
  const imageActions = document.querySelector('.image-actions');
  const downloadBtn = document.getElementById('download-btn');
  const fullscreenBtn = document.getElementById('fullscreen-btn');
  const fullscreenModal = document.getElementById('fullscreen-modal');
  const fullscreenImage = document.getElementById('fullscreen-image');
  const closeModal = document.querySelector('.close-modal');
  
  // Parameter control elements
  const blurControls = document.getElementById('blur-controls');
  const blurStrength = document.getElementById('blur-strength');
  const blurStrengthValue = document.getElementById('blur-strength-value');
  const scalingControls = document.getElementById('scaling-controls');
  const scaleFactor = document.getElementById('scale-factor');
  const scaleFactorValue = document.getElementById('scale-factor-value');
  const scalingMethod = document.getElementById('scaling-method');
  const aiUpscaleControls = document.getElementById('ai-upscale-controls');
  
  // Comparison slider elements
  const comparisonSlider = document.getElementById('comparison-slider');
  const comparisonOriginal = document.getElementById('comparison-original');
  const comparisonProcessed = document.getElementById('comparison-processed');
  const sliderValueLabel = document.querySelector('.slider-value');
  
  let originalImageData = null;
  let processedImageData = null;
  
  // Handle process type selection
  document.querySelectorAll('input[name="process-type"]').forEach(radio => {
    radio.addEventListener('change', () => {
      const processType = radio.value;
      
      // Hide all parameter controls
      blurControls.style.display = 'none';
      scalingControls.style.display = 'none';
      aiUpscaleControls.style.display = 'none';
      
      // Show relevant controls based on selected process type
      if (processType === 'blur') {
        blurControls.style.display = 'block';
      } else if (processType === 'scaling') {
        scalingControls.style.display = 'block';
      } else if (processType === 'ai_upscale') {
        aiUpscaleControls.style.display = 'block';
      }
    });
  });
  
  // Update slider value displays
  blurStrength.addEventListener('input', () => {
    blurStrengthValue.textContent = blurStrength.value;
  });
  
  scaleFactor.addEventListener('input', () => {
    scaleFactorValue.textContent = scaleFactor.value;
  });
  
  // Comparison slider functionality
  comparisonSlider.addEventListener('input', (e) => {
    const value = e.target.value;
    comparisonProcessed.style.opacity = value / 100;
    sliderValueLabel.textContent = `${value}%`;
  });
  
  // Handle image upload
  imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    
    if (file && file.type.match('image.*')) {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        // Store the base64 image data
        originalImageData = e.target.result;
        
        // Show preview
        previewImage.src = originalImageData;
        previewImage.style.display = 'block';
        uploadPlaceholder.style.display = 'none';
        
        // Display original image
        originalContainer.innerHTML = '';
        const originalImg = document.createElement('img');
        originalImg.src = originalImageData;
        originalContainer.appendChild(originalImg);
        
        // Enable process button
        processButton.disabled = false;
        
        // Clear processed image and hide actions
        processedContainer.innerHTML = '';
        comparisonContainer.style.display = 'none';
        imageActions.style.display = 'none';
      };
      
      reader.readAsDataURL(file);
    }
  });
  
  // Handle image processing
  processButton.addEventListener('click', async () => {
    if (!originalImageData) return;
    
    // Show loading state
    processButton.disabled = true;
    processButton.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Processing...';
    
    try {
      // Get selected process type
      const processType = document.querySelector('input[name="process-type"]:checked').value;
      
      // Prepare parameters based on process type
      const params = {};
      
      if (processType === 'blur') {
        params.blur_strength = blurStrength.value;
      } else if (processType === 'scaling') {
        params.scale_factor = scaleFactor.value;
        params.method = scalingMethod.value;
      } else if (processType === 'ai_upscale') {
        params.scale_factor = document.querySelector('input[name="ai-scale-factor"]:checked').value;
      }
      
      // Send image to server for processing
      const response = await fetch('/api/process-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          image: originalImageData,
          processType: processType,
          params: params
        })
      });
      
      if (!response.ok) {
        throw new Error(`Error processing image: ${response.statusText}`);
      }
      
      const data = await response.json();
      processedImageData = data.processed_image;
      
      // Display processed image
      processedContainer.innerHTML = '';
      const processedImg = document.createElement('img');
      processedImg.src = processedImageData;
      processedContainer.appendChild(processedImg);
      
      // Setup comparison slider
      setupComparisonSlider(originalImageData, processedImageData);
      comparisonContainer.style.display = 'block';
      
      // Show image actions
      imageActions.style.display = 'flex';
      
      // Display notification
      showNotification('Image processed successfully', 'success');
      
    } catch (error) {
      console.error('Error:', error);
      showNotification(error.message || 'An error occurred while processing the image.', 'error');
    } finally {
      // Reset button state
      processButton.disabled = false;
      processButton.innerHTML = 'Process Image';
    }
  });
  
  // Setup comparison slider
  function setupComparisonSlider(beforeImage, afterImage) {
    // Set the images
    comparisonOriginal.src = beforeImage;
    comparisonProcessed.src = afterImage;
    
    // Reset slider position
    comparisonSlider.value = 50;
    comparisonProcessed.style.opacity = 0.5;
    sliderValueLabel.textContent = '50%';
    
    // Scroll to comparison
    setTimeout(() => {
      comparisonContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 300);
  }
  
  // Show notification
  function showNotification(message, type = 'info') {
    // Check if notification container exists, if not create it
    let notificationContainer = document.querySelector('.notification-container');
    if (!notificationContainer) {
      notificationContainer = document.createElement('div');
      notificationContainer.className = 'notification-container';
      document.body.appendChild(notificationContainer);
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    // Set icon based on type
    let icon = '';
    switch (type) {
      case 'success':
        icon = '<i class="fa fa-check-circle"></i>';
        break;
      case 'error':
        icon = '<i class="fa fa-exclamation-circle"></i>';
        break;
      default:
        icon = '<i class="fa fa-info-circle"></i>';
    }
    
    // Set content
    notification.innerHTML = `
      ${icon}
      <p>${message}</p>
      <button class="close-notification">&times;</button>
    `;
    
    // Add to container
    notificationContainer.appendChild(notification);
    
    // Setup close button
    notification.querySelector('.close-notification').addEventListener('click', () => {
      notification.classList.add('fade-out');
      setTimeout(() => {
        notification.remove();
      }, 300);
    });
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.classList.add('fade-out');
        setTimeout(() => {
          if (notification.parentNode) {
            notification.remove();
          }
        }, 300);
      }
    }, 5000);
  }
  
  // Handle download button
  downloadBtn.addEventListener('click', () => {
    if (!processedImageData) return;
    
    const link = document.createElement('a');
    link.href = processedImageData;
    link.download = 'processed_image.jpg';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showNotification('Image downloaded successfully', 'success');
  });
  
  // Handle fullscreen button
  fullscreenBtn.addEventListener('click', () => {
    if (!processedImageData) return;
    
    fullscreenImage.src = processedImageData;
    fullscreenModal.style.display = 'block';
    document.body.style.overflow = 'hidden';
  });
  
  // Close fullscreen modal
  closeModal.addEventListener('click', () => {
    fullscreenModal.style.display = 'none';
    document.body.style.overflow = 'auto';
  });
  
  // Close modal when clicking outside the image
  fullscreenModal.addEventListener('click', (e) => {
    if (e.target === fullscreenModal) {
      fullscreenModal.style.display = 'none';
      document.body.style.overflow = 'auto';
    }
  });
  
  // Allow drag and drop for image upload
  const uploadLabel = document.querySelector('.upload-label');
  
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadLabel.addEventListener(eventName, preventDefaults, false);
  });
  
  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }
  
  ['dragenter', 'dragover'].forEach(eventName => {
    uploadLabel.addEventListener(eventName, highlight, false);
  });
  
  ['dragleave', 'drop'].forEach(eventName => {
    uploadLabel.addEventListener(eventName, unhighlight, false);
  });
  
  function highlight() {
    uploadLabel.style.borderColor = '#0275d8';
    uploadLabel.classList.add('highlight');
  }
  
  function unhighlight() {
    uploadLabel.style.borderColor = '#ccc';
    uploadLabel.classList.remove('highlight');
  }
  
  uploadLabel.addEventListener('drop', handleDrop, false);
  
  function handleDrop(e) {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    
    if (file && file.type.match('image.*')) {
      imageUpload.files = dt.files;
      
      // Trigger change event manually
      const event = new Event('change');
      imageUpload.dispatchEvent(event);
    }
  }
}); 