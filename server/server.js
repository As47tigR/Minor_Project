const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');

// Create Express app
const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_BACKEND_URL = 'http://localhost:5000';

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// API Routes
app.post('/api/process-image', async (req, res) => {
  try {
    const { image, processType, params } = req.body;

    if (!image || !processType) {
      return res.status(400).json({ error: 'Image and process type are required' });
    }

    // Forward the request to Python backend
    const response = await axios.post(`${PYTHON_BACKEND_URL}/process`, {
      image: image,
      process_type: processType,
      params: params || {}
    });

    return res.json(response.data);
  } catch (error) {
    console.error('Error processing image:', error.message);
    return res.status(500).json({ 
      error: 'Error processing image',
      details: error.response ? error.response.data : error.message
    });
  }
});

// Serve the main HTML file for any other route
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Access the application at http://localhost:${PORT}`);
}); 