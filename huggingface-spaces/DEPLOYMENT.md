# Hugging Face Spaces Deployment Guide

## Prerequisites

1. A Hugging Face account (free): https://huggingface.co/join
2. Git installed on your machine
3. Hugging Face CLI (optional but recommended)

## Step 1: Create a New Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Owner**: Your username
   - **Space name**: `smart-inventory-forecaster`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU basic (free)
   - **Visibility**: Public (required for free tier)
3. Click "Create Space"

## Step 2: Clone the Space

```bash
# Install Hugging Face CLI if you haven't
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

# Clone your new empty Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/smart-inventory-forecaster
cd smart-inventory-forecaster
```

## Step 3: Copy Files

Copy all files from the `huggingface-spaces` folder:

```bash
# From the smart-inventory-manager directory
cp huggingface-spaces/* /path/to/smart-inventory-forecaster/
```

Or manually copy:
- `app.py`
- `requirements.txt`
- `README.md`
- `sample_data.csv` (optional - for testing)

## Step 4: Push to Hugging Face

```bash
cd smart-inventory-forecaster

# Add all files
git add .

# Commit
git commit -m "Initial deployment: SARIMA + Prophet forecasting API"

# Push to Hugging Face
git push
```

## Step 5: Wait for Build

1. Go to your Space: `https://huggingface.co/spaces/YOUR_USERNAME/smart-inventory-forecaster`
2. Click on the "Logs" tab to monitor the build
3. The build typically takes 3-5 minutes (Prophet installation is slow)
4. Once complete, the app will be live!

## Step 6: Test the API

### Using the Web Interface

1. Go to your Space URL
2. Upload the `sample_data.csv` file (or your own data)
3. Wait for models to train
4. Test forecasting for different categories

### Using the Gradio Client (Python)

```python
from gradio_client import Client

# Connect to your Space
client = Client("YOUR_USERNAME/smart-inventory-forecaster")

# Upload data and train models
result = client.predict(
    file="path/to/your/data.csv",
    api_name="/load_dataset"
)
print(result)

# Get forecast
forecast = client.predict(
    category="Electronics",
    weeks=13,
    api_name="/get_forecast"
)
print(forecast)

# Get inventory recommendations
recommendations = client.predict(
    api_name="/get_recommendations"
)
print(recommendations)
```

### Using REST API (curl)

```bash
# Get the API endpoint from your Space's "Use via API" button

# Example: Upload file
curl -X POST "https://YOUR_USERNAME-smart-inventory-forecaster.hf.space/api/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data.csv"
```

## Connecting to Your Frontend (Lovable AI)

In your Lovable AI frontend, configure the API base URL:

```javascript
// config.js
const API_BASE_URL = 'https://YOUR_USERNAME-smart-inventory-forecaster.hf.space';

// Example: Get forecast
async function getForecast(category, weeks) {
  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      data: [category, weeks],
      fn_index: 1  // Index of get_forecast function
    })
  });
  return response.json();
}
```

Or use the Gradio JavaScript client:

```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("YOUR_USERNAME/smart-inventory-forecaster");

// Get forecast
const result = await client.predict("/get_forecast", {
  category: "Electronics",
  weeks: 13
});
console.log(result.data);
```

## Handling Cold Starts

HF Spaces free tier goes to sleep after 48 hours of inactivity. First request after sleep takes ~30 seconds.

To minimize impact:
1. Add a health check endpoint (already included)
2. Use loading states in your frontend
3. Consider upgrading to persistent hardware ($5/month) for always-on

## Troubleshooting

### Build Failed
- Check the Logs tab for error details
- Common issues:
  - `prophet` may need `pystan` - already handled in requirements
  - Memory errors - reduce dataset size

### Slow Performance
- First load trains models - subsequent requests are faster
- Consider pre-training and saving model pickle files

### API Not Responding
- Check if Space is "Running" (not "Sleeping")
- Visit the Space URL to wake it up
- Check Logs for errors

## Files Created

```
huggingface-spaces/
├── app.py              # Gradio app with forecasting logic
├── requirements.txt    # Python dependencies
├── README.md           # HF Spaces metadata and documentation
├── sample_data.csv     # 5000-row sample for testing
└── DEPLOYMENT.md       # This file
```

## Updating the Space

To update your deployed Space:

```bash
cd smart-inventory-forecaster

# Make changes to files
# ...

# Commit and push
git add .
git commit -m "Update: description of changes"
git push
```

The Space will automatically rebuild and redeploy.

---

**Your Space URL will be:**
`https://huggingface.co/spaces/YOUR_USERNAME/smart-inventory-forecaster`

**Your API endpoint will be:**
`https://YOUR_USERNAME-smart-inventory-forecaster.hf.space`
