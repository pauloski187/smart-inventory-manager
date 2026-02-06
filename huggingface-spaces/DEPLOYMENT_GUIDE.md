# Hugging Face Spaces Deployment Guide

## Step-by-Step Deployment Instructions

### Prerequisites

- Hugging Face account (free)
- Git installed locally
- Access to your HF Space: `pauloski07/sales-inventory-forecasting`

### Option 1: Upload via Hugging Face Web Interface (Easiest)

1. **Go to your Space**: https://huggingface.co/spaces/pauloski07/sales-inventory-forecasting

2. **Click on "Files" tab**

3. **Upload the following files** (one by one or drag-and-drop):
   - `app.py` (the new FastAPI version)
   - `requirements.txt` (the new lightweight dependencies)
   - `README.md` (the new API documentation)
   - `Dockerfile` (for Docker-based deployment)
   - `sample_data.csv` (sample data file)

4. **Commit the changes** with a message like: "Deploy Full FastAPI Backend"

5. **Wait for rebuild**: HF Spaces will automatically rebuild your app (takes 2-5 minutes)

6. **Check deployment**: Visit your Space URL once the build completes

### Option 2: Git Push (For Developers)

1. **Clone your Space repository**:
   ```bash
   git clone https://huggingface.co/spaces/pauloski07/sales-inventory-forecasting
   cd sales-inventory-forecasting
   ```

2. **Copy the new files** from `huggingface-spaces/` folder:
   ```bash
   # From your project root
   cp huggingface-spaces/app.py sales-inventory-forecasting/
   cp huggingface-spaces/requirements.txt sales-inventory-forecasting/
   cp huggingface-spaces/README.md sales-inventory-forecasting/
   cp huggingface-spaces/Dockerfile sales-inventory-forecasting/
   cp huggingface-spaces/sample_data.csv sales-inventory-forecasting/
   ```

3. **Commit and push**:
   ```bash
   cd sales-inventory-forecasting
   git add .
   git commit -m "Deploy Full FastAPI Backend for Lovable AI integration"
   git push
   ```

4. **Monitor deployment**: Go to your Space on HF to see build progress

### Option 3: Sync from Local Project (Recommended for Active Development)

If you're in the project directory already:

```bash
# Navigate to HF Spaces folder
cd huggingface-spaces

# Initialize git (if not already a repo)
git init
git remote add origin https://huggingface.co/spaces/pauloski07/sales-inventory-forecasting

# Add files
git add app.py requirements.txt README.md Dockerfile sample_data.csv

# Commit
git commit -m "Deploy Full FastAPI Backend"

# Push to HF Spaces
git push -u origin main --force
```

## Verification Steps

Once deployed, verify your API is working:

### 1. Check Health Endpoint

Visit: https://pauloski07-sales-inventory-forecasting.hf.space/health

Expected response:
```json
{
  "status": "healthy",
  "service": "smart-inventory-manager",
  "version": "1.0.0",
  "platform": "huggingface-spaces"
}
```

### 2. Check API Documentation

Visit: https://pauloski07-sales-inventory-forecasting.hf.space/docs

You should see interactive Swagger UI with all endpoints.

### 3. Test a Sample Endpoint

Visit: https://pauloski07-sales-inventory-forecasting.hf.space/analytics/dashboard/summary

You should get JSON response with dashboard metrics.

### 4. Test from Command Line

```bash
curl https://pauloski07-sales-inventory-forecasting.hf.space/health
```

## Troubleshooting

### Issue: Build Failed / Exit Code 137

**Problem**: Memory exhaustion during build

**Solution**:
- The new version is optimized for HF Spaces free tier
- Uses SQLite instead of PostgreSQL
- No heavy ML models loaded at startup
- If still fails, try upgrading to HF Spaces Pro (https://huggingface.co/pricing)

### Issue: API Returns 503 or Timeout

**Problem**: Space is sleeping (free tier behavior)

**Solution**:
- HF Spaces free tier sleeps after inactivity
- First request wakes it up (may take 30-60 seconds)
- Consider upgrading to persistent hardware if needed

### Issue: Database is Empty

**Problem**: Sample data not loading

**Solution**:
- Check that `sample_data.csv` is uploaded
- Check Space logs for errors
- The app will create minimal sample data if CSV not found

### Issue: CORS Errors from Frontend

**Problem**: Frontend can't access API

**Solution**:
- The API has CORS enabled for all origins
- Check that you're using HTTPS (not HTTP)
- Clear browser cache

## Space Configuration

Your Space uses these settings (in README.md front matter):

```yaml
---
title: Smart Inventory Manager API
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
license: mit
---
```

### To Change SDK from Gradio to Docker

If your Space is still configured for Gradio, update the README.md front matter:

**Change**:
```yaml
sdk: gradio
sdk_version: 4.44.0
```

**To**:
```yaml
sdk: docker
```

Then commit and push.

## Connecting to Lovable AI

Once your API is live:

1. **API Base URL**: Use `https://pauloski07-sales-inventory-forecasting.hf.space`

2. **In Lovable AI Project Settings**:
   - Set backend URL to the HF Space URL
   - No authentication required
   - All endpoints support CORS

3. **Example Frontend Code**:

```typescript
// config.ts
export const API_BASE_URL = 'https://pauloski07-sales-inventory-forecasting.hf.space';

// In your components
const response = await fetch(`${API_BASE_URL}/analytics/dashboard/summary`);
const data = await response.json();
```

## Performance Optimization

### For Free Tier

- API uses SQLite (lightweight)
- Sample data pre-loaded on startup
- No heavy ML models
- Optimized for low memory usage

### For Better Performance

Upgrade to HF Spaces Pro:
- Persistent hardware (no cold starts)
- More memory and CPU
- Custom domains
- Private Spaces option

Visit: https://huggingface.co/pricing

## Support

- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **API Documentation**: Visit `/docs` endpoint on your Space

## Next Steps

1. ✅ Deploy the files to HF Spaces
2. ✅ Verify API is working
3. ✅ Connect Lovable AI frontend
4. 🎨 Build your dashboard UI with Lovable AI
5. 🚀 Launch your inventory management system!

---

**Your Space URL**: https://huggingface.co/spaces/pauloski07/sales-inventory-forecasting

**Live API URL**: https://pauloski07-sales-inventory-forecasting.hf.space

**API Docs**: https://pauloski07-sales-inventory-forecasting.hf.space/docs
