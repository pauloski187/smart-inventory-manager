#!/bin/bash

# HF Spaces Deployment Script
# This script pushes your FastAPI backend to Hugging Face Spaces

echo "🚀 Deploying to Hugging Face Spaces..."

# Set HF Space repository URL
HF_SPACE_URL="https://huggingface.co/spaces/pauloski07/sales-inventory-forecasting"

# Check if we're in the right directory
if [ ! -f "app.py" ] || [ ! -f "Dockerfile" ]; then
    echo "❌ Error: Must run from huggingface-spaces directory"
    echo "   Run: cd huggingface-spaces && bash deploy.sh"
    exit 1
fi

# Initialize git if not already
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    git remote add origin $HF_SPACE_URL
fi

# Add files
echo "📝 Adding files..."
git add app.py requirements.txt README.md Dockerfile sample_data.csv

# Check if there are changes
if git diff --cached --quiet; then
    echo "ℹ️  No changes to commit"
else
    # Commit
    echo "💾 Committing changes..."
    git commit -m "Deploy FastAPI backend with Docker SDK"

    # Push to HF Space
    echo "⬆️  Pushing to Hugging Face Spaces..."
    echo "   (You may need to enter your HF token)"
    git push -u origin main --force

    echo ""
    echo "✅ Deployment complete!"
    echo ""
    echo "📍 Your Space: https://huggingface.co/spaces/pauloski07/sales-inventory-forecasting"
    echo "🔗 API URL: https://pauloski07-sales-inventory-forecasting.hf.space"
    echo ""
    echo "⏳ Wait 2-5 minutes for rebuild, then test:"
    echo "   https://pauloski07-sales-inventory-forecasting.hf.space/health"
fi
