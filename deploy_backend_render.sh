#!/bin/bash

# Quick script to help deploy backend to Render
# This creates a render.yaml file for easy deployment

echo "🚀 Creating Render deployment configuration..."
echo ""

cat > render.yaml << 'EOF'
services:
  - type: web
    name: friendly-friends-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: FLASK_ENV
        value: production
      - key: FLASK_SECRET_KEY
        generateValue: true
      - key: OPENAI_API_KEY
        sync: false  # Set this in Render dashboard
EOF

echo "✅ Created render.yaml"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://render.com"
echo "2. Sign up/login with GitHub"
echo "3. Click 'New +' → 'Blueprint'"
echo "4. Connect your GitHub repo: mridul6275-blip/friendly-friends-app"
echo "5. Select render.yaml"
echo "6. Add OPENAI_API_KEY in Environment Variables"
echo "7. Click 'Apply'"
echo ""
echo "Your backend will be deployed to:"
echo "https://friendly-friends-backend.onrender.com"
echo ""
echo "⏱️  Deployment takes 5-10 minutes"

