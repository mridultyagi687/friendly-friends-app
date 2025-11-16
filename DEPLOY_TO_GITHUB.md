# 🚀 Deploy to GitHub Pages - Complete Guide

## Step 1: Deploy Backend First (Required!)

The backend needs to be deployed separately since GitHub Pages only hosts static files.

### Option A: Deploy to Render (Recommended - Free)

1. **Go to [render.com](https://render.com)** and sign up with GitHub
2. **Click "New +" → "Web Service"**
3. **Connect your GitHub repository**
4. **Configure:**
   - **Name:** `friendly-friends-backend`
   - **Root Directory:** `backend`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
   - **Instance Type:** Free
5. **Add Environment Variables:**
   - `FLASK_SECRET_KEY` = (generate with: `openssl rand -hex 32`)
   - `OPENAI_API_KEY` = (your OpenAI API key)
   - `FLASK_ENV` = `production`
6. **Click "Create Web Service"**
7. **Wait for deployment** (takes 5-10 minutes)
8. **Copy your backend URL:** `https://friendly-friends-backend.onrender.com`

### Option B: Deploy to Railway

```bash
cd backend
railway login
railway init
railway up
railway variables set FLASK_SECRET_KEY=$(openssl rand -hex 32)
railway variables set OPENAI_API_KEY=your_key
railway domain
```

---

## Step 2: Push Code to GitHub

```bash
# Make sure you're in the project root
cd "/Users/mridul/Documents/Friendly Friends App"

# Add all files
git add .

# Commit
git commit -m "Initial commit - Ready for deployment"

# Create repository on GitHub (go to github.com and create new repo)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

**OR** if you already have a remote:
```bash
git push origin main
```

---

## Step 3: Enable GitHub Pages

1. **Go to your GitHub repository**
2. **Click "Settings" → "Pages"**
3. **Source:** Select "GitHub Actions"
4. **Save**

---

## Step 4: Set Backend URL Secret (Optional)

If your backend URL is different from the default:

1. **Go to repository Settings → Secrets and variables → Actions**
2. **Click "New repository secret"**
3. **Name:** `VITE_API_URL`
4. **Value:** Your backend URL (e.g., `https://friendly-friends-backend.onrender.com`)
5. **Click "Add secret"**

---

## Step 5: Trigger Deployment

The deployment will automatically trigger when you push to `main`.

**OR** manually trigger:
1. Go to **Actions** tab
2. Click **"Deploy to GitHub Pages"**
3. Click **"Run workflow"**

---

## Step 6: Access Your App

After deployment completes (5-10 minutes):

**Your app will be live at:**
```
https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/
```

**Example:**
```
https://mridul.github.io/friendly-friends-app/
```

---

## Troubleshooting

### Build Fails
- Check Actions tab for error logs
- Make sure `frontend/package.json` has correct dependencies
- Verify Node.js version in workflow (currently 18)

### Backend Not Connecting
- Check backend URL in GitHub Secrets
- Verify backend is running and accessible
- Check CORS settings in backend

### 404 Errors
- Make sure GitHub Pages source is set to "GitHub Actions"
- Check that `vite.config.js` has correct base path
- Verify build completed successfully

---

## Quick Deploy Script

Run this to deploy everything:

```bash
# 1. Deploy backend to Render (manual - follow Step 1)
# 2. Then run:
cd "/Users/mridul/Documents/Friendly Friends App"
git add .
git commit -m "Deploy to GitHub Pages"
git push origin main
```

Your app will be live in ~10 minutes! 🎉

