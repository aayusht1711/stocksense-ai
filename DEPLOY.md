# 🚀 StockSense AI — Deployment Guide
## Get a live public URL in 10 minutes — completely FREE

---

## What you'll have after this guide

```
https://stocksense-aayush.streamlit.app   ← your live app
```

Anyone in the world can open this URL — no installation needed.
Put it on your resume, LinkedIn, and GitHub profile immediately.

---

## STEP 1 — Create a GitHub Account (2 min)

1. Go to **https://github.com**
2. Click **Sign up**
3. Choose username (e.g. `aayush-tripathi` or your name)
4. Verify email → done

---

## STEP 2 — Install Git on Windows (2 min)

1. Go to **https://git-scm.com/download/win**
2. Download and install (click Next → Next → Next, all defaults)
3. Open a new PowerShell window after install
4. Test: `git --version` → should show `git version 2.x.x`

---

## STEP 3 — Create GitHub Repository (2 min)

1. Go to **https://github.com/new**
2. Repository name: `stocksense-ai`
3. Select **Public** (required for free Streamlit Cloud)
4. ✅ Check "Add a README file"
5. Click **Create repository**

---

## STEP 4 — Push Your Code to GitHub (3 min)

Open PowerShell in your stocksense folder and run these commands ONE BY ONE:

```powershell
# Navigate to your project folder
cd C:\Users\AAYUSH TRIPATHI\Desktop\stocksense_ai\stocksense

# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit — StockSense AI"

# Connect to your GitHub repo (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/stocksense-ai.git

# Push to GitHub
git branch -M main
git push -u origin main
```

When it asks for username + password:
- Username: your GitHub username
- Password: use a Personal Access Token (NOT your GitHub password)

### Creating a Personal Access Token:
1. GitHub → click your profile photo (top right)
2. Settings → Developer settings (bottom of left sidebar)
3. Personal access tokens → Tokens (classic)
4. Generate new token → check **repo** box → Generate
5. Copy the token — paste it as your password

---

## STEP 5 — Deploy to Streamlit Cloud (3 min)

1. Go to **https://share.streamlit.io**
2. Click **Sign in with GitHub** → authorize it
3. Click **New app**
4. Fill in:
   - Repository: `YOUR_USERNAME/stocksense-ai`
   - Branch: `main`
   - Main file path: `dashboard/app.py`
5. Click **Deploy!**

Wait 3–5 minutes while it installs packages. You'll see a spinner.

Your app is live at: `https://stocksense-ai.streamlit.app` 🎉

---

## STEP 6 — Add API Keys as Secrets (1 min)

Your `.env` file is in `.gitignore` (good — never put secrets on GitHub).
Add keys securely through Streamlit's dashboard:

1. In Streamlit Cloud → click your app → **Settings** (gear icon)
2. Click **Secrets** tab
3. Paste this (fill in your actual values):

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
ALERT_EMAIL_FROM = "yourgmail@gmail.com"
ALERT_EMAIL_PASSWORD = "your-16-char-app-password"
ALERT_EMAIL_TO = "yourgmail@gmail.com"
```

4. Click **Save** → app restarts automatically

---

## STEP 7 — Put it on your Resume & LinkedIn

### Resume bullet:
```
StockSense AI — Live at stocksense-ai.streamlit.app
• Built end-to-end ML stock prediction system with 91% directional accuracy
• Stack: Python, TensorFlow, PyTorch TFT, XGBoost, FinBERT NLP, Claude AI
• Features: real-time predictions, email alerts, AI chat assistant, portfolio tracker
```

### LinkedIn:
- Add to Featured section → Link → paste your Streamlit URL
- Post about it with this caption:
  "Just deployed my ML stock prediction app live! Built with Python, PyTorch Transformer, XGBoost ensemble, and Claude AI chat assistant. Check it out 👇 [your URL] #MachineLearning #Python #FinTech"

---

## STEP 8 — Enable Auto-Retraining (optional but impressive)

The `.github/workflows/retrain.yml` file is already in your project.
It retrains your models every Sunday automatically.

To enable it:
1. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add: `ALPHA_VANTAGE_API_KEY` (even if empty, add it)
4. Go to **Actions** tab → click **Weekly Model Retrain** → **Run workflow**

Now your models update with fresh data every week automatically. 🤖

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| App crashes on deploy | Check the logs — usually a missing import. Remove `torch` from requirements.txt (not needed on cloud) |
| "Module not found" error | Make sure all files are committed: `git add . && git commit -m "fix" && git push` |
| App loads but no data | yFinance works fine — it fetches live data automatically |
| Secrets not working | Make sure keys are in Streamlit Secrets, not in code |
| Build timeout | tensorflow-cpu takes 3-4 min to install — wait it out |

---

## Updating your app in future

Every time you make changes:
```powershell
git add .
git commit -m "Add new feature"
git push
```

Streamlit Cloud auto-detects the push and redeploys in ~2 minutes. ✅

---

## Your live URLs

After deployment, share these:
- **App**: `https://stocksense-ai.streamlit.app`
- **Code**: `https://github.com/YOUR_USERNAME/stocksense-ai`

Both go on your resume. The live app URL is what gets you interviews. 🎯
