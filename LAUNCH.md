# ⚡ LAUNCH IN 5 MINUTES

## Just 5 Commands

```bash
# 1️⃣ Clone and enter
git clone <repo-url>
cd local_analytic_chatbot

# 2️⃣ Install dependencies
pip install fastapi uvicorn pandas numpy matplotlib requests

# 3️⃣ Download the AI model (4.3GB)
# See "📥 Download Model" section below ⬇️

# 4️⃣ Launch backend
cd backend
python3 -m uvicorn fastapi_app:app --reload

# 5️⃣ Open browser
# Visit: http://localhost:8000
```

---

## 📥 Download Model (IMPORTANT!)

### What Model?
The app uses **Mistral 7B** - a powerful open-source AI model (~4.3GB)
- No internet required after download
- Runs locally on your machine
- No API costs
- Privacy-first (all data stays local)

### Step 1: Create models directory
```bash
mkdir -p models
cd models
```

### Step 2: Download the model file

**Option A: Using curl (Recommended)**
```bash
# Download Mistral 7B Instruct Q4 (~4.3GB)
curl -L https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf -o ggml-mistral-7b-instruct-q4.gguf

# This will take 5-15 minutes depending on internet speed
```

**Option B: Manual Download (if curl fails)**
1. Visit: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
2. Click the file: `mistral-7b-instruct-v0.1.Q4_K_M.gguf`
3. Click download (⬇️ button)
4. Save to: `models/ggml-mistral-7b-instruct-q4.gguf`

**Option C: Using git-lfs (if you have git-lfs installed)**
```bash
# Install git-lfs first: https://git-lfs.github.com
git lfs clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
```

### Step 3: Verify download
```bash
ls -lh models/ggml-mistral-7b-instruct-q4.gguf

# Should show ~4.3GB file
# Example output:
# -rw-r--r--  1 user  staff  4.3G Dec  8 10:30 ggml-mistral-7b-instruct-q4.gguf
```

### Step 4: Back to root and launch
```bash
cd ..
cd backend
python3 -m uvicorn fastapi_app:app --reload
```

---

### ⏱️ Download Time Guide

| Connection Speed | Estimated Time |
|------------------|-----------------|
| 100 Mbps | 5-8 minutes |
| 50 Mbps | 10-15 minutes |
| 25 Mbps | 20-30 minutes |
| 10 Mbps | 1+ hours |

**💡 Pro Tip:** Start download, go get coffee! ☕

---

### 🆘 If Download Fails

**Problem: Connection timeout**
```bash
# Try with a longer timeout and resume
wget --timeout=0 --continue \
  https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
  -O ggml-mistral-7b-instruct-q4.gguf
```

**Problem: Partial download**
```bash
# Resume download
curl -C - -o ggml-mistral-7b-instruct-q4.gguf \
  https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

**Problem: Disk space error**
```bash
# Check available space
df -h

# You need at least 5GB free
# Delete old files or upgrade storage
```

**Problem: Model file not found at startup**
- Check file exists: `ls models/ggml-mistral-7b-instruct-q4.gguf`
- Check file size: Should be ~4.3GB
- Check path: File must be in `models/` folder
- Check permissions: `chmod 644 models/ggml-mistral-7b-instruct-q4.gguf`

---

### 📊 Alternative Models (if you want different size)

**If you have less disk space:**
```bash
# Smaller model (3.3GB) - slightly less capable
# Q5_K_M variant (5.2GB) - higher quality
# Q3_K_M variant (2.5GB) - minimum quality
```

Visit: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF

Pick any `.gguf` file and download it. The app will work with any Mistral 7B variant.

---

### ✅ Complete Checklist

- [ ] Created `models/` directory
- [ ] Downloaded model file (4.3GB)
- [ ] File is named `ggml-mistral-7b-instruct-q4.gguf`
- [ ] File is in `models/` folder
- [ ] File size is ~4.3GB
- [ ] Ready to launch!

---

## Quick Launch After Model Download

```bash
# Everything is ready! Just run:
cd backend
python3 -m uvicorn fastapi_app:app --reload

# Then open: http://localhost:8000
```

---

## What You Get

| Feature | Speed | What It Does |
|---------|-------|-------------|
| 📜 **Query History** | Auto | Save 50 queries, click rerun |
| 💡 **Smart Errors** | Auto | Fixes shown, not confusing |
| ⚡ **Query Cache** | 40-100x | Repeat queries instant |
| ⚠️ **Data Warnings** | Auto | Know data issues upfront |
| 📊 **Profiling** | Auto | See all stats on upload |

---

## Example Usage

```
1. Upload CSV → See warnings
2. Ask "What's the average?" → Get answer
3. Ask same thing again → Instant (cached!)
4. Click Query History → Rerun old queries
5. Get error → See helpful suggestion
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Port 8000 busy | `--port 8001` |
| Missing pandas | `pip install pandas` |
| No model file | Works anyway (fallback mode) |
| Upload fails | `mkdir -p data/uploads` |

---

## Full Docs

- **Quick Start:** `QUICK_START.md` (this file with more details)
- **Features:** `IMPROVEMENTS_SUMMARY.md`
- **Deep Dive:** `IMPLEMENTATION_GUIDE.md`
- **API Docs:** `http://localhost:8000/api/docs`

---

**That's it! Start exploring your data.** 🎉
