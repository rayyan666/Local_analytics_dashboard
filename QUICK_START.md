# 🚀 Quick Start - Launch in 5 Minutes

## Prerequisites
- Python 3.8+
- Git
- ~2GB disk space for Mistral model (optional)

## Step 1: Clone & Setup (1 min)

```bash
# Navigate to where you want the project
cd ~/Documents

# Clone the repository
git clone <repo-url>
cd local_analytic_chatbot

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Step 2: Install Dependencies (2 min)

```bash
# Install required packages
pip install fastapi uvicorn pandas numpy matplotlib llama-cpp-python

# Or install from requirements.txt if available
pip install -r requirements.txt
```

## Step 3: Prepare Model (Optional, 1 min)

The app works without a local model, but for best results:

```bash
# The model should already be in: models/ggml-mistral-7b-instruct-q4.gguf
# If not present, the app will use fallback mode

# To download manually (skip if already present):
# Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-GGUF
# Place in: models/ directory
```

## Step 4: Start Backend (1 min)

```bash
# From the project root directory
cd backend

# Start the server
python3 -m uvicorn fastapi_app:app --reload

# You should see:
# INFO:     Uvicorn running on http://127.0.0.1:8000
# INFO:     Application startup complete
```

## Step 5: Open Frontend (1 min)

Open your web browser and navigate to:
```
http://localhost:8000
```

That's it! 🎉

---

## What You Can Do Right Away

### 1. Upload a CSV File
- Click **"Upload CSV"** button
- Select any CSV file from your computer
- See automatic data profiling with quality warnings

### 2. Ask Questions
- Type your question in the text box
- Example: "What's the average price?" or "Show me a distribution"
- Get instant analysis with charts

### 3. Use Query History
- Click **"📜 Query History"** to see all previous queries
- 🔄 Click **Rerun** to instantly re-execute
- ⭐ Click **Star** to bookmark favorites
- 🗑️ Click **Delete** to remove

### 4. View Smart Errors
- If you make a mistake, get helpful suggestions
- Suggestions include:
  - What went wrong (❌)
  - How to fix it (💡)
  - Example code (📝)

### 5. Check Cache Stats
- Open new tab to: `http://localhost:8000/cache-stats`
- See performance metrics:
  - Cache hit rate
  - Number of cached queries
  - Performance improvement

---

## Troubleshooting

### Issue: "Port 8000 already in use"
```bash
# Use different port
python3 -m uvicorn fastapi_app:app --port 8001 --reload
# Then visit http://localhost:8001
```

### Issue: "Module not found" error
```bash
# Make sure you're in the right directory
cd local_analytic_chatbot/backend

# Install missing package
pip install [missing-package-name]
```

### Issue: "Model file not found"
```bash
# The app will work without the model
# For local LLM features, download and place:
# models/ggml-mistral-7b-instruct-q4.gguf

# Or run in fallback mode (uses template generation)
```

### Issue: "Can't upload CSV"
```bash
# Make sure data/uploads directory exists
mkdir -p data/uploads

# Check permissions
chmod 755 data/uploads
```

---

## Feature Quick Reference

| Feature | What It Does | Access |
|---------|-------------|--------|
| **Query History** | Save/rerun 50 queries | 📜 Button in sidebar |
| **Smart Errors** | Fix error messages with suggestions | Auto-displayed on error |
| **Query Caching** | 40-100x speedup for repeats | Automatic |
| **Data Validation** | Warn about quality issues | Shows in profile |
| **Cache Stats** | View performance metrics | /cache-stats endpoint |

---

## Next Steps

### For Development
1. Read `IMPLEMENTATION_GUIDE.md` for full details
2. Check `IMPROVEMENTS_SUMMARY.md` for feature overview
3. See individual docs: `ERROR_IMPROVEMENTS.md`, `CACHING_LAYER.md`, `DATA_VALIDATION.md`

### For Testing
1. Upload a CSV with some issues
2. Ask multiple questions
3. Verify caching works (same query runs fast 2nd time)
4. Try error scenarios (misspelled column names)
5. Check query history persistence

### For Deployment
1. Stop the dev server (Ctrl+C)
2. Run without `--reload` for production:
   ```bash
   python3 -m uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
   ```
3. Consider using Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 fastapi_app:app
   ```

---

## Key Directories

```
local_analytic_chatbot/
├── backend/                    # Python backend
│   ├── fastapi_app.py         # Main API server
│   ├── utils/
│   │   ├── error_suggester.py # Error analysis (NEW)
│   │   ├── query_cache.py     # Query caching (NEW)
│   │   ├── data_profiler.py   # Data validation (ENHANCED)
│   │   └── ...
│   └── ...
├── static/                     # Frontend
│   ├── index.html             # Main UI (ENHANCED)
│   └── dashboard.html
├── data/
│   └── uploads/               # Uploaded CSV files
├── models/                     # LLM models
│   └── ggml-mistral-7b-instruct-q4.gguf (optional)
└── README.md                   # Full documentation
```

---

## API Endpoints

All available at `http://localhost:8000`:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Main UI |
| `/upload` | POST | Upload CSV |
| `/chat` | POST | Ask question |
| `/cache-stats` | GET | View cache metrics |
| `/cache-clear` | POST | Clear cache |
| `/api/docs` | GET | Swagger docs |

---

## Performance Tips

1. **First query**: Takes 2-5 seconds (normal)
2. **Repeat queries**: Takes 5-50ms (cached!)
3. **Batch operations**: Process multiple files separately
4. **Memory**: Cache limited to 100 entries (~5MB)

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4 cores |
| RAM | 2GB | 8GB |
| Disk | 100MB | 2GB+ (with model) |
| Python | 3.8 | 3.10+ |
| OS | Any | Linux/Mac |

---

## Example Queries

Try these to test the system:

```
1. "What columns do I have?"
   → Shows available columns

2. "How many rows are there?"
   → Returns row count

3. "What's the average price?"
   → Calculates statistics

4. "Show me a distribution"
   → Creates visualization

5. "Any missing values?"
   → Reports data quality

6. "Summarize this data"
   → Professional summary
```

---

## Success Checklist

- [ ] Backend started without errors
- [ ] Frontend loaded at localhost:8000
- [ ] Can upload a CSV file
- [ ] Can ask a question and get results
- [ ] Query History button works
- [ ] Rerun functionality works
- [ ] Cache stats show metrics
- [ ] Error messages are helpful

---

## Getting Help

1. **Error message unclear?**
   - Check `ERROR_IMPROVEMENTS.md` for pattern meanings

2. **Want to understand caching?**
   - Read `CACHING_LAYER.md`

3. **Curious about data validation?**
   - See `DATA_VALIDATION.md`

4. **Need full API reference?**
   - Visit `/api/docs` in browser
   - Or read `IMPLEMENTATION_GUIDE.md`

5. **Code not working?**
   - Check `CHANGES_SUMMARY.md` for what was modified
   - All files are syntax-validated ✅

---

## That's It! 🎉

You now have a fully functional analytics chatbot with:
- ✅ Query history
- ✅ Smart error messages
- ✅ Query caching (40-100x speedup)
- ✅ Data validation
- ✅ Professional UI

**Enjoy!** 🚀

---

*For more details, see the full documentation in the project root.*
