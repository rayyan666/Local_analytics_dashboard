# backend/fastapi_app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil, os, uuid, json
from pathlib import Path
from .llm_adapters.llama_cpp_adapter import LlamaAdapter
from .executors.sandbox_executor import execute_python_safely, execute_sql_safely
from .reports.report_generator import make_pdf, decode_base64_png
from .utils.db_connectors import sqlite_info
from .utils.code_validator import CodeValidator
from .utils.data_profiler import deep_data_profile

app = FastAPI(title="Local Analytic Chatbot", docs_url="/api/docs")

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MODELS_DIR = BASE / "models"
STATIC_DIR = BASE / "static"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Simple in-memory DB registry
DB_REGISTRY = {}

# Simple cache for uploaded file data (filename -> dataframe)
FILE_CACHE = {}
MAX_CACHE_SIZE = 5  # Keep last 5 files in memory

DEFAULT_MODEL = str(MODELS_DIR / "ggml-mistral-7b-instruct-q4.gguf")
llm = LlamaAdapter(model_path=DEFAULT_MODEL)

@app.get("/", include_in_schema=False)
def root():
    """Serve the frontend HTML at root"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")

    return RedirectResponse(url="/api/docs")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    import pandas as pd
    uid = str(uuid.uuid4())
    filename = f"{uid}_{file.filename}"
    out_path = UPLOAD_DIR / filename
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    profile = None
    import numpy as np
    def clean_floats(obj):
        if isinstance(obj, dict):
            return {k: clean_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_floats(x) for x in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        return obj
    if filename.lower().endswith('.csv'):
        try:
            df = pd.read_csv(out_path, nrows=1000)
            profile = deep_data_profile(df)
            profile = clean_floats(profile)
        except Exception as e:
            profile = {"error": f"Could not profile CSV: {str(e)}"}
    return {"file_path": str(out_path), "profile": profile}

@app.post("/register_sqlite")
def register_sqlite(path: str = Form(...), alias: str = Form(...)):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="path not found")
    DB_REGISTRY[alias] = sqlite_info(path)
    return {"status": "ok", "alias": alias}

class ChatRequest(BaseModel):
    user_prompt: str
    source_type: str  # "db" or "file"
    source_alias: str = None
    file_path: str = None
    file_paths: list = None  # Multiple files for analysis
    return_pdf: bool = False

# Enhanced system prompt for professional data analysis and summaries
ENHANCED_SYSTEM_PROMPT = r"""
You are a professional Data Analyst and Business Analyst providing expert insights.

RESPONSE FOR SUMMARY/ANALYSIS QUESTIONS:
- Provide 2-4 sentences with concrete metrics and insights
- Include key findings, anomalies, and business implications
- Format: [PROFESSIONAL ANALYSIS]

EXAMPLES:
Q: "What's in this data?"
A: Dataset contains 5,420 user records across 15 attributes. Key demographics show 42% of users aged 25-35, 
   68% concentrated in urban areas. Data quality is excellent with only 0.3% missing values and no duplicates detected.

Q: "Any anomalies?"
A: Identified 3 data quality issues: 3 users with age > 120 (likely data entry errors), 12 records with missing 
   emails (0.2%), and 1 city with extreme activity spike in Nov 2024. Recommend flagging age > 100 for review.

Q: "Summarize the dataset"
A: This customer dataset spans 18 months with 12,500 transactions. Revenue averages $450/transaction with 35% repeat 
   customers. Geographic distribution shows strong presence in North America (72%) and Europe (20%). No significant 
   seasonal patterns detected.
""".strip()

# Use a raw string for system prompt to avoid unicodeescape problems when the file contains backslashes
SYSTEM_PROMPT = r"""
You are a data analysis expert. Your job is to generate ONE-LINE Python code for analysis and visualization, strictly following the rules below.

The input may include a line like:
Available columns: ['user_id', 'gender', 'location_city', ...]
These are the ONLY valid columns you may use after normalization.

========================================
CRITICAL GLOBAL RULES
========================================
1. ALWAYS load the file first:
   df = pd.read_csv(FILE_PATH)

2. ALWAYS normalize column names immediately after loading:
   df.columns = df.columns.str.lower().str.strip()

3. AFTER normalization, use ONLY lowercase column names.
   - Column 'Location_CITY' becomes 'location_city'
   - NEVER use the original casing in code.
   - NEVER invent new columns.

4. Code MUST be ONE LINE ONLY, using semicolons to separate statements.

5. Code MUST end with:
   RESULT = plt     (for charts)
   RESULT = df      (for data tables)

6. Do NOT write imports. The following are already available:
   pd, np, plt, FILE_PATH, CHART_PATH

7. For ANY visualization:
   - ALWAYS call plt.savefig(CHART_PATH)
   - ALWAYS call plt.tight_layout() before saving.

8. For speed:
   - ALWAYS limit data using df.head(100) OR df.sample(100)
   - NEVER operate on the full dataframe when it’s not necessary.

========================================
COLUMN & SCHEMA AWARENESS
========================================
9. You MUST use ONLY the actual dataset columns listed under:
   "Available columns: [...]"
   - First, mentally normalize them to lowercase and strip spaces.
   - Use these normalized names in the code.

10. When the user mentions a column name:
    - Normalize it (lowercase, strip spaces).
    - If the normalized name is present in the available columns, use it.

11. If the user’s requested column is NOT in the available columns:
    - Try to find the closest existing column by name similarity
      (e.g. 'genre' ~ 'gender', 'cityname' ~ 'city', etc.).
    - If such a close match exists, USE THAT COLUMN and explicitly mention
      this assumption in your ONE-SENTENCE explanation, e.g.:
      "Using 'gender' as the closest match to the requested 'genre'."

12. If there is NO reasonable column match:
    - DO NOT write code that accesses a non-existent column.
    - INSTEAD, safely return a small preview of the dataframe:
      df = pd.read_csv(FILE_PATH); df.columns = df.columns.str.lower().str.strip(); RESULT = df.head(5)
    - In your explanation sentence, clearly say that the requested column
      was not found and you are returning a preview so the user can check
      the exact column names.

========================================
GROUPBY, COUNTS & AVOIDING ERRORS
========================================
13. NEVER invent synthetic columns like "user_count", "count", "value", etc.
    - For counts of rows, use:
        df['some_column'].value_counts()
      or:
        df.groupby('some_column').size()

14. NEVER access groupby columns using attribute style:
    - WRONG:
        df.groupby('location_city').user_count
    - RIGHT:
        df.groupby('location_city').size()
      or:
        df.groupby('location_city')['existing_numeric_column'].sum()

15. Do NOT chain non-existent columns after groupby. You MUST only use
    columns that actually exist in the dataset.

========================================
VISUALIZATION TEMPLATES
========================================
Use these templates as patterns. Adapt to the actual column name but keep the structure.
CRITICAL: All code must be on ONE line with semicolons. No line breaks.

PIE CHART:
df = pd.read_csv(FILE_PATH); df.columns = df.columns.str.lower().str.strip(); g = df['col'].value_counts().head(10); plt.figure(figsize=(8,8)); plt.pie(g.values, labels=g.index, autopct='%1.0f%%'); plt.title('Chart'); plt.savefig(CHART_PATH); RESULT = plt

BAR CHART:
df = pd.read_csv(FILE_PATH); df.columns = df.columns.str.lower().str.strip(); g = df['col'].value_counts().head(20); plt.figure(figsize=(12,5)); plt.bar(g.index, g.values); plt.xticks(rotation=45); plt.title('Chart'); plt.savefig(CHART_PATH); RESULT = plt

SUMMARY TABLE:
df = pd.read_csv(FILE_PATH); df.columns = df.columns.str.lower().str.strip(); RESULT = df.describe().head(10)

Replace 'col' with the actual column name from available columns list.

========================================
SMART INTERPRETATION OF USER REQUESTS
========================================
16. If the user asks about:
    - "distribution", "how many", "number of users", "count"
    then:
    - Interpret this as counting rows per category of a valid column.
    - Use value_counts() or groupby().size(), NOT a fabricated column.

17. If the user’s intent is ambiguous (e.g. just "show distribution of users"):
    - Choose a categorical column with relatively few unique values
      (e.g. gender, category, city) based on the available columns.
    - Explain in your ONE sentence which column you chose and why.

========================================
RESPONSE FORMAT (VERY IMPORTANT)
========================================
Output EXACTLY these two lines (no more, no less):

[ONE SHORT SENTENCE]
<CODE>[ENTIRE CODE AS ONE SINGLE LINE WITH SEMICOLONS]</CODE>

CRITICAL REQUIREMENTS:
✅ MUST be ONE SINGLE LINE inside <CODE>...</CODE>
✅ MUST have semicolons between statements
✅ MUST end with RESULT = plt or RESULT = df
✅ NO line breaks, NO indentation, NO newlines
✅ NO markdown, NO backticks outside <CODE> tags
✅ NO "if", loops, functions, or complex statements

WORKING EXAMPLE for "plot users per category":
plotting distribution of users by category.
<CODE>df = pd.read_csv(FILE_PATH); df.columns = df.columns.str.lower().str.strip(); g = df['category_user'].value_counts().head(20); plt.figure(figsize=(12,5)); plt.bar(g.index, g.values); plt.xticks(rotation=45); plt.title('Users by Category'); plt.savefig(CHART_PATH); RESULT = plt</CODE>
""".strip()


def build_multi_file_context(file_paths):
    """Build context from multiple files for analysis"""
    import pandas as pd
    import re
    context = []
    file_summaries = {}
    
    for file_path in file_paths:
        try:
            path_obj = Path(file_path) if file_path.startswith('/') else (BASE / file_path)
            if not path_obj.exists():
                continue
                
            df = pd.read_csv(path_obj, nrows=5000)
            df.columns = df.columns.str.lower().str.strip()
            
            # Store for potential use
            if file_path not in FILE_CACHE or len(FILE_CACHE) < MAX_CACHE_SIZE:
                FILE_CACHE[file_path] = df
            
            filename = re.sub(r'^[a-f0-9\-]+_', '', path_obj.name)
            summary = {
                'file': filename,
                'rows': len(df),
                'columns': list(df.columns),
                'shape': f"{len(df)} rows × {len(df.columns)} columns",
                'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                'missing': int((df.isnull().sum() > 0).sum())
            }
            file_summaries[filename] = summary
            col_list = ', '.join(summary['columns'][:10])
            if len(summary['columns']) > 10:
                col_list += f", ... (+{len(summary['columns']) - 10} more)"
            context.append(f"File: {filename} - {summary['shape']}, Columns: {col_list}")
        except Exception as e:
            context.append(f"File {file_path}: Error reading - {str(e)}")
    
    return "\n".join(context), file_summaries


@app.post("/analyze")
def analyze_files(req: ChatRequest):
    """Enhanced endpoint: analyze multiple files, provide summaries and insights"""
    import pandas as pd
    import re
    
    # Determine which files to analyze
    files_to_analyze = req.file_paths if req.file_paths else ([req.file_path] if req.file_path else [])
    
    if not files_to_analyze:
        return JSONResponse({"error": "No files provided"}, status_code=400)
    
    # Build context from files
    file_context, summaries = build_multi_file_context(files_to_analyze)
    
    # Check if this is a summary/analysis question or a visualization question
    prompt_lower = req.user_prompt.lower()
    needs_code = any(word in prompt_lower for word in ['chart', 'plot', 'graph', 'show', 'visual', 'visualize', 'draw'])
    
    # Use appropriate system prompt
    if needs_code:
        system_prompt = SYSTEM_PROMPT
    else:
        system_prompt = ENHANCED_SYSTEM_PROMPT
    
    # Build the full prompt
    full_prompt = (
        system_prompt + "\n\n" +
        "FILES AVAILABLE:\n" + file_context + "\n\n" +
        f"User Question: {req.user_prompt}\n" +
        "Provide professional data analysis with insights:\n"
    )
    
    try:
        resp = llm.generate(
            full_prompt,
            max_tokens=512,
            temperature=0.1,
            top_p=0.85,
            top_k=40,
            repeat_penalty=1.1
        )
    except Exception as e:
        return JSONResponse({
            "error": "LLM generation error",
            "detail": str(e),
            "message": "Failed to analyze data. Please try again."
        }, status_code=500)
    
    # Extract response
    raw_text = None
    if isinstance(resp, dict):
        try:
            if "choices" in resp and len(resp["choices"]) > 0:
                choice = resp["choices"][0]
                raw_text = choice.get("text") or (choice.get("message") or {}).get("content")
            else:
                raw_text = resp.get("text") or resp.get("generated_text") or resp.get("output")
        except Exception:
            raw_text = str(resp)
    if raw_text is None:
        raw_text = str(resp)
    
    raw_text = raw_text.strip() if raw_text else ""
    
    # Check if it's a code response
    code_match = re.search(r'<CODE>(.*?)(?:</CODE>|$)', raw_text, re.DOTALL)
    
    if code_match and needs_code:
        # Execute code for visualization
        code = code_match.group(1).strip().replace('\n', ' ')
        code = ' '.join(code.split())
        response_text = re.sub(r'<CODE>.*?(?:</CODE>|$)', '', raw_text, flags=re.DOTALL).strip()
        
        # Use first file for execution
        file_path = files_to_analyze[0]
        
        # Validate and execute
        is_valid, error_msg = CodeValidator.validate(code)
        if not is_valid:
            return JSONResponse({
                "error": "Code validation failed",
                "detail": error_msg,
                "message": f"Invalid code: {error_msg}"
            }, status_code=400)
        
        try:
            result_data = execute_python_safely(code, file_path=file_path)
            if not result_data.get("ok"):
                return JSONResponse({
                    "error": "Code execution error",
                    "detail": result_data.get("error", "Unknown error"),
                    "message": response_text if response_text else "Error executing analysis"
                }, status_code=500)
            
            return {
                "message": response_text or "Analysis complete",
                "result": result_data,
                "summaries": summaries
            }
        except Exception as e:
            return JSONResponse({
                "error": "Execution failed",
                "detail": str(e),
                "message": response_text if response_text else "An error occurred"
            }, status_code=500)
    else:
        # Return analysis/summary directly
        return {
            "message": raw_text,
            "result": None,
            "summaries": summaries,
            "is_summary": True
        }


@app.post("/chat")
def chat(req: ChatRequest):
    # Get column info from cached file if available - pre-normalize columns
    columns_info = ""
    if req.file_path in FILE_CACHE:
        try:
            cached_df = FILE_CACHE[req.file_path]
            # Pre-normalize to lowercase for better matching
            col_list = sorted([c.lower().strip() for c in cached_df.columns])
            columns_info = f"\nAvailable columns in data: {col_list}\n"
        except Exception:
            pass
    
    # Build prompt with better formatting
    prompt = (
        SYSTEM_PROMPT + 
        columns_info + 
        "\n" +
        f"User question: {req.user_prompt}\n" +
        f"File path: {req.file_path}\n" +
        "Generate code now:\n"
    )
    try:
        # Optimized for speed AND precision
        # Increased tokens to allow full code generation
        resp = llm.generate(
            prompt,
            max_tokens=512,      # Increased to allow full template generation
            temperature=0.0,     # Fully deterministic
            top_p=0.85,          # Tighter nucleus sampling
            top_k=40,            # Better precision
            repeat_penalty=1.1,  # Reduce repetition
            stop=None            # Don't use stop tokens - let full code generate
        )
    except Exception as e:
        return JSONResponse({
            "error": "LLM generation error",
            "detail": str(e),
            "message": "Failed to generate code. Please try again with a simpler query."
        }, status_code=500)

    # Extract text from response with better handling
    raw_text = None
    if isinstance(resp, dict):
        try:
            # Multiple extraction methods
            if "choices" in resp and len(resp["choices"]) > 0:
                choice = resp["choices"][0]
                raw_text = choice.get("text") or (choice.get("message") or {}).get("content")
            else:
                raw_text = resp.get("text") or resp.get("generated_text") or resp.get("output")
        except Exception:
            raw_text = str(resp)
    if raw_text is None or raw_text == "":
        raw_text = str(resp)

    raw_text = raw_text.strip() if raw_text else ""
    
    # DEBUG: Log raw response length
    import sys
    print(f"DEBUG: raw_text length={len(raw_text) if raw_text else 0}, ends_with_close_tag={raw_text.endswith('</CODE>') if raw_text else False}", file=sys.stderr)
    
    if not raw_text:
        return JSONResponse({
            "error": "Empty LLM response",
            "detail": "The language model returned no output",
            "message": "Please try again or check your data"
        }, status_code=500)
    
    # Extract code from <CODE>...</CODE> tags
    import re
    code_match = re.search(r'<CODE>(.*?)(?:</CODE>|$)', raw_text, re.DOTALL)
    
    if code_match:
        # Extract code and execute it
        code = code_match.group(1).strip()
        
        # Remove any newlines that might have been added by LLM - must be single line
        code = code.replace('\n', ' ')
        # Clean up extra spaces
        code = ' '.join(code.split())
        
        # Remove the <CODE> tags from response text
        response_text = re.sub(r'<CODE>.*?(?:</CODE>|$)', '', raw_text, flags=re.DOTALL).strip()
        
        file_path = req.file_path
        
        # DEBUG: Log the code before validation
        import sys
        print(f"DEBUG: Generated code: {repr(code[:300])}", file=sys.stderr)
        
        # Validate the generated code before execution
        is_valid, error_msg = CodeValidator.validate(code)
        if not is_valid:
            return JSONResponse({
                "error": "Code validation failed",
                "detail": error_msg,
                "llm_raw": raw_text[:200],
                "message": f"Invalid code: {error_msg}"
            }, status_code=400)
        
        try:
            result_data = execute_python_safely(code, file_path=file_path)
            
            # Check if execution had an error
            if not result_data.get("ok"):
                error_detail = result_data.get("error", "Unknown error")
                # Provide helpful suggestions for common errors
                helpful_msg = error_detail
                if "KeyError" in error_detail:
                    helpful_msg = f"{error_detail}\n\nTIP: Column may not exist. Try asking 'What columns do I have?'"
                elif "timeout" in error_detail.lower() or "killed" in error_detail.lower():
                    helpful_msg = f"{error_detail}\n\nTIP: Limit data size with .head(100) or use sampling for faster processing"
                
                return JSONResponse({
                    "error": "Code execution error",
                    "detail": helpful_msg,
                    "message": response_text if response_text else "Error executing analysis code"
                }, status_code=500)
                
        except Exception as exec_err:
            import traceback
            return JSONResponse({
                "error": "Code execution failed",
                "detail": str(exec_err),
                "traceback": traceback.format_exc()[:500],
                "message": response_text if response_text else "An error occurred while processing your request"
            }, status_code=500)
        
        # If chart present and return_pdf requested, create PDF
        pdf_path = None
        if req.return_pdf and result_data.get("result") and result_data["result"].get("chart_png_base64"):
            try:
                b64 = result_data["result"]["chart_png_base64"]
                img_bytes = decode_base64_png(b64)
                pdf_name = f"report_{uuid.uuid4().hex}.pdf"
                pdf_path = DATA_DIR / pdf_name
                make_pdf("Report", result_data["result"].get("records", []), img_bytes, str(pdf_path))
            except Exception as e:
                pass  # Non-critical error, continue without PDF

        return {
            "message": response_text if response_text else "Analysis complete",
            "result": result_data,
            "pdf_path": str(pdf_path) if pdf_path else None
        }
    else:
        # No code tags - return natural language response
        return {
            "message": raw_text,
            "result": None
        }


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    """Serve the dashboard page"""
    dashboard_path = STATIC_DIR / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path), media_type="text/html")
    raise HTTPException(status_code=404, detail="Dashboard not found")


@app.get("/list-files")
def list_files():
    """List all uploaded CSV files"""
    if not UPLOAD_DIR.exists():
        return []
    
    files = []
    for f in UPLOAD_DIR.glob("*_*.csv"):
        files.append(f"data/uploads/{f.name}")
    
    return sorted(files, reverse=True)


@app.get("/load-data")
def load_data(file: str):
    """Load CSV file and return as JSON"""
    import pandas as pd
    
    try:
        # Handle both absolute and relative paths
        if file.startswith("/"):
            # Absolute path - use directly but verify it's within UPLOAD_DIR
            file_path = Path(file)
        else:
            # Relative path - resolve from base
            file_path = BASE / file
        
        # Security: verify the file is within UPLOAD_DIR
        try:
            file_path.resolve().relative_to(UPLOAD_DIR.resolve())
        except ValueError:
            raise ValueError("Invalid file path - must be in uploads directory")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Read CSV
        df = pd.read_csv(str(file_path))
        
        # Normalize columns
        df.columns = df.columns.str.lower().str.strip()
        
        # Convert to records, handling special types
        records = []
        for _, row in df.iterrows():
            record = {}
            for col, val in row.items():
                if pd.isna(val):
                    record[col] = None
                elif isinstance(val, (int, float)):
                    record[col] = float(val) if isinstance(val, float) else int(val)
                else:
                    record[col] = str(val)
            records.append(record)
        
        return records
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


# Serve frontend static files under /static
app.mount("/static", StaticFiles(directory=str(BASE / "static"), html=True), name="static")
