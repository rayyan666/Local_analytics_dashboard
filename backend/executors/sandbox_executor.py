# backend/executors/sandbox_executor.py

import os
import io
import base64
import tempfile
import traceback
import shutil
import multiprocessing as mp
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt


def _encode_chart_to_base64(chart_path: str) -> Optional[str]:
    """Read a PNG file and return base64 string, or None if not present."""
    if not os.path.exists(chart_path):
        return None
    with open(chart_path, "rb") as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode("utf-8")


# Module-level function for multiprocessing (must be picklable)
def _target_run(code: str, file_path: str, chart_path: str, q: mp.Queue):
    """Execute code in isolated process."""
    try:
        # Set resource limits (soft limits - may not work on all platforms)
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))  # 2GB
            resource.setrlimit(resource.RLIMIT_CPU, (25, 25))  # 25s
        except Exception:
            pass  # Platform may not support resource limits

        # Import required modules
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tempfile

        # Create temp directory for chart output
        temp_chart_dir = tempfile.gettempdir()
        chart_path_full = os.path.join(temp_chart_dir, "chart.png")

        # Provide a constrained global namespace for exec
        safe_globals = {
            "__name__": "__main__",
            "pd": pd,
            "np": np,
            "plt": plt,
            "FILE_PATH": file_path,
            "CHART_PATH": chart_path_full,  # Provide absolute path
        }

        # Execute the code
        try:
            exec(code, safe_globals)
        except SyntaxError as se:
            q.put({
                "ok": False,
                "error": f"Syntax error in code: {str(se)}\nCode: {code[:150]}..."
            })
            return
        except TimeoutError:
            q.put({
                "ok": False,
                "error": "Code execution timeout (>15s). Try using .head(50) or .sample(50) to limit data."
            })
            return
        except Exception as e:
            q.put({
                "ok": False,
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[:500]}"
            })
            return

        result = safe_globals.get("RESULT", None)
        out = {}

        # Check if result is a dataframe
        if hasattr(result, "to_dict"):
            out["type"] = "dataframe"
            out["records"] = result.head(100).to_dict(orient="records")
            out["columns"] = list(result.columns)
        # Check if it's a matplotlib figure or pyplot
        elif result is plt or (hasattr(result, 'savefig')):
            out["type"] = "plot"
            out["value"] = "matplotlib figure"
        else:
            out["type"] = "raw"
            out["value"] = str(result)

        # Check for chart.png in temp directory (matplotlib output)
        if os.path.exists(chart_path_full):
            try:
                with open(chart_path_full, "rb") as f:
                    out["chart_png_base64"] = base64.b64encode(f.read()).decode("ascii")
                os.remove(chart_path_full)  # Clean up immediately
            except Exception:
                pass  # Silently skip if chart read fails
        
        q.put({"ok": True, "result": out})
    except Exception as e:
        q.put({
            "ok": False,
            "error": f"Unexpected error: {traceback.format_exc()[:500]}"
        })


def execute_python_safely(code: str, file_path: str) -> Dict[str, Any]:
    """
    Execute LLM-generated Python code in a controlled multiprocess environment.
    Timeout: 15 seconds (reduced from 20 for better UX)
    Memory limit: 2GB
    CPU limit: 25 seconds
    
    Expected behavior:
      - Code uses: FILE_PATH and CHART_PATH
      - Code ends with: RESULT = plt (for plots) or RESULT = df (for data)
      - For plots: code calls plt.savefig(CHART_PATH)

    Return structure:
      {
        "ok": True/False,
        "result": {
          "records": [...],              # optional (from DataFrame)
          "dataframe_json": {...},       # optional
          "chart_png_base64": "..."      # optional
        },
        "error": "..."                   # if ok == False
      }
    """
    
    # Temporary directory to hold the chart file
    tmp_dir = tempfile.mkdtemp(prefix="la_chart_")
    chart_path = os.path.join(tmp_dir, "chart.png")
    
    # Create queue for result passing
    q: mp.Queue = mp.Queue()

    # Run in separate process with timeout
    try:
        process = mp.Process(target=_target_run, args=(code, file_path, chart_path, q))
        process.start()
        process.join(timeout=15)  # Reduced from 20 to 15 seconds
        
        if process.is_alive():
            # Timeout occurred
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()  # Force kill if still alive
            return {
                "ok": False,
                "error": "Code execution timeout (>15s). Try limiting data with .head(50) or .sample(50).",
                "result": None
            }
        
        # Get result from queue
        if not q.empty():
            return q.get()
        else:
            return {
                "ok": False,
                "error": "No result returned from code execution",
                "result": None
            }
    except Exception as e:
        return {
            "ok": False,
            "error": f"Process error: {str(e)}",
            "result": None
        }
    finally:
        # Cleanup - don't return anything here, just clean up resources
        try:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
        except Exception:
            pass


# Optional: if you use SQL in other parts, keep a stub here
def execute_sql_safely(*args, **kwargs) -> Dict[str, Any]:
    """
    Placeholder for SQL execution if used elsewhere.
    Implement similar contract:
      { "ok": True/False, "result": {...}, "error": "..." }
    """
    return {
        "ok": False,
        "error": "execute_sql_safely not implemented",
        "result": None,
    }
