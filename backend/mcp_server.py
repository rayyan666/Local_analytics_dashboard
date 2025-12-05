# backend/mcp_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3, os, json
import pandas as pd
from typing import Any, Dict, Optional

app = FastAPI(title="MCP Server (local)", docs_url="/mcp/docs")

class SchemaRequest(BaseModel):
    db_type: str
    connection: Dict[str, Any]

class QueryRequest(BaseModel):
    db_type: str
    connection: Dict[str, Any]
    sql: str

class FileParseRequest(BaseModel):
    file_path: str
    format: Optional[str] = None

@app.get("/")
def root():
    return {"msg": "MCP server running. Use /tool/* endpoints."}

@app.post("/tool/schema")
def get_schema(req: SchemaRequest):
    if req.db_type == "sqlite":
        path = req.connection.get("path")
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=400, detail="sqlite path not found")
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()
        schema = {}
        for tname, sql in tables:
            cur2 = conn.execute(f"PRAGMA table_info('{tname}')").fetchall()
            schema[tname] = [{"cid": r[0], "name": r[1], "type": r[2]} for r in cur2]
        conn.close()
        return {"schema": schema}
    raise HTTPException(status_code=400, detail="Unsupported db_type")

@app.post("/tool/execute_sql")
def execute_sql(req: QueryRequest):
    if req.db_type == "sqlite":
        path = req.connection.get("path")
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=400, detail="sqlite path not found")
        conn = sqlite3.connect(path)
        try:
            cur = conn.execute(req.sql)
            cols = [c[0] for c in cur.description] if cur.description else []
            rows = cur.fetchmany(1000)
            conn.commit()
            # Convert rows to list (sqlite returns tuples)
            rows_list = [list(r) for r in rows]
            return {"columns": cols, "rows": rows_list}
        finally:
            conn.close()
    raise HTTPException(status_code=400, detail="Unsupported db_type")

@app.post("/tool/parse_file")
def parse_file(req: FileParseRequest):
    fp = req.file_path
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="file not found")
    fmt = req.format or (os.path.splitext(fp)[1].lstrip(".").lower())
    if fmt == "csv":
        df = pd.read_csv(fp, nrows=1000)
        return {"preview": df.head(20).to_dict(orient="records"), "columns": df.columns.tolist(), "rows": len(df)}
    elif fmt == "json":
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"preview": data if isinstance(data, (list,dict)) else str(data)}
    else:
        raise HTTPException(status_code=400, detail="unsupported file format")
