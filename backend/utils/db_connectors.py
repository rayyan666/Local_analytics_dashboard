# backend/utils/db_connectors.py
# Small helper module for DB connection metadata
import os
from typing import Dict

def sqlite_info(path: str) -> Dict[str, str]:
    return {"type": "sqlite", "path": path}
