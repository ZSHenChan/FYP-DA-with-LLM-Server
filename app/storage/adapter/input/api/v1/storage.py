import os
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi import APIRouter, HTTPException

storage_router = APIRouter()

STORAGE_BASE_PATH = "session"

# DEBUGGG, route not registered

# @storage_router.get("/storage/{session_id}/{run_id}/{filename}")
# async def get_diagram(session_id: str, run_id: str, filename: str):
#     # Construct the path safely
#     file_path = os.path.join(STORAGE_BASE_PATH, session_id, run_id, "figures", filename)
    
#     # 1. Security Check: Prevent directory traversal (optional but recommended)
#     if not os.path.abspath(file_path).startswith(os.path.abspath(STORAGE_BASE_PATH)):
#         raise HTTPException(status_code=400, detail="Invalid path")

#     # 2. Check if file exists
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="Diagram not found")

#     return FileResponse(
#         file_path
#     )