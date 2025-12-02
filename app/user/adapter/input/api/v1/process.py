import json, os
import asyncio
from asyncio import Queue
from core.config import config
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import APIRouter, HTTPException,  UploadFile, File, Form, Depends, Path, Request
from dependency_injector.wiring import Provide, inject
from app.services.agent import get_master_agent
from typing import Dict, Union, Optional, List
from app.services.agent.utils import SessionWorkspace

process_router = APIRouter()

SESSION_QUEUES: Dict[str, Queue] = {}

def generate_id(prefix: str | None = None) -> str:
    import uuid
    """
    Generate a shorter run ID using first 8 characters of UUID.
    
    Args:
        prefix: Prefix for the run ID (default: "run")
        
    Returns:
        Generated run ID string
        
    Example:
        - generate_run_id_short() -> "run_a1b2c3d4"
    """
    if not prefix:
        return uuid.uuid4().hex[:config.UUID_LEN]
    return f"{prefix}_{uuid.uuid4().hex[:config.UUID_LEN]}"

async def run_agent_work(
    master_agent, 
    human_input: str, 
    file_names: List[str],
    workspace: SessionWorkspace,
    q: Queue
):
    loop = asyncio.get_running_loop()

    def progress_callback(msg: Union[str, int]):
        data = json.dumps({"type": "progress", "message": str(msg)})
        loop.call_soon_threadsafe(q.put_nowait, data)

    async def heartbeat(queue: Queue):
        """Sends a ping every 20 seconds to keep the connection alive."""
        while True:
            await asyncio.sleep(20)
            ping_data = json.dumps({"type": "ping", "message": "still-processing"})
            try:
                # Use call_soon_threadsafe because this runs
                # in a different coroutine context
                loop.call_soon_threadsafe(queue.put_nowait, ping_data)
            except Exception as e:
                # If queue is closed, stop the heartbeat
                print(f"Heartbeat stopping: {e}")
                break
            
    heartbeat_task = asyncio.create_task(heartbeat(q))
            
    try:
        result = await asyncio.to_thread(
            master_agent.run_request,
            human_input,
            file_names,
            workspace,
            progress_callback
        )
        
        final_data = json.dumps({"type": "response", "message": result})
        await q.put(final_data)

    except Exception as e:
        error_data = json.dumps({"type": "error", "message": str(e)})
        await q.put(error_data)
    
    finally:
        heartbeat_task.cancel() # Stop the heartbeat task
        try:
            # Wait for heartbeat to fully cancel
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        await q.put("[DONE]")

@process_router.post("")
async def start_processing(
    prompt: str = Form(...),
    files: List[UploadFile] = Form(...)
):
    from pathlib import Path
    session_id = generate_id(prefix='sess')
    run_id = generate_id(prefix='run')
    
    q = Queue()
    if session_id in SESSION_QUEUES:
        raise HTTPException(status_code=500, detail="Session ID collision")
    
    SESSION_QUEUES[session_id] = q

    workspace = SessionWorkspace(session_id, run_id)
    dest_dir = workspace.data_dir
    file_names: List[str] = []

    if files and len(files) > 0:
        try:
            for file in files:
                filename = Path(file.filename).name
                file_path = dest_dir / filename
                
                file_content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(file_content)
                
                file_names.append(filename)
                
        except Exception as e:
            SESSION_QUEUES.pop(session_id, None)
            raise HTTPException(status_code=500, detail=f"Failed to save one or more files: {e}")

    master_agent = get_master_agent()
    asyncio.create_task(
        run_agent_work(
            master_agent,
            prompt,
            file_names,
            workspace,
            q
        )
    )

    # Return the session_id
    return JSONResponse({"status": "success", "session_id": session_id})

@inject
async def response_generator(session_id: str, request: Request):
    q = SESSION_QUEUES.get(session_id)
    
    if not q:
        data = json.dumps({"type": "error", "message": "Invalid or expired session ID"})
        yield f"data: {data}\n\n"
        yield f"data: [DONE]\n\n"
        return
    
    try:
          while True:
              if await request.is_disconnected():
                  print(f"Client for {session_id} disconnected.")
                  break
              
              try:
                  msg = await asyncio.wait_for(q.get(), timeout=1.0)
                  
                  if msg == "[DONE]":
                      yield f"data: [DONE]\n\n"
                      break
                  
                  yield f"data: {msg}\n\n"
                  q.task_done()
              
              except asyncio.TimeoutError:
                  continue
                  
    except asyncio.CancelledError:
        print(f"Generator for {session_id} was cancelled.")
    
    finally:
        # Clean up the queue from the global dict
        print(f"Cleaning up queue for {session_id}")
        SESSION_QUEUES.pop(session_id, None)

@process_router.get("/events/{session_id}")
async def stream_progress(request: Request, session_id: str):
    return StreamingResponse(response_generator(session_id=session_id, request=request), media_type="text/event-stream")