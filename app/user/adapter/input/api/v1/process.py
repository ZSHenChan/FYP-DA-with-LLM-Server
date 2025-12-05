import json, os, logging
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

logger = logging.getLogger(__name__)

def generate_id(prefix: str | None = None) -> str:
    import uuid
    """
    Generate a shorter run ID using first 8 characters of UUID.
    
    Args:
        prefix: Prefix for the run ID
        
    Returns:
        Generated run ID string
        
    Example:
        - generate_run_id_short(prefix='run') -> "run_a1b2c3d4"
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
        """Sends a ping every config.KEEPALIVE_INTERVAL seconds to keep the connection alive."""
        while True:
            await asyncio.sleep(config.KEEPALIVE_INTERVAL)
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
    request: Request,
    prompt: str = Form(...),
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[])
):
    from pathlib import Path

    if session_id:
        # Validate if session actually exists on disk to prevent phantom sessions
        # (Optional security check)
        potential_path = Path(config.SESSION_FILEPATH) / session_id
        if not potential_path.exists():
            # You can either raise error or just create a new one. 
            # Creating new is usually safer for UX.
            logger.warning(f"Session {session_id} not found, creating new.")
            session_id = generate_id(prefix='sess')
    else:
        session_id = generate_id(prefix='sess')

    run_id = generate_id(prefix='run')
    
    q = Queue()
    SESSION_QUEUES[session_id] = q

    workspace = SessionWorkspace(session_id, run_id)
    file_names: List[str] = []

    # Save new files (if any)
    if files:
        try:
            for file in files:
                if not file.filename: continue
                
                file_path = workspace.data_dir / Path(file.filename).name
                
                # Async read/write
                content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                file_names.append(file.filename)
        except Exception as e:
            SESSION_QUEUES.pop(session_id, None)
            raise HTTPException(status_code=500, detail=f"File save failed: {e}")
        
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

@process_router.get("/events/{session_id}")
async def stream_progress(request: Request, session_id: str):
    async def event_generator():
        q = SESSION_QUEUES.get(session_id)
        
        if not q:
            # Send an error event then close
            err = json.dumps({"type": "error", "message": "Session expired or invalid"})
            yield f"data: {err}\n\n"
            yield f"data: [DONE]\n\n"
            return
        
        yield f": connected\n\n"
        
        try:
            while True:
                # Check for client disconnect
                if await request.is_disconnected():
                    logger.info(f"Client {session_id} disconnected")
                    break
                
                try:
                    # Wait for message with timeout to allow checking disconnect status
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    
                    if msg == "[DONE]":
                        yield f"data: [DONE]\n\n"
                        break
                    
                    yield f"data: {msg}\n\n"
                    q.task_done()
                
                except asyncio.TimeoutError:
                    # Just loop back to check connection status
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Stream cancelled for {session_id}")
        
        finally:
            # Cleanup: Remove the queue to free memory
            # In a chat app, we remove it because the response is done.
            # The next POST /start will create a NEW queue.
            SESSION_QUEUES.pop(session_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")