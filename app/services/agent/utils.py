import os, json, sys, io, logging
from dataclasses import dataclass
from pathlib import Path
from core.config import config
from .schemas import GlobalAgentState

logger = logging.getLogger(config.SESS_LOG_NAME)

from pathlib import Path
from core.config import config

class SessionWorkspace:
    """
    Encapsulates all file system operations for a specific run.
    No other class should import 'config' for path manipulation.
    """
    def __init__(self, sess_id: str, run_id: str):
        self.sess_id = sess_id
        self.run_id = run_id

        # stored at session/{sess_id}
        self.session_base = Path(config.SESSION_FILEPATH) / sess_id
        self.data_dir = self.session_base / config.DATA_FILEPATH
        
        # stored at session/{sess_id}/{run_id}
        self.run_base = self.session_base / run_id
        self.figure_dir = self.run_base / config.FIGURE_FILEPATH
        self.model_dir = self.run_base / config.MODEL_FILEPATH
        
        # Ensure directories exist
        for d in [self.session_base, self.data_dir, self.figure_dir, self.model_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_log_path(self) -> Path:
        return self.run_base / config.SESS_LOG_FILENAME

    def save_json(self, filename: str, data: dict):
        sess_log = logging.getLogger(config.SESS_LOG_NAME)

        filepath = os.path.join(self.run_base, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=True, default=str)
            sess_log.info(f"Agent State stored in {filepath}")
        except Exception as e:
            sess_log.error(f"Failed to save agent state: {e}")

    def list_figures(self) -> list[str]:
        """
        Returns a sorted list of absolute string paths for all images 
        generated in the current run's figure directory.
        """
        image_paths = []
        # We look for extensions defined in your config (e.g., .png, .jpg)
        for ext in config.VISUAL_ALLOWED_EXTENSIONS:
            # Using pathlib's glob is cleaner than os.glob
            # pattern e.g., "*.png"
            pattern = f"*{ext}" if ext.startswith(".") else f"*.{ext}"
            found = list(self.figure_dir.glob(pattern))
            image_paths.extend(found)
        
        # Sort them so the analysis order is deterministic
        image_paths.sort(key=lambda p: p.name)
        
        # Convert Path objects to strings for the State dict
        return [str(p) for p in image_paths]
    
def load_prompt(agent_name, key: str='system_prompt') -> str:
    file_path = f'prompts.json'
    try:
        with open(file_path, 'r') as f:
            prompt_data = json.load(f)
        result = prompt_data[agent_name][key]
        return result
    except FileNotFoundError:
        logger.error(f"Error: The file was not found at {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {file_path}. Check for syntax...")
        raise
    except KeyError as e:
        logger.error(f"Error: Missing key {e} in your JSON file.")
        raise

def increase_num_steps(state: GlobalAgentState):
    try:
      state['num_steps'] += 1
    except Exception as e:
      pass

@dataclass
class ExecuteResult:
    success: bool
    message: str | None
    namespace: dict

class CodeExecutor:
    """A class to execute code and capture printed output."""
    def __init__(self, namespace: dict):
        self.namespace = namespace

    def execute(self, code: str) -> tuple[bool, str]:
        old_stdout = sys.stdout
        old_stderr = sys.stderr  
        redirected_output = io.StringIO()
        redirected_errors = io.StringIO()

        sys.stdout = redirected_output
        sys.stderr = redirected_errors

        exec_globals = self.namespace

        try:
            exec(code, exec_globals)

            stdout_content = redirected_output.getvalue()
            stderr_content = redirected_errors.getvalue()
            combined_output = stdout_content
            if stderr_content:
                combined_output += "\n--- Warnings/Errors ---\n" + stderr_content
            return True, combined_output 
        
        except Exception as e:
            stdout_content = redirected_output.getvalue()
            stderr_content = redirected_errors.getvalue()
            
            error_message = f"{type(e).__name__}: {e}"
            
            # Combine all captured info for the debug prompt
            combined_output = "--- Captured Stdout ---\n" + stdout_content
            combined_output += "\n--- Captured Stderr ---\n" + stderr_content
            combined_output += "\n--- Exception ---\n" + error_message
            
            return False, combined_output
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if '__builtins__' in exec_globals:
                exec_globals.pop('__builtins__')

def write_response_txt(response: str, sess_id: str, run_id: str, filename: str = "response.txt", ensure_dir: bool = True) -> str:
    """
    Save `response` as a text file under {config.DATA_FILEPATH}{sess_id}/{run_id}/filename.
    """
    dest_dir = Path(os.path.join(config.SESSION_FILEPATH,sess_id,run_id))
    if ensure_dir:
        dest_dir.mkdir(parents=True, exist_ok=True)
    filepath = dest_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(response)
    logger.info(f"Saved response to {filepath}")
    return str(filepath)

def comment_block(text: str, prefix: str = "# ") -> str:
    lines = text.splitlines()
    out_lines = [(prefix + line) if line.strip() else prefix.rstrip() for line in lines]
    return "\n".join(out_lines)

def write_with_commented_instructions(path: str, instructions: str, marker: str = "# --- INSTRUCTIONS ---"):
    commented = comment_block(instructions)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + marker + "\n")
        f.write(commented + "\n")
        f.write(marker.replace("INSTRUCTIONS", "END_INSTRUCTIONS") + "\n")