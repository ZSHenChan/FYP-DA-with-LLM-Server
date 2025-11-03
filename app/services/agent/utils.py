import os
import json
import sys
import io
from dataclasses import dataclass
from pathlib import Path
from core.config import config
from .schemas import GlobalAgentState

def load_prompt(agent_name, key: str='system_prompt') -> str:
    file_path = f'prompts.json'
    try:
        with open(file_path, 'r') as f:
            prompt_data = json.load(f)
        result = prompt_data[agent_name][key]
        return result
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Check for syntax...")
        raise
    except KeyError as e:
        print(f"Error: Missing key {e} in your JSON file.")
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
        # ... (same as original file)
        old_stdout = sys.stdout
        old_stderr = sys.stderr  
        redirected_output = io.StringIO()
        redirected_errors = io.StringIO()
        sys.stdout = redirected_output
        sys.stderr = redirected_errors
        try:
            if 'agent_state' in self.namespace:
                exec_globals = {'agent_state': self.namespace['agent_state']}
                exec_globals.update(self.namespace)
            else:
                exec_globals = self.namespace
            exec(code, exec_globals)
            self.namespace.update(exec_globals)
            stdout_content = redirected_output.getvalue()
            stderr_content = redirected_errors.getvalue()
            combined_output = stdout_content
            if stderr_content:
                combined_output += "\n--- Warnings/Errors ---\n" + stderr_content
            return True, combined_output
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

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
    print(f"Saved response to {filepath}")
    return str(filepath)