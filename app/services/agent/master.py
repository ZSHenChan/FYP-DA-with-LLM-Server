import os
from typing import List, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from core.config import config
from .schemas import GlobalAgentState
from .utils import load_prompt, write_response_txt
from .graph import TaskGraph
from .sub_agents import AnalysisAgent

OPENAI_API_KEY = config.OPENAI_API_KEY

class MasterAgent:
    def __init__(self, model:str, tools=[], max_retries:int = 3):

        self.max_retries = max_retries
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=OPENAI_API_KEY,
            temperature=config.TEMPERATURE,
            max_completion_tokens=config.MAX_COMPLETION_TOKENS,
            timeout=config.TIMEOUT,
            max_retries=config.LLM_MAX_RETRIES
          )
        self.tools = tools
        self.instructions_ans = load_prompt(agent_name='master', key="system_prompt_ans")
        self.instructions_user_req = load_prompt(agent_name='master', key="system_prompt_user_req")
        self.task_graph: TaskGraph = TaskGraph(model=config.OPENAI_MODEL)
        self.analysis_agent = AnalysisAgent(model=config.OPENAI_MODEL)
    

    def _collect_visualization_paths(self, sess_id:str, run_id:str) -> list[str]:
        """
        Collect all image file paths from the figures directory.
        
        Args:
            figures_dir: Directory containing visualization files
            
        Returns:
            List of file paths as strings
        """
        import glob
        from pathlib import Path
        # Ensure directory exists
        figures_dir = Path(os.path.join(config.SESSION_FILEPATH, sess_id, run_id, config.FIGURE_FILEPATH))
        if not os.path.exists(figures_dir):
            return []
        
        # Common image extensions
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf']
        
        image_paths = []
        for ext in image_extensions:
            pattern = os.path.join(figures_dir, ext)
            image_paths.extend(glob.glob(pattern))
        
        # Sort for consistent ordering
        return sorted(image_paths)

    def generate_id(self, prefix: str | None = None, uuid_len: int = config.UUID_LEN) -> str:
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
            return uuid.uuid4().hex[:uuid_len]
        return f"{prefix}_{uuid.uuid4().hex[:uuid_len]}"

    def _write_agent_state_to_json(self, agent_state: GlobalAgentState, ensure_dir: bool = True, indent: int = 1) -> None:
        """
        Write agent_state (GraphState or plain dict) to a JSON file.
        Non-serializable values are converted via str().
        """
        from pathlib import Path

        import json, os
        run_dir = Path(os.path.join(config.DATA_FILEPATH,agent_state['sess_id'],agent_state['run_id']))

        if ensure_dir:
            dirpath = os.path.dirname(run_dir) or "."
            os.makedirs(dirpath, exist_ok=True)

        filepath = os.path.join(run_dir, 'agent-state.json')

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(agent_state, f, ensure_ascii=False, indent=indent, default=str)
            print(f"Agent State stored in {filepath}")
        except Exception as e:
            print(f"Failed to save agent state: {e}")

    def _refine_task_graph_on_additional_request(self):
        # TODO
        raise RuntimeError('Need dev')
    
    def _initialize_agent_state(self, sess_id:str, run_id: str, requirement:str, file_list: List[str]) -> GlobalAgentState:
        state_dict = {'sess_id':sess_id, 'run_id':run_id, 'requirement':requirement, 'num_steps':0, 'raw_data_filenames':file_list, 'evaluation_results':[], 'visualization_paths':[], 'agent_messages':[]}
        state: GlobalAgentState = state_dict # type: ignore
        return state
    
    def _provide_answer(self, state: GlobalAgentState) -> str:
        messages = [
            SystemMessage(content= self.instructions_ans),
            AIMessage(content=f'evaluation results:{str(state["evaluation_results"])}'),
            HumanMessage(content=f'user question: {str(state["requirement"])}'),
        ]

        response_str: str = self.llm.invoke(messages).text()

        return response_str
    
    def _summarize_user_request(self, human_req: str) -> str:
        messages = [
            SystemMessage(content= self.instructions_user_req),
            HumanMessage(content=f'user request:{human_req}'),
        ]

        response = self.llm.invoke(messages)
        return response.text
    
    def _migrate_all_current_run_data(self, sess_id:str, run_id:str, create_dest: bool = True):
        """
        Move all current run files {config.SESSION_FILEPATH}/{sess_id}/{run_id}.
        
        Args:
            sess_id: Session ID
            run_id: Run ID
            create_dest: Create destination directory if it doesn't exist (default: True)
            
        Returns:
            Dictionary with results: {'success': int, 'failed': int, 'errors': list}
        """
        import shutil
        from pathlib import Path
        results = {'success': 0, 'failed': 0, 'errors': []}
    
        # Convert to Path objects
        source_dir = Path(os.path.join(config.TEMP_FILEPATH, sess_id))
        dest_dir = Path(os.path.join(config.SESSION_FILEPATH, sess_id, run_id))
        
        # Check if source exists
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
        
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {source_dir}")
        
        # Create destination if needed
        if create_dest:
            dest_dir.mkdir(parents=True, exist_ok=True)
        elif not dest_dir.exists():
            raise FileNotFoundError(f"Destination directory does not exist: {dest_dir}")
        
        # Move all folders/files
        for item_name in os.listdir(source_dir):
            try:
                source_path = os.path.join(source_dir, item_name)
                target_path = os.path.join(dest_dir, item_name)
                shutil.move(source_path, target_path)
                results['success'] += 1
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"{item_name}: {str(e)}")
        
        print(f"\nSummary: {results['success']} files moved, {results['failed']} failed")
        return results

    def process_requirement(self, human_input:str, file_list: List[str], sess_id: str, progress_callback: Union[callable, None]) -> str:
        run_id = self.generate_id(prefix='run')
        print(f"== Run ID: {run_id} ==\n")

        # Summarize user request
        # user_request = self._summarize_user_request(human_req=human_input)
        user_request = human_input

        state: GlobalAgentState = self._initialize_agent_state(sess_id=sess_id, run_id=run_id, requirement=user_request, data_path=f'tmp/{sess_id}/data/data.csv')

        if len(self.task_graph.nodes) == 0:
            print('Initiating TaskGraph')
            if progress_callback:
                progress_callback(f'Initiating TaskGraph')
            self.task_graph.initialize_and_populate_task_graph(global_agent_state=state, human_input=user_request, file_list=file_list, progress_callback=progress_callback)
        else:
            print('Starting to refine TaskGraph')
            if progress_callback:
                progress_callback(f'Starting to refine TaskGraph')
            self._refine_task_graph_on_additional_request()


        self.task_graph.print_graph(sess_id=state['sess_id'], run_id=state['run_id'], verbose=True)
        workflow_state = self.task_graph.run_workflow(agent_state=state)

        self._migrate_all_current_run_data(sess_id=state['sess_id'], run_id=state['run_id'])
        # Collect all diagrams under data/figures
        visualisation_paths = self._collect_visualization_paths(sess_id=state['sess_id'], run_id=state['run_id'])
        workflow_state['visualization_paths'] = visualisation_paths
        
        print("== Analysis in Progress")
        if progress_callback:
            progress_callback(f'Analysis in Progress')
        final_state = self.analysis_agent.analyze_all_diagrams(state=workflow_state, prompt=f'Give insights on these diagrams regarding user request:{workflow_state["requirement"]}')
        self._write_agent_state_to_json(agent_state=final_state)

        print("== Fabricating Final Answer")
        if progress_callback:
            progress_callback(f'Fabricating Final Answer')
        response_str = self._provide_answer(state=final_state)
        write_response_txt(response=str(response_str), sess_id=state['sess_id'], run_id=state['run_id'])

        return response_str
    
    def process_demo(self, human_input:str, file_list: List[str], sess_id: str, progress_callback: Union[callable, None] = None) -> str:
        from time import sleep
        if progress_callback:
            progress_callback(f"Initiating Task")
            sleep(10)
            progress_callback(f'Analysing Diagrams')
            sleep(3)
            progress_callback(f'Fabricating Final Answer')
            sleep(3)

        return '''Short summary of what the data already suggests
- There is an average negative association between gold price and traded volume: higher volumes tend to occur with lower prices.  
- The relationship is weak and unstable: scatter points are widely spread with clusters and outliers, residuals from a simple linear fit show curvature and changing spread, and residuals have heavy tails.  
- Time structure matters: price trends upward while volume tends downward with spikes. The price–volume link appears to change over time (nonstationarity / regime changes).
'''
        # return read_text_file("test-response.txt")
    

def get_master_agent():
    return MasterAgent(model=config.OPENAI_MODEL)