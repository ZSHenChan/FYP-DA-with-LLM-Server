import os, logging
from typing import List, Union
from contextlib import contextmanager
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from core.config import config
from .schemas import GlobalAgentState
from .utils import load_prompt, write_response_txt, SessionWorkspace
from .file_utils import FileUtils
from .graph import TaskGraph
from .sub_agents import AnalysisAgent


OPENAI_API_KEY = config.OPENAI_API_KEY

c_logger = logging.getLogger(config.CENTRAL_LOG_NAME)

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
    

    @contextmanager
    def _session_logger(self, workspace: SessionWorkspace):
        """Context manager to handle session-specific logging setup/teardown."""
        log_path = workspace.get_log_path()

        handler = logging.FileHandler(log_path)
        sess_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(sess_formatter)
        handler.setLevel(logging.DEBUG)
        
        sess_logger = logging.getLogger(config.SESS_LOG_NAME)
        sess_logger.setLevel(logging.DEBUG)
        sess_logger.addHandler(handler)
        # ... setup handler ...
        sess_logger = logging.getLogger(config.SESS_LOG_NAME)
        sess_logger.addHandler(handler)
        try:
            yield sess_logger
        finally:
            sess_logger.removeHandler(handler)
            handler.close()

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
        image_extensions = config.VISUAL_ALLOWED_EXTENSIONS
        
        image_paths = []
        for ext in image_extensions:
            pattern = os.path.join(figures_dir, ext)
            image_paths.extend(glob.glob(pattern))
        
        # Sort for consistent ordering
        return sorted(image_paths)

    # Not in use
    def _write_agent_state_to_json(self, sess_id: str, run_id: str, agent_state, ensure_dir: bool = True, indent: int = 1) -> None:
        """
        Write agent_state (GraphState or plain dict) to a JSON file.
        Non-serializable values are converted via str().
        """
        from pathlib import Path

        import json, os
        run_dir = Path(os.path.join(config.SESSION_FILEPATH, sess_id, run_id))
        sess_log = logging.getLogger(config.SESS_LOG_NAME)
        if ensure_dir:
            dirpath = os.path.dirname(run_dir) or "."
            os.makedirs(dirpath, exist_ok=True)

        filepath = os.path.join(run_dir, 'agent-state.json')

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(agent_state, f, ensure_ascii=False, indent=indent, default=str)
            sess_log.info(f"Agent State stored in {filepath}")
        except Exception as e:
            sess_log.error(f"Failed to save agent state: {e}")

    def _refine_task_graph_on_additional_request(self):
        # TODO
        raise RuntimeError('Need dev')
    
    def _initialize_agent_state(self, workplace:SessionWorkspace, requirement:str, file_list: List[str]) -> GlobalAgentState:
        state_dict = {'sess_id':workplace.sess_id, 'run_id':workplace.run_id, 'requirement':requirement, 'num_steps':0, 'raw_data_filenames':file_list, 'evaluation_results':[], 'visualization_paths':[], 'agent_messages':[]}
        state: GlobalAgentState = state_dict # type: ignore
        return state
    
    def _generate_final_report(self, state: GlobalAgentState) -> str:
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
    

    def run_request(self, 
                    human_input: str, 
                    file_list: List[str], 
                    workspace: SessionWorkspace,
                    progress_callback=None) -> str:
        
        import time
        workspace = SessionWorkspace(workspace.sess_id, workspace.run_id)
        
        start_time = time.time()
        log_summary_extra = {
            'sess_id': workspace.sess_id,
            'run_id': workspace.run_id,
            'user_request': human_input,
            'model':{'name':config.OPENAI_MODEL,'timeout':config.TIMEOUT,'cache':config.CACHE,'temperature':config.TEMPERATURE,'max_tokens':config.MAX_COMPLETION_TOKENS}
        }

        status = 'FAILED'
        state = self._initialize_agent_state(workplace=workspace, requirement=human_input, file_list=file_list)
        agent_state_version = {'initial':state}

        with self._session_logger(workspace) as logger:
            try:
                
                file_context = FileUtils.format_files_for_llm(file_list, workspace.data_dir)
                
                if len(self.task_graph.nodes) == 0:
                    logger.info('Initiating TaskGraph')
                    if progress_callback:
                        progress_callback(f'Initiating TaskGraph')
                    self.task_graph.generate_plan(human_input, file_context)
                else:
                    logger.info('Updating TaskGraph')
                    if progress_callback:
                        progress_callback(f'Starting to refine TaskGraph')
                    self._refine_task_graph_on_additional_request()
            
                workflow_state = self.task_graph.execute_pipeline(
                    initial_state=state, 
                    workspace=workspace,
                    progress_callback=progress_callback
                )
                self.task_graph.print_graph(sess_id=state['sess_id'], run_id=state['run_id'], verbose=True)
                agent_state_version['after workflow'] = workflow_state
                
                if progress_callback:
                    progress_callback(f'Analysis in Progress') 
                final_state = self._analyze_results(workflow_state, workspace)
                agent_state_version['final'] = final_state
                
                if progress_callback:
                    progress_callback(f'Fabricating Final Answer')
                answer = self._generate_final_report(agent_state_version)
                status = "SUCCESS"

                return answer

            except Exception as e:
                logger.error(f"Run failed: {e}", exc_info=True)
                return "An error occurred during execution."
            
            finally:
                duration_sec = time.time() - start_time
                log_summary_extra['status'] = status
                log_summary_extra['duration_sec'] = round(duration_sec, 2)
                c_logger.info(
                    f"Finished request",
                    extra=log_summary_extra
                )
                workspace.save_json("agent_state.json", final_state)
                logger.info(f"Closing log handler for run_id {workspace.run_id}")

                
    def _analyze_results(self, state: GlobalAgentState, workspace: SessionWorkspace) -> GlobalAgentState:
        """
        Checks the workspace for generated figures and triggers the 
        AnalysisAgent if any are found.
        """
        # 1. Ask the workspace what images exist
        diagram_paths = workspace.list_figures()
        
        # 2. Update the state with these paths so the AnalysisAgent knows what to look at
        # We create a copy to avoid mutating the state passed in unexpectedly, 
        # though AnalysisAgent usually returns a new state anyway.
        state['visualization_paths'] = diagram_paths

        # 3. If no diagrams were created, we might skip analysis or just return state
        if not diagram_paths:
            c_logger.info("No diagrams found to analyze.")
            return state

        # 4. Trigger the Analysis Agent
        # The prompt asks for insights specifically regarding the user's original requirement.
        prompt = f"Give insights on these diagrams regarding user request: {state['requirement']}"
        
        # Note: integration with your existing AnalysisAgent logic
        final_state = self.analysis_agent.analyze_all_diagrams(
            state=state, 
            prompt=prompt
        )
        
        return final_state
    
    def process_requirement(self, human_input:str, file_list: List[str], sess_id: str, run_id: str, progress_callback: Union[callable, None]) -> str:
        import time
        workspace = SessionWorkspace(sess_id, run_id)

        start_time = time.time()
        log_summary_extra = {
            'sess_id': sess_id,
            'run_id': run_id,
            'user_request': human_input,
            'model':{'name':config.OPENAI_MODEL,'timeout':config.TIMEOUT,'cache':config.CACHE,'temperature':config.TEMPERATURE,'max_tokens':config.MAX_COMPLETION_TOKENS}
        }
        c_logger.info(
            f"Start processing request: {human_input}",
            extra={'sess_id': sess_id, 'run_id': run_id, 'files': file_list}
        )

        status = 'failure'

        # Configure Session logger
        sess_log_dir = os.path.join(config.SESSION_FILEPATH, sess_id, run_id)
        os.makedirs(sess_log_dir, exist_ok=True)
        sess_log_file_path = os.path.join(sess_log_dir, config.SESS_LOG_FILENAME)

        sess_file_handler = logging.FileHandler(sess_log_file_path)
        sess_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sess_file_handler.setFormatter(sess_formatter)
        sess_file_handler.setLevel(logging.DEBUG)
        
        sess_logger = logging.getLogger(config.SESS_LOG_NAME)
        sess_logger.setLevel(logging.DEBUG)
        sess_logger.addHandler(sess_file_handler)

        # Summarize user request
        # user_request = self._summarize_user_request(human_req=human_input)
        user_request = human_input

        state: GlobalAgentState = self._initialize_agent_state(sess_id=sess_id, run_id=run_id, requirement=user_request, file_list=file_list)
        agent_state_version = {'initial':state}

        try:
            if len(self.task_graph.nodes) == 0:
                sess_logger.info('Initiating TaskGraph')
                if progress_callback:
                    progress_callback(f'Initiating TaskGraph')
                self.task_graph.initialize_task_graph(global_agent_state=state, human_input=user_request, file_list=file_list, progress_callback=progress_callback)
            else:
                sess_logger.info('Updating TaskGraph')
                if progress_callback:
                    progress_callback(f'Starting to refine TaskGraph')
                self._refine_task_graph_on_additional_request()


            workflow_state = self.task_graph.execute_pipeline(initial_state=state, progress_callback=progress_callback)
            self.task_graph.print_graph(sess_id=state['sess_id'], run_id=state['run_id'], verbose=True)
            agent_state_version['after workflow'] = workflow_state
            
            # Collect all diagrams under figures/
            visualisation_paths = self._collect_visualization_paths(sess_id=state['sess_id'], run_id=state['run_id'])
            workflow_state['visualization_paths'] = visualisation_paths
            
            if progress_callback:
                progress_callback(f'Analysis in Progress')
            final_state = self.analysis_agent.analyze_all_diagrams(state=workflow_state, prompt=f'Give insights on these diagrams regarding user request:{workflow_state["requirement"]}')
            agent_state_version['final'] = final_state

            if progress_callback:
                progress_callback(f'Fabricating Final Answer')

            # Synthesis final response
            response_str = self._generate_final_report(state=final_state)
            write_response_txt(response=str(response_str), sess_id=state['sess_id'], run_id=state['run_id'])

            # self._migrate_run_outputs(sess_id=state['sess_id'], run_id=state['run_id'])

            # try:
            #     main_temp_dir = Path(os.path.join(config.TEMP_FILEPATH, sess_id))
            #     main_temp_dir.rmdir()
            # except OSError as e:
            #     # We expect this to fail if not empty, which is fine
            #     sess_logger.error(f"Could not fully clean up temp dir {main_temp_dir}: {e}")

            status = "success"
            sess_logger.info("Run completed successfully.")
            return response_str
        
        except Exception as e:
            sess_logger.error(f"FATAL ERROR in run {run_id}: {e}", exc_info=True)
            c_logger.error(
                f"Finished request",
                extra={'traceback',e}
            ) 
            return 'Something went wrong'

        finally:
            # Post action log
            duration_sec = time.time() - start_time
            log_summary_extra['status'] = status
            log_summary_extra['duration_sec'] = round(duration_sec, 2)
            c_logger.info(
                f"Finished request",
                extra=log_summary_extra
            )
            self._write_agent_state_to_json(sess_id=sess_id, run_id=run_id, agent_state=agent_state_version)

            # Clean up session logger
            sess_logger.info(f"Closing log handler for run_id {run_id}")
            sess_logger.removeHandler(sess_file_handler)
            sess_file_handler.close()
    
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
    

def get_master_agent():
    return MasterAgent(model=config.OPENAI_MODEL)