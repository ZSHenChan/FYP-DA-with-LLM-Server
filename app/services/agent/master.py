import logging
from typing import Any, Dict, List
from contextlib import contextmanager
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from core.config import config
from .schemas import GlobalAgentState, FinalAnswer, PydanticDiagramResult
from .utils import load_prompt, SessionWorkspace
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

    
    def _initialize_agent_state(self, workplace:SessionWorkspace, requirement:str, file_list: List[str]) -> GlobalAgentState:
        state_dict = {'sess_id':workplace.sess_id, 'run_id':workplace.run_id, 'requirement':requirement, 'num_steps':0, 'raw_data_filenames':file_list, 'evaluation_results':[], 'visualization_paths':[], 'agent_messages':[]}
        state: GlobalAgentState = state_dict # type: ignore
        return state
    
    def _generate_final_report(self, state: GlobalAgentState) -> FinalAnswer:
        messages = [
            SystemMessage(content= self.instructions_ans),
            AIMessage(content=f'analysis results:{str(state["analysis_result"])}'),
            HumanMessage(content=f'user question: {str(state["requirement"])}'),
        ]

        response_str: str = self.llm.invoke(messages).text()

        return FinalAnswer(text=response_str, run_id=state['run_id'], figures=state['analysis_result'].figures)
    
    # Not in use
    def _summarize_user_request(self, human_req: str) -> str:
        messages = [
            SystemMessage(content= self.instructions_user_req),
            HumanMessage(content=f'user request:{human_req}'),
        ]

        response = self.llm.invoke(messages)
        return response.text
    
    def run_request_demo(self,
                    human_input: str, 
                    file_list: List[str], 
                    workspace: SessionWorkspace,
                    progress_callback=None) -> FinalAnswer:
        
        # return FinalAnswer(text="Result.", run_id=workspace.run_id, figures=[PydanticDiagramResult(filename="gold_annual_returns.png", text="Analysis Result")])
        with self._session_logger(workspace) as logger:
            try:
                state = self._initialize_agent_state(workplace=workspace, requirement=human_input, file_list=[])
                diagram_paths = workspace.list_figures()
                state['visualization_paths'] = diagram_paths
                if progress_callback:
                    progress_callback(f'Analysis in Progress') 
                final_state = self.analysis_agent.analyze_all_diagrams(state=state, prompt=f'Give insights on these diagrams regarding user request:{state["requirement"]}')
                if progress_callback:
                    progress_callback(f'Fabricating Final Answer')
                final_result = self._generate_final_report(final_state)
                logger.info(final_result)
                return final_result

            except Exception as e:
                logger.error(f"Run failed: {e}", exc_info=True)
                return FinalAnswer(text="An error occurred during execution.", run_id=workspace.run_id)
            
    def run_request(self, 
                    human_input: str, 
                    file_list: List[str], 
                    workspace: SessionWorkspace,
                    progress_callback=None) -> FinalAnswer:
        
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
                file_context: List[Dict[str, Any]] = FileUtils.format_files_for_llm(file_list, workspace.data_dir)
                
                previous_state = workspace.load_graph_state()
                if previous_state:
                    logger.info("Restoring previous session state...")
                    if progress_callback:
                        progress_callback('Restoring previous session state...')
                    self.task_graph = TaskGraph.from_dict(previous_state)
                    
                    logger.info('Updating TaskGraph')
                    if progress_callback:
                        progress_callback(f'Starting to refine TaskGraph')
                    self.task_graph.refine_plan(add_input=human_input, file_list=file_context)
                else:
                    logger.info('Initiating TaskGraph')
                    if progress_callback:
                        progress_callback(f'Initiating TaskGraph')
                    self.task_graph.generate_plan(human_input=human_input, file_list=file_context)

                # Graph Execution
                workflow_state = self.task_graph.execute_pipeline(
                    initial_state=state,
                    workspace=workspace,
                    progress_callback=progress_callback
                )
                self.task_graph.print_graph(sess_id=state['sess_id'], run_id=state['run_id'], verbose=True)
                agent_state_version['after workflow'] = workflow_state
                
                # Diagram Analysis
                if progress_callback:
                    progress_callback(f'Analysis in Progress') 

                diagram_paths = workspace.list_figures()
                final_state = workflow_state
                if not diagram_paths:
                    logger.info("No diagrams found to analyze.")

                else:
                    logger.info(f"{len(diagram_paths)} diagrams found to analyze.")
                    final_state['visualization_paths'] = diagram_paths
                    final_state = self.analysis_agent.analyze_all_diagrams(state=final_state, prompt=f'Give insights on these diagrams regarding user request:{final_state["requirement"]}')
                
                agent_state_version['final'] = final_state
                # Response Synthesis
                if progress_callback:
                    progress_callback(f'Fabricating Final Answer')
                final_result = self._generate_final_report(final_state)
                status = "SUCCESS"

                return final_result
            
            except Exception as e:
                logger.error(f"Run failed: {e}", exc_info=True)
                return FinalAnswer(text="An error occurred during execution.", run_id=workspace.run_id)
            
            finally:
                logger.info('Saving TaskGraph state...')
                workspace.save_graph_state(self.task_graph.to_dict())

                # Generate Request Summary
                duration_sec = time.time() - start_time
                log_summary_extra['status'] = status
                log_summary_extra['duration_sec'] = round(duration_sec, 2)
                c_logger.info(
                    f"Finished request",
                    extra=log_summary_extra
                )
                workspace.save_json("agent_state.json", agent_state_version)
                logger.info(f"Closing log handler for run_id {workspace.run_id}")


def get_master_agent():
    return MasterAgent(model=config.OPENAI_MODEL)