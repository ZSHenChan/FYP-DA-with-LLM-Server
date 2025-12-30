import logging
from typing import Any, Dict, List, Union
from contextlib import contextmanager
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, messages_to_dict, messages_from_dict
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
        """Initialize master agent with LLM client, tools, prompts, and sub-agents."""
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
        self.instructions_ans = load_prompt(agent_name=config.AGENT_NAME_MASTER, key=config.PROMPT_KEY_MASTER_ANS)
        self.instructions_user_req = load_prompt(agent_name=config.AGENT_NAME_MASTER, key=config.PROMPT_KEY_MASTER_REQ)
        self.task_graph: TaskGraph = TaskGraph(model=config.OPENAI_MODEL)
        self.analysis_agent = AnalysisAgent(model=config.OPENAI_MODEL)
        self.conversation_history: List[BaseMessage] = []
  
    def _get_history_file_path(self, workspace: SessionWorkspace) -> str:
        """
        Constructs the path for the session-level history file.
        It should be outside the specific run folder, in the session root.
        """
        import os
        session_dir = os.path.dirname(workspace.run_base) 
        return os.path.join(session_dir, config.FILENAME_SESSION_HISTORY)

    def _load_conversation_history(self, workspace: SessionWorkspace):
        """Loads existing conversation history from JSON."""
        import os, json
        history_path = self._get_history_file_path(workspace)
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversation_history = messages_from_dict(data)
                c_logger.info(f"Loaded {len(self.conversation_history)} messages.")
            except Exception as e:
                c_logger.error(f"Failed to load history: {e}")
                self.conversation_history = []
        else:
            self.conversation_history = []

    def _save_conversation_history(self, workspace: SessionWorkspace):
        """Saves current conversation history to JSON."""
        import json
        history_path = self._get_history_file_path(workspace)
        try:
            messages_data = messages_to_dict(self.conversation_history)
            
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(messages_data, f, indent=2, ensure_ascii=False)
            c_logger.info(f"Saved history to {history_path}")
        except Exception as e:
            c_logger.error(f"Failed to save history: {e}")

    @contextmanager
    def _session_logger(self, workspace: SessionWorkspace):
        """Context manager to handle session-specific logging setup/teardown."""
        log_path = workspace.get_log_path()

        handler = logging.FileHandler(log_path)
        sess_formatter = logging.Formatter(config.SESS_LOG_FORMAT)
        handler.setFormatter(sess_formatter)
        handler.setLevel(logging.DEBUG)
        
        sess_logger = logging.getLogger(config.SESS_LOG_NAME)
        sess_logger.setLevel(logging.DEBUG)
        sess_logger.addHandler(handler)

        # setup handler
        sess_logger = logging.getLogger(config.SESS_LOG_NAME)
        sess_logger.addHandler(handler)
        try:
            yield sess_logger
        finally:
            sess_logger.removeHandler(handler)
            handler.close()

    def _initialize_agent_state(self, workplace:SessionWorkspace, requirement:str, file_list: List[str]) -> GlobalAgentState:
        """Create the initial agent state payload for a run."""
        state = GlobalAgentState(sess_id=workplace.sess_id, run_id=workplace.run_id, requirement=requirement, num_steps=0, raw_data_filenames=file_list, evaluation_results=[], visualization_paths=[], agent_messages=[])
        return state
    
    def _generate_final_report(self, state: GlobalAgentState) -> FinalAnswer:
        """Generate the final user-facing answer from the aggregated state."""
        messages = [
            SystemMessage(content= self.instructions_ans),
            AIMessage(content=f'analysis results:{str(state["analysis_result"])}'),
            HumanMessage(content=f'user question: {str(state["requirement"])}'),
        ]
        response_content: str = self.llm.invoke(messages).content

        return FinalAnswer(text=response_content, run_id=state['run_id'], figures=state['analysis_result'].figures)
    
    #! Not in use
    def _summarize_user_request(self, human_req: str) -> str:
        """Summarize the user request with the master prompt."""
        messages = [
            SystemMessage(content= self.instructions_user_req),
            HumanMessage(content=f'user request:{human_req}'),
        ]

        response = self.llm.invoke(messages)
        return response.text
    
    def _log_and_notify(self, message: str, logger: logging.Logger, progress_callback: Union[callable, None] = None, level: str = "info"):
        """
        Logs a message and optionally sends it to the progress callback.
        """
        if level.lower() == "error":
            logger.error(message)
        elif level.lower() == "warning":
            logger.warning(message)
        else:
            logger.info(message)

        if progress_callback:
            progress_callback(message)
    
    def run_request_demo(self,
                    human_input: str, 
                    file_list: List[str], 
                    workspace: SessionWorkspace,
                    progress_callback=None) -> FinalAnswer:
        """Demo pathway: analyze existing figures and produce a final answer."""
        
        # return FinalAnswer(text="Result.", run_id=workspace.run_id, figures=[PydanticDiagramResult(filename="gold_annual_returns.png", text="Analysis Result")])
        with self._session_logger(workspace) as logger:
            try:
                state = self._initialize_agent_state(workplace=workspace, requirement=human_input, file_list=[])
                diagram_paths = workspace.list_figures()
                state['visualization_paths'] = diagram_paths
                self._log_and_notify('Analysis in Progress', logger=logger, progress_callback=progress_callback)
                final_state = self.analysis_agent.analyze_all_diagrams(state=state, prompt=f'Give insights on these diagrams regarding user request:{state["requirement"]}')
                self._log_and_notify('Fabricating Final Answer', logger=logger, progress_callback=progress_callback)
                final_result = self._generate_final_report(final_state)
                logger.info(final_result)
                return final_result

            except Exception as e:
                logger.error(f"Run failed: {e}", exc_info=True)
                return FinalAnswer(text=config.ERROR_MSG_EXECUTION_FAILED, run_id=workspace.run_id)
            
    def run_request(self, 
                    human_input: str, 
                    file_list: List[str], 
                    workspace: SessionWorkspace,
                    progress_callback=None) -> FinalAnswer:
        """Full workflow: build/refine task graph, execute tasks, analyze diagrams, and craft answer."""
        
        import time
        workspace = SessionWorkspace(workspace.sess_id, workspace.run_id)
        
        start_time = time.time()
        log_summary = {
            'sess_id': workspace.sess_id,
            'run_id': workspace.run_id,
            'user_request': human_input,
            'model':{'name':config.OPENAI_MODEL,'timeout':config.TIMEOUT,'cache':config.CACHE,'temperature':config.TEMPERATURE,'max_tokens':config.MAX_COMPLETION_TOKENS}
        }
        status = config.STATUS_FAILED

        self._load_conversation_history(workspace)
        self.conversation_history.append(
            HumanMessage(content=human_input, additional_kwargs={"run_id": workspace.run_id})
        )

        state: GlobalAgentState = self._initialize_agent_state(workplace=workspace, requirement=human_input, file_list=file_list)
        agent_state_version = {'initial':state}

        with self._session_logger(workspace) as logger:
            try:
                file_context: List[Dict[str, Any]] = FileUtils.format_files_for_llm(file_list, workspace.data_dir)
                
                previous_state = workspace.load_graph_state()
                if previous_state:
                    self._log_and_notify("Restoring previous session state...", logger=logger, progress_callback=progress_callback)
                    self.task_graph = TaskGraph.from_dict(previous_state)
                    
                    self._log_and_notify('Updating TaskGraph', logger=logger, progress_callback=progress_callback)
                    self.task_graph.refine_plan(file_context=file_context, history=self.conversation_history)
                else:
                    self._log_and_notify('Initiating TaskGraph', logger=logger, progress_callback=progress_callback)
                    self.task_graph.generate_plan(human_input=human_input, file_list=file_context)

                # Graph Execution
                workflow_state: GlobalAgentState = self.task_graph.execute_pipeline(
                    initial_state=state,
                    workspace=workspace,
                    progress_callback=progress_callback,
                    stop_on_failure=False
                )
                self.task_graph.save_code(sess_id=state['sess_id'], run_id=state['run_id'], verbose=True)
                agent_state_version['after workflow'] = workflow_state
                
                # Diagram Analysis
                self._log_and_notify('Analyzing Diagrams...', logger=logger, progress_callback=progress_callback)

                diagram_paths = workspace.list_figures()
                final_state: GlobalAgentState = workflow_state
                if not diagram_paths:
                    logger.info("No diagrams found to analyze.")

                else:
                    logger.info(f"{len(diagram_paths)} diagrams found to analyze.")
                    final_state['visualization_paths'] = diagram_paths
                    final_state = self.analysis_agent.analyze_all_diagrams(state=final_state, prompt=f'Give insights on these diagrams regarding user request:{final_state["requirement"]}')
                
                agent_state_version['final'] = final_state
                
                # Response Synthesis
                self._log_and_notify('Fabricating Final Answer', logger=logger, progress_callback=progress_callback)
                final_result: FinalAnswer = self._generate_final_report(final_state)
                
                self.conversation_history.append(
                    AIMessage(content=final_result['text'], additional_kwargs={"run_id": workspace.run_id})
                )

                status = config.STATUS_SUCCESS
                return final_result
            
            except Exception as e:
                logger.error(f"Run failed: {e}", exc_info=True)
                return FinalAnswer(text=config.ERROR_MSG_EXECUTION_FAILED, run_id=workspace.run_id)
            
            finally:
                self._save_conversation_history(workspace)
                workspace.save_graph_state(self.task_graph.to_dict())

                logger.info('Saved TaskGraph and Conversation History')

                # Generate Request Summary
                duration_sec = time.time() - start_time
                log_summary['status'] = status
                log_summary['duration_sec'] = round(duration_sec, 2)
                c_logger.info(
                    f"Finished request",
                    extra=log_summary
                )
                workspace.save_json(data=agent_state_version)
                logger.info(f"Closing log handler for run_id {workspace.run_id}")


def get_master_agent():
    """Factory helper to construct a MasterAgent with the default model."""
    return MasterAgent(model=config.OPENAI_MODEL)