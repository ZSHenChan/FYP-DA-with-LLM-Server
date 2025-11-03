# import os, json, sys, io
# from core.config import config

# OPENAI_API_KEY = config.OPENAI_API_KEY
# PROMPT_CONFIG = {
#     "prompts": {
#         "master": "1.31",
#         "code": "1.2",
#         "analysis": "1.0"
#     },
#     "model":"gpt-5-mini-2025-08-07"
# }

# IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf"]

# def load_prompt(agent_name, key: str='system_prompt') -> str:
#     file_path = f'prompts.json'
#     try:
#         with open(file_path, 'r') as f:
#             prompt_data = json.load(f)

#         result = prompt_data[agent_name][key]
#         return result

#     except FileNotFoundError:
#         print(f"Error: The file was not found at {file_path}")
#         raise
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode JSON from {file_path}. Check for syntax errors.")
#         raise
#     except KeyError as e:
#         print(f"Error: Missing key {e} in your JSON file.")
#         raise

# from enum import Enum
# from typing import TypedDict, List, Any, Union, Dict

# class AgentMessage(TypedDict):
#     """Represents a structured message from an agent to the shared state."""
#     sender: str
    
#     content: Any

# class GlobalAgentState(TypedDict):
#     """A comprehensive state for a data science pipeline."""
#     sess_id: str
#     run_id: str
#     requirement: str
#     num_steps: int

#     raw_data_path: List[str]
    
#     evaluation_results: List[AgentMessage]
#     visualization_paths: List[str]

#     agent_messages: List[AgentMessage]

# def increase_num_steps(state: GlobalAgentState):
#     try:
#       state['num_steps'] += 1
#     except Exception as e:
#       pass

# from dataclasses import dataclass

# # TODO run code in venv (Docker)

# class CodeExecutor:
#     """A class to execute code and capture printed output."""
#     def __init__(self, namespace: dict):
#         self.namespace = namespace

#     def execute(self, code: str) -> tuple[bool, str]:
#         old_stdout = sys.stdout
#         old_stderr = sys.stderr  
        
#         redirected_output = io.StringIO()
#         redirected_errors = io.StringIO()
        
#         sys.stdout = redirected_output
#         sys.stderr = redirected_errors

#         try:
#             if 'agent_state' in self.namespace:
#                 exec_globals = {'agent_state': self.namespace['agent_state']}
#                 exec_globals.update(self.namespace)
#             else:
#                 exec_globals = self.namespace
                
#             exec(code, exec_globals)
#             self.namespace.update(exec_globals)

#             stdout_content = redirected_output.getvalue()
#             stderr_content = redirected_errors.getvalue()
#             combined_output = stdout_content
#             if stderr_content:
#                 combined_output += "\n--- Warnings/Errors ---\n" + stderr_content

#             return True, combined_output
#         except Exception as e:
#             return False, f"{type(e).__name__}: {e}"
#         finally:
#             sys.stdout = old_stdout
#             sys.stderr = old_stderr

# @dataclass
# class ExecuteResult:
#     success: bool
#     message: str | None
#     namespace: dict

# import base64
# import copy
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# from pydantic import BaseModel, Field

# class PydanticAnalysisResult(BaseModel):
#     """Represents a structured message from an agent to the shared state."""
#     result: str = Field(..., description="analysis result")
    
# class AnalysisAgent:
#     """A class to perform analysis based on code outputs."""
#     def __init__(self, model:str):
#         self.instruction = load_prompt(agent_name='analysis')
#         self.llm = ChatOpenAI(
#             model=model,
#             openai_api_key=OPENAI_API_KEY,
#             temperature=config.TEMPERATURE,
#             max_completion_tokens=config.MAX_COMPLETION_TOKENS,
#             timeout=config.TIMEOUT,
#             max_retries=config.LLM_MAX_RETRIES
#         )
    
#     def _encode_image(self, image_path: str):
#         """Helper function to encode image to base64."""
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode('utf-8')
    
#     def analyze_diagram(self, prompt:str, file_path_list: List[str]) -> str:
#         """
#         Encodes the image and asks the vision model a question about it.
#         """
       
#         messages = [
#             SystemMessage(content= self.instruction),
#             HumanMessage(content=prompt),
#             HumanMessage(content=[{
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/png;base64,{self._encode_image(x)}"
#                     }
#                 } for x in file_path_list])
#         ]

#         structured_llm = self.llm.with_structured_output(PydanticAnalysisResult)

#         pydantic_response: PydanticAnalysisResult = structured_llm.invoke(messages) # type: ignore

#         response_str: str = pydantic_response.result

#         return response_str
    
#     def analyze_all_diagrams(self, prompt:str, state: GlobalAgentState) -> GlobalAgentState:
        
#         diagram_list: List[str] = state['visualization_paths']
#         print(f'== Analyzing {len(diagram_list)} diagrams...')

#         state_copy = copy.deepcopy(state)

#         analysis_result: AgentMessage = {'sender': 'Analysis Agent', 'content': []}

#         # for diagram_path in diagram_list:
#         analysis_result_str = self.analyze_diagram(prompt=prompt, file_path_list=diagram_list)

#         analysis_result['content'].append({'analysis_result': analysis_result_str})

#         if not state_copy.get('evaluation_results'):
#             state_copy['evaluation_results'] = []
#         state_copy['evaluation_results'].append(analysis_result)
#         return state_copy
        

# from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI

# class PydanticActionNode(BaseModel):
#     """Schema for action node."""
#     action_id: int = Field(..., description="The sequential ID of this action step, starting from 1.")
#     description: str = Field(..., description="A brief, natural language description of what this code does.")
#     code: str = Field(..., description="A valid, executable snippet of Python code for this action.")

# class PydanticActionGraph(BaseModel):
#     """Schema for task graph."""
#     task_nodes: List[PydanticActionNode] = Field(default=[] , description="List of task nodes for the process.")

# class TaskStatus(str, Enum):
#     """Enum for task status values."""
#     SUCCESS = "success"
#     FAILED = "failed" 
#     PENDING = "pending"

# class TaskType(str, Enum):
#     """Enum for task type values."""
#     DATA_LOADING = "data_loading"
#     EXPLORATION = "exploration"
#     FEATURE_ENGINEERING = "feature_engineering"
#     MODEL_TRAINING = "model_training"
#     EVALUATION = "evaluation"
#     VISUALIZATION = "visualization"

# class ActionNode:
#     """Represents a single executable code snippet within a task."""
#     def __init__(self, action_id: int, code: str, description: str):
#         self.action_id = action_id
#         self.description = description
#         self.code = code
#         self.status = TaskStatus.PENDING
#         self.result = None

#     def __repr__(self):
#         return f"ActionNode(id={self.action_id}, description='{self.description}', status='{self.status.value}', code='{self.code[:30]}...')"

# class ActionGraph:
#     """Manages the sequence of actions for a single parent task."""
#     def __init__(self):
#         self.nodes: List[ActionNode] = []
#         self.result: ExecuteResult | None = None

#     def add_action(self, node: ActionNode):
#         self.nodes.append(node)
#         self.nodes.sort(key=lambda n: n.action_id)
        
#     def __repr__(self):
#         return f"ActionGraph with {len(self.nodes)} actions."
    
#     def execute_action_graph(self, namespace: dict):
#         executor = CodeExecutor(namespace=namespace)
#         last_message = ''
#         for current_action_node in self.nodes:
#             exec_success, result = executor.execute(current_action_node.code)
#             if not exec_success:
#                 current_action_node.status = TaskStatus.FAILED
#                 self.result = ExecuteResult(success=False, message=result, namespace=executor.namespace)
#                 return
#             current_action_node.status = TaskStatus.SUCCESS
#             last_message = result
        
#         self.result = ExecuteResult(success=True, message=last_message, namespace=executor.namespace)

#     def print_actions(self):
#         for action in self.nodes:
#             print(f"  - ID: {action.action_id}")
#             print(f"    Description: {action.description}")
#             print(f"    Code: {action.code}")
#             print("--------------------")

# from typing import List

# class PydanticTaskNode(BaseModel):
#     """Schema for task node."""
#     task_id: str = Field(..., description="Unique id for the task node in number, e.g. 1, 2, 3 etc")
#     dependencies: List[str] = Field(..., description="A list of unique ids of nodes must be completed before this.")
#     instruction: str = Field(..., description="A concise instruction for the task node.")
#     task_type: TaskType = Field(description="Current status of the task")
#     output: str = Field(..., description="description of what data or model is produced.")

#     #  produces: List[str] = Field(default_factory=list, description="Named artifacts produced by this task (e.g. cleaned.csv, model.pkl).")
#     # consumes: List[str] = Field(default_factory=list, description="Named artifacts consumed by this task.")
#     # condition: Optional[str] = Field(None, description="Optional condition expression (evaluated at runtime) to decide whether to run this task.")
#     # parallelizable: bool = Field(True, description="Whether this task can be run in parallel with other independent tasks.")
#     # retry_policy: Optional[Dict[str, Any]] = Field(None, description="Retry policy, e.g. {'max_retries': 3, 'backoff': 'exponential'}")
    
# class PydanticTaskGraph(BaseModel):
#     """Schema for task graph."""
#     task_nodes: List[PydanticTaskNode] = Field(..., description="List of task nodes for the process.")

# from graphlib import TopologicalSorter, CycleError
# from langchain_core.prompts import ChatPromptTemplate
# import copy

# class TaskNode:
#     """Represents a single node in the task graph."""
#     def __init__(self, node_id: str, instruction: str, dependencies: list[str], task_type: TaskType, output: str, model: str):
#         if not isinstance(node_id, str) or not node_id:
#             raise ValueError("task_id must be a non-empty string.")
        
#         self.node_id = node_id
#         self.instruction = instruction
#         self.dependencies = dependencies
#         self.status = TaskStatus.PENDING
#         self.task_type = task_type
#         self.output = output
#         self.result: str | None = None
#         self.action_graph = ActionGraph()
#         self.llm = ChatOpenAI(
#             model=model,
#             openai_api_key=OPENAI_API_KEY,
#             temperature=config.TEMPERATURE,
#             max_completion_tokens=config.MAX_COMPLETION_TOKENS,
#             timeout=config.TIMEOUT,
#             max_retries=config.LLM_MAX_RETRIES
#         )

#     def __repr__(self):
#         """Provides a string representation for the task node."""
#         return (f"TaskNode(id='{self.node_id}', status='{self.status.value}', "
#                 f"instruction='{self.instruction[:30]}...', deps={self.dependencies})")

#     def generate_action_graph(self, global_agent_state: GlobalAgentState, namespace: dict, tool_sets=[], additional_instruction: str = "" ) -> ActionGraph:
#         structured_llm = self.llm.with_structured_output(PydanticActionGraph)
#         data_dir = os.path.join(config.TEMP_FILEPATH, global_agent_state['sess_id'], config.DATA_FILEPATH)
#         figure_dir = os.path.join(config.TEMP_FILEPATH, global_agent_state['sess_id'], config.DATA_FILEPATH)
#         model_dir = os.path.join(config.TEMP_FILEPATH, global_agent_state['sess_id'], config.DATA_FILEPATH)
        
#         sys_prompt_template = load_prompt(agent_name='code')
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", sys_prompt_template),
#             ("human", "Instruction: {instruction}. Additional Instruction: {additional_instruction}. Agent State: {agent_state}. namespace: {namespace}")
#         ])
#         chain = prompt | structured_llm
        
#         pydantic_action_graph: PydanticActionGraph = chain.invoke({"instruction": self.instruction, "additional_instruction":additional_instruction, "agent_state":global_agent_state, "namespace":namespace, "data_dir":{data_dir}, "figure_dir":{figure_dir}, "model_dir":{model_dir}}) # type: ignore
        
#         action_graph = ActionGraph()
#         for p_node in pydantic_action_graph.task_nodes:
#             action_node = ActionNode(
#                 action_id=p_node.action_id,
#                 description=p_node.description,
#                 code=p_node.code
#             )
#             action_graph.add_action(action_node)
        
#         increase_num_steps(global_agent_state)
#         return action_graph

#     def refine_and_update_action_graph(self, global_agent_state: GlobalAgentState, namespace: dict):
#         refined_graph: ActionGraph = self.generate_action_graph(global_agent_state=global_agent_state, namespace=namespace, additional_instruction=f"A few errors were encountered when running the actions one-by-one. The current ActionGraph is {str(self.action_graph.nodes)}. You need to debug the code using the error message: {str(self.action_graph.result)}")
#         self.action_graph.nodes = refined_graph.nodes
#         self.action_graph.result = None

#     def iterative_refining(self, global_agent_state: GlobalAgentState, max_retries: int = config.TASK_NODE_MAX_RETRIES):
#         """Generate and refine the ActionGraph within a trial limit."""
#         debug_agent_state: GlobalAgentState = copy.deepcopy(global_agent_state)
        
#         namespace = {'agent_state':{}}

#         action_graph: ActionGraph = self.generate_action_graph(global_agent_state=debug_agent_state, namespace=namespace)
#         self.action_graph = action_graph
#         for _ in range(max_retries):
#             namespace = {'agent_state':{}}
#             self.action_graph.execute_action_graph(namespace)
#             if self.action_graph.result is None:
#                 raise Exception("Action Graph Result is None")
#             if self.action_graph.result.success is False:
#                 self.refine_and_update_action_graph(global_agent_state=debug_agent_state, namespace=namespace)
#                 increase_num_steps(global_agent_state)
#             else:
#                 self.status = TaskStatus.SUCCESS
#                 break
#         if self.action_graph.result is not None and self.action_graph.result.success is False:
#             self.status = TaskStatus.FAILED
#             self.result = str(self.action_graph.result)

#     def run_action_graph(self, agent_state: GlobalAgentState) -> GlobalAgentState:
#         namespace = {'agent_state':{}}
#         agent_state_copy = copy.deepcopy(agent_state)

#         self.action_graph.execute_action_graph(namespace)
#         if self.action_graph.result is not None and self.action_graph.result.success:
#             # Update current agent work to Global Agent State
#             final_state = self.action_graph.result.namespace.get('agent_state',{})
#             if final_state:
#                 agent_state_copy['agent_messages'].append({'sender': self.task_type + ' agent', 'content':final_state})
#             return agent_state_copy
#         else:
#             raise RuntimeError(f'Failed to run ActionGraph {self.node_id}: {self.action_graph.result}')

# class TaskGraph:
#     """Manages the entire Directed Acyclic Graph (DAG) of tasks."""
#     def __init__(self, model:str, max_retries: int = config.TASK_GRAPH_MAX_RETRIES):
#         self.max_retries: int = max_retries
#         self.nodes: dict[str, TaskNode] = {}
#         self.sys_instructions: str = load_prompt(agent_name='master')
#         self.llm: ChatOpenAI = ChatOpenAI(
#             model=model,
#             openai_api_key=OPENAI_API_KEY,
#             temperature=config.TEMPERATURE,
#             max_completion_tokens=config.MAX_COMPLETION_TOKENS,
#             timeout=config.TIMEOUT,
#             max_retries=config.LLM_MAX_RETRIES
#           )

#     def add_task(self, task: TaskNode, replace: bool = False):
#         """Adds a TaskNode to the graph.
#         If a task with the same id exists:
#           - if replace is True, overwrite the existing TaskNode,
#           - otherwise raise ValueError.
#         """
#         if not isinstance(task.node_id, str) or not task.node_id:
#             raise ValueError("task_id must be a non-empty string.")
#         if task.node_id in self.nodes:
#             if replace:
#                 self.nodes[task.node_id] = task
#                 print(f"Replaced existing Task with id '{task.node_id}'.")
#                 return
#             raise ValueError(f"Task with id '{task.node_id}' already exists. Pass replace=True to overwrite.")
#         self.nodes[task.node_id] = task

#     def _read_text_file(self, file_path: str) -> str:
#         """Reads a text-based file (txt, csv, json, etc.) into a string."""
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 return f.read()
#         except Exception as e:
#             print(f"Error reading {file_path}: {e}")
#             return f"Error reading file: {file_path}"
    
#     def _synthesis_dataset_info(self, filepath: str) -> str:
#         import io
#         import pandas as pd
#         df = pd.read_csv(filepath)

#         buffer = io.StringIO()

#         df.info(buf=buffer)

#         return buffer.getvalue()
    
#     def _collect_file_list_data(self, file_list: List[str]) -> List[Dict[str, Union[str, Dict[str, str]]]]:
#         import mimetypes
#         message_parts: List[Dict[str, Union[str, Dict[str, str]]]] = []

#         for file_path in file_list:
#             try:
#                 mime_type, _ = mimetypes.guess_type(file_path)
                
#                 if mime_type:
#                     file_type = mime_type.split('/')[0]
                    
#                     if file_type == 'image':
#                         # It's an image: encode it and add as image_url
#                         encoded_image = self.encode_image(file_path)
#                         message_parts.append({
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:{mime_type};base64,{encoded_image}"
#                             }
#                         })
#                     elif file_type == 'csv' in mime_type:
#                         try:
#                             dataset_info = self._synthesis_dataset_info(filepath=file_path)
#                             message_parts.append({
#                                 "type": "text",
#                                 "text": f"The dataset info: {dataset_info}"
#                             })
#                         except Exception as e:
#                             content = self._read_text_file(file_path)
#                             message_parts.append({
#                                 "type": "text",
#                                 "text": f"The dataset: {content[:1000]}"
#                             })
#                     elif file_type == 'text' or 'json' in mime_type:
#                         # It's a text file: read it and add as text
#                         content = self._read_text_file(file_path)
#                         # We wrap the content in markers for clarity
#                         formatted_content = f"""
# ---
# File Name: {file_path}
# Content:
# {content}
# ---
# """
#                         message_parts.append({
#                             "type": "text",
#                             "text": formatted_content
#                         })
#                     else:
#                         print(f"Warning: Skipping unsupported file type: {file_path} (MIME: {mime_type})")
#                 else:
#                     # Fallback for unknown extensions
#                     if file_path.endswith(('.txt', '.csv', '.json', '.py', '.md')):
#                         content = self._read_text_file(file_path)
#                         formatted_content = f"""
# ---
# File Name: {file_path}
# Content:
# {content}
# ---
# """
#                         message_parts.append({
#                             "type": "text",
#                             "text": formatted_content
#                         })
#                     else:
#                         print(f"Warning: Skipping unknown file type: {file_path}")
                        
#             except Exception as e:
#                 print(f"Error processing file {file_path}: {e}")

#         return message_parts

#     def _generate_task_graph(self, human_input:str, sess_id: str, file_list: List[str]):
#         """Generates a task graph"""
#         structured_llm = self.llm.with_structured_output(PydanticTaskGraph)

#         message_parts = self._collect_file_list_data(file_list=file_list)
#         messages = [
#             SystemMessage(content= self.sys_instructions),
#             HumanMessage(content=message_parts),
#             HumanMessage(content=f"The dataset info: {self._synthesis_dataset_info(sess_id=sess_id)}"),
#             HumanMessage(content=f"User request: {human_input}")
#         ]

#         pydantic_response: PydanticTaskGraph = structured_llm.invoke(messages) # type: ignore
#         # pydantic_response: PydanticTaskGraph = result.content # type: ignore

#         for pydantic_node in pydantic_response.task_nodes:
#             node_data = pydantic_node.model_dump()
#             node = TaskNode(
#                 node_id=node_data["task_id"],
#                 instruction=node_data["instruction"], 
#                 dependencies=node_data["dependencies"], 
#                 task_type=node_data["task_type"],
#                 output=node_data["output"],
#                 model=config.OPENAI_MODEL
#             )
#             self.add_task(node)
    
#     def _refine_and_update_task_graph(self, add_instructions:str):
#         # TODO
#         raise RuntimeError(f'failed to run Task Graph: {add_instructions}')

#     def initialize_and_populate_task_graph(self, human_input:str, global_agent_state: GlobalAgentState, file_list: List[str], progress_callback: Union[callable, None] = None):
#         self._generate_task_graph(human_input=human_input, sess_id=global_agent_state['sess_id'], file_list=file_list)
#         increase_num_steps(global_agent_state)
#         print(f'== Task Graph created with {len(self.nodes)} Nodes.')
#         if progress_callback:
#             progress_callback(f'Task Graph created with {len(self.nodes)} Nodes.')
#         for node in self.nodes.values():
#             print(f'==== Initializing Node {node.node_id}')
#             if progress_callback:
#                 progress_callback(f'Initializing Node {node.node_id}')
#             node.iterative_refining(global_agent_state=global_agent_state)
#             if node.status is TaskStatus.FAILED:
#                 print(f'Iterate refining for Task Node {node.node_id} failed to run in limit {self.max_retries}: {node.action_graph.result}')
#                 if progress_callback:
#                     progress_callback(f'Iterate refining for Task Node {node.node_id} failed to run in limit {self.max_retries}: {node.action_graph.result}')
#                 self._refine_and_update_task_graph(node.result or '')

#     def _get_execution_order(self) -> list[str]:
#         """
#         Determines the execution order of tasks using topological sort.
#         This is crucial for executing tasks in the correct sequence based on their dependencies.
#         """
#         graph_representation = {
#             task_id: node.dependencies for task_id, node in self.nodes.items()
#         }
        
#         try:
#             ts = TopologicalSorter(graph_representation)
#             return list(ts.static_order())
#         except CycleError as e:
#             print(f"Error: A cycle was detected in the task graph. Cannot determine execution order. Details: {e}")
#             return []

#     def print_graph(self, sess_id:str, run_id: str, verbose: bool = False, ensure_dir=True, filename: str = 'code'):
#         """Prints a summary of all tasks and their dependencies."""
#         if not self.nodes:
#             print("Graph is empty.")
#             return
        
#         if verbose:
#             try:
#                 dest_dir = os.path.join(config.DATA_FILEPATH,sess_id,run_id)
#                 if ensure_dir:
#                     dest_dir.mkdir(parents=True, exist_ok=True)
#                 filepath = dest_dir / filename
#                 with open(filepath, "w", encoding="utf-8") as f:
                    
#                     f.write(f"# RUN ID -- {run_id}\n\n")
#                     for task_id, node in self.nodes.items():
#                         f.write(f"# --- Task {task_id} ---\n")
#                         f.write(f"# Instruction: {node.instruction}\n")
#                         # if node has an action_graph, write each action's code
#                         if getattr(node, 'action_graph', None) and node.action_graph.nodes:
#                             for action in node.action_graph.nodes:
#                                 f.write(f"# Action {action.action_id}: {action.description}\n")
#                                 f.write(action.code + "\n\n")
#                         else:
#                             f.write("# No action graph available\n\n")
#                 print(f"Saved action graphs codes to {filepath}")
#             except Exception as e:
#                 print(f"Failed to write action graphs to file: {e}")
        
#         else:
#             print("--- Task Graph ---")
#             for task_id, node in self.nodes.items():
#                 print(f"  - ID: {task_id}, Status: {node.status.value}")
#                 print(f"    Instruction: {node.instruction}")
#                 print(f"    Dependencies: {node.dependencies or 'None'}")
#             print("--------------------")

#     def run_workflow(self, agent_state: GlobalAgentState, stop_on_failure: bool = True) -> GlobalAgentState:
#         """
#         Run tasks in topological order. For each TaskNode:
#           - evaluate optional `condition` (simple eval with limited globals),
#           - execute via TaskNode.iterate_refining(llm, agent_state),
#           - update agent_state from action_graph result if present.
#         Returns final agent_state dict.
#         """
#         # ensure order
#         order = self._get_execution_order()
#         if not order:
#             raise RuntimeError("No execution order (empty graph or cycle detected).")

#         current_state: GlobalAgentState = copy.deepcopy(agent_state)

#         for tid in order:
#             if tid not in self.nodes:
#                 print(f"Skipping unknown task id {tid}")
#                 continue

#             node = self.nodes[tid]
#             print(f"== Running task {tid}")

#             try:
#                 current_state = node.run_action_graph(agent_state=current_state)
#                 print(f'Current State: {current_state}')
#             except Exception as e:
#                 print(f"Exception when running task {tid}: {type(e).__name__}: {e}")
#                 node.status = TaskStatus.FAILED
#                 if stop_on_failure:
#                     break
#                 else:
#                     continue

#             if node.status == TaskStatus.FAILED and stop_on_failure:
#                 print("Stopping workflow due to failure.")
#                 break

#         return current_state

# def select_task_node(task_graph: TaskGraph) -> TaskNode | None:
#     """
#     Select a task node with PENDING status from the task graph.
#     Returns the first pending task found, or None if no pending tasks exist.
#     """
#     for task in task_graph.nodes.values():
#         if task == TaskStatus.PENDING:
#             return task
    
#     return None

# def is_graph_finished(task_graph: TaskGraph) -> bool:
#     """
#     Checks if there is any pending nodes in graph.
#     Returns boolean value to indicate the graph condition.
#     """
#     for task in task_graph.nodes.values():
#         if task.status == TaskStatus.PENDING:
#             return False
    
#     return True
  
# def write_response_txt(response: str, sess_id: str, run_id: str, filename: str = "response.txt", ensure_dir: bool = True) -> str:
#     """
#     Save `response` as a text file under {config.DATA_FILEPATH}{sess_id}/{run_id}/filename.
#     Returns the written file path as string.
#     """
#     from pathlib import Path

#     dest_dir = Path(os.path.join(config.SESSION_FILEPATH,sess_id,run_id))
#     if ensure_dir:
#         dest_dir.mkdir(parents=True, exist_ok=True)

#     filepath = dest_dir / filename
#     with open(filepath, "w", encoding="utf-8") as f:
#         f.write(response)

#     print(f"Saved response to {filepath}")
#     return str(filepath)

# from pydantic import BaseModel, Field

# class PydanticMasterResult(BaseModel):
#     """Represents a structured message from an agent to the shared state."""
#     response: str = Field(..., description="analysis result summary to return to user")
    
# class MasterAgent:
#     def __init__(self, model:str, tools=[], max_retries:int = 3):

#         self.max_retries = max_retries
#         self.llm = ChatOpenAI(
#             model=model,
#             openai_api_key=OPENAI_API_KEY,
#             temperature=config.TEMPERATURE,
#             max_completion_tokens=config.MAX_COMPLETION_TOKENS,
#             timeout=config.TIMEOUT,
#             max_retries=config.LLM_MAX_RETRIES
#           )
#         self.tools = tools
#         self.instructions_ans = load_prompt(agent_name='master', key="system_prompt_ans")
#         self.instructions_user_req = load_prompt(agent_name='master', key="system_prompt_user_req")
#         self.task_graph: TaskGraph = TaskGraph(model=config.OPENAI_MODEL)
#         self.analysis_agent = AnalysisAgent(model=config.OPENAI_MODEL)
    

#     def _collect_visualization_paths(self, sess_id:str, run_id:str) -> list[str]:
#         """
#         Collect all image file paths from the figures directory.
        
#         Args:
#             figures_dir: Directory containing visualization files
            
#         Returns:
#             List of file paths as strings
#         """
#         import glob
#         from pathlib import Path
#         # Ensure directory exists
#         figures_dir = Path(os.path.join(config.SESSION_FILEPATH, sess_id, run_id, config.FIGURE_FILEPATH))
#         if not os.path.exists(figures_dir):
#             return []
        
#         # Common image extensions
#         image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf']
        
#         image_paths = []
#         for ext in image_extensions:
#             pattern = os.path.join(figures_dir, ext)
#             image_paths.extend(glob.glob(pattern))
        
#         # Sort for consistent ordering
#         return sorted(image_paths)

#     def generate_id(self, prefix: str | None = None, uuid_len: int = config.UUID_LEN) -> str:
#         import uuid
#         """
#         Generate a shorter run ID using first 8 characters of UUID.
        
#         Args:
#             prefix: Prefix for the run ID (default: "run")
            
#         Returns:
#             Generated run ID string
            
#         Example:
#             - generate_run_id_short() -> "run_a1b2c3d4"
#         """
#         if not prefix:
#             return uuid.uuid4().hex[:uuid_len]
#         return f"{prefix}_{uuid.uuid4().hex[:uuid_len]}"

#     def _write_agent_state_to_json(self, agent_state: GlobalAgentState, ensure_dir: bool = True, indent: int = 1) -> None:
#         """
#         Write agent_state (GraphState or plain dict) to a JSON file.
#         Non-serializable values are converted via str().
#         """
#         from pathlib import Path

#         import json, os
#         run_dir = Path(os.path.join(config.DATA_FILEPATH,agent_state['sess_id'],agent_state['run_id']))

#         if ensure_dir:
#             dirpath = os.path.dirname(run_dir) or "."
#             os.makedirs(dirpath, exist_ok=True)

#         filepath = os.path.join(run_dir, 'agent-state.json')

#         try:
#             with open(filepath, "w", encoding="utf-8") as f:
#                 json.dump(agent_state, f, ensure_ascii=False, indent=indent, default=str)
#             print(f"Agent State stored in {filepath}")
#         except Exception as e:
#             print(f"Failed to save agent state: {e}")


#     def _refine_task_graph_on_additional_request(self):
#         # TODO
#         raise RuntimeError('Need dev')
    
#     def _initialize_agent_state(self, sess_id:str, run_id: str, requirement:str, data_path:str) -> GlobalAgentState:
#         state_dict = {'sess_id':sess_id, 'run_id':run_id, 'requirement':requirement, 'num_steps':0, 'raw_data_path':[data_path], 'evaluation_results':[], 'visualization_paths':[], 'agent_messages':[]}
#         state: GlobalAgentState = state_dict # type: ignore
#         return state
    
#     def _provide_answer(self, state: GlobalAgentState) -> str:
#         messages = [
#             SystemMessage(content= self.instructions_ans),
#             AIMessage(content=f'evaluation results:{str(state["evaluation_results"])}'),
#             HumanMessage(content=f'user question: {str(state["requirement"])}'),
#         ]

#         response_str: str = self.llm.invoke(messages).text()

#         return response_str
    
#     def _summarize_user_request(self, human_req: str) -> str:
#         messages = [
#             SystemMessage(content= self.instructions_user_req),
#             HumanMessage(content=f'user request:{human_req}'),
#         ]

#         response = self.llm.invoke(messages)
#         return response.text
    
#     def _migrate_all_current_run_data(self, sess_id:str, run_id:str, create_dest: bool = True):
#         """
#         Move all current run files {config.SESSION_FILEPATH}/{sess_id}/{run_id}.
        
#         Args:
#             sess_id: Session ID
#             run_id: Run ID
#             create_dest: Create destination directory if it doesn't exist (default: True)
            
#         Returns:
#             Dictionary with results: {'success': int, 'failed': int, 'errors': list}
#         """
#         import shutil
#         from pathlib import Path
#         results = {'success': 0, 'failed': 0, 'errors': []}
    
#         # Convert to Path objects
#         source_dir = Path(os.path.join(config.TEMP_FILEPATH, sess_id))
#         dest_dir = Path(os.path.join(config.SESSION_FILEPATH, sess_id, run_id))
        
#         # Check if source exists
#         if not source_dir.exists():
#             raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
        
#         if not source_dir.is_dir():
#             raise NotADirectoryError(f"Source path is not a directory: {source_dir}")
        
#         # Create destination if needed
#         if create_dest:
#             dest_dir.mkdir(parents=True, exist_ok=True)
#         elif not dest_dir.exists():
#             raise FileNotFoundError(f"Destination directory does not exist: {dest_dir}")
        
#         # Move all folders/files
#         for item_name in os.listdir(source_dir):
#             try:
#                 source_path = os.path.join(source_dir, item_name)
#                 target_path = os.path.join(dest_dir, item_name)
#                 shutil.move(source_path, target_path)
#                 results['success'] += 1
#             except Exception as e:
#                 results['failed'] += 1
#                 results['errors'].append(f"{item_name}: {str(e)}")
        
#         print(f"\nSummary: {results['success']} files moved, {results['failed']} failed")
#         return results

#     def process_requirement(self, human_input:str, file_list: List[str], sess_id: str, progress_callback: Union[callable, None]) -> str:
#         run_id = self.generate_id(prefix='run')
#         print(f"== Run ID: {run_id} ==\n")

#         # Summarize user request
#         # user_request = self._summarize_user_request(human_req=human_input)
#         user_request = human_input

#         state: GlobalAgentState = self._initialize_agent_state(sess_id=sess_id, run_id=run_id, requirement=user_request, data_path=f'tmp/{sess_id}/data/data.csv')

#         if len(self.task_graph.nodes) == 0:
#             print('Initiating TaskGraph')
#             if progress_callback:
#                 progress_callback(f'Initiating TaskGraph')
#             self.task_graph.initialize_and_populate_task_graph(global_agent_state=state, human_input=user_request, file_list=file_list, progress_callback=progress_callback)
#         else:
#             print('Starting to refine TaskGraph')
#             if progress_callback:
#                 progress_callback(f'Starting to refine TaskGraph')
#             self._refine_task_graph_on_additional_request()


#         self.task_graph.print_graph(sess_id=state['sess_id'], run_id=state['run_id'], verbose=True)
#         workflow_state = self.task_graph.run_workflow(agent_state=state)

#         self._migrate_all_current_run_data(sess_id=state['sess_id'], run_id=state['run_id'])
#         # Collect all diagrams under data/figures
#         visualisation_paths = self._collect_visualization_paths(sess_id=state['sess_id'], run_id=state['run_id'])
#         workflow_state['visualization_paths'] = visualisation_paths
        
#         print("== Analysis in Progress")
#         if progress_callback:
#             progress_callback(f'Analysis in Progress')
#         final_state = self.analysis_agent.analyze_all_diagrams(state=workflow_state, prompt=f'Give insights on these diagrams regarding user request:{workflow_state["requirement"]}')
#         self._write_agent_state_to_json(agent_state=final_state)

#         print("== Fabricating Final Answer")
#         if progress_callback:
#             progress_callback(f'Fabricating Final Answer')
#         response_str = self._provide_answer(state=final_state)
#         write_response_txt(response=str(response_str), sess_id=state['sess_id'], run_id=state['run_id'])

#         return response_str
    
#     def process_demo(self, human_input:str, file_path: str, sess_id: str, progress_callback: Union[callable, None] = None) -> str:
#         from time import sleep
#         if progress_callback:
#             progress_callback(f"Initiating Task")
#             sleep(10)
#             progress_callback(f'Analysing Diagrams')
#             sleep(3)
#             progress_callback(f'Fabricating Final Answer')
#             sleep(3)

#         return '''Short summary of what the data already suggests
# - There is an average negative association between gold price and traded volume: higher volumes tend to occur with lower prices.  
# - The relationship is weak and unstable: scatter points are widely spread with clusters and outliers, residuals from a simple linear fit show curvature and changing spread, and residuals have heavy tails.  
# - Time structure matters: price trends upward while volume tends downward with spikes. The price–volume link appears to change over time (nonstationarity / regime changes).
# '''
#         # return read_text_file("test-response.txt")
    

# def get_master_agent():
#     return MasterAgent(model=config.OPENAI_MODEL)






