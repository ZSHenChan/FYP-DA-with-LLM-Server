import os
import copy
from graphlib import TopologicalSorter, CycleError
from typing import List, Dict, Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import config
from .schemas import (
    GlobalAgentState, PydanticActionGraph, TaskStatus, 
    TaskType, PydanticTaskGraph
)
from .utils import CodeExecutor, ExecuteResult, load_prompt, increase_num_steps

OPENAI_API_KEY = config.OPENAI_API_KEY

class ActionNode:
    """Represents a single executable code snippet within a task."""
    def __init__(self, action_id: int, code: str, description: str):
        self.action_id = action_id
        self.description = description
        self.code = code
        self.status = TaskStatus.PENDING
        self.result = None

    def __repr__(self):
        return f"ActionNode(id={self.action_id}, description='{self.description}', status='{self.status.value}', code='{self.code[:30]}...')"

class ActionGraph:
    """Manages the sequence of actions for a single parent task."""
    def __init__(self):
        self.nodes: List[ActionNode] = []
        self.result: ExecuteResult | None = None

    def add_action(self, node: ActionNode):
        self.nodes.append(node)
        self.nodes.sort(key=lambda n: n.action_id)
        
    def __repr__(self):
        return f"ActionGraph with {len(self.nodes)} actions."
    
    def execute_action_graph(self, namespace: dict):
        executor = CodeExecutor(namespace=namespace)
        last_message = ''
        for current_action_node in self.nodes:
            exec_success, result = executor.execute(current_action_node.code)
            if not exec_success:
                current_action_node.status = TaskStatus.FAILED
                self.result = ExecuteResult(success=False, message=result, namespace=executor.namespace)
                return
            current_action_node.status = TaskStatus.SUCCESS
            last_message = result
        
        self.result = ExecuteResult(success=True, message=last_message, namespace=executor.namespace)

    def print_actions(self):
        for action in self.nodes:
            print(f"  - ID: {action.action_id}")
            print(f"    Description: {action.description}")
            print(f"    Code: {action.code}")
            print("--------------------")

class TaskNode:
    """Represents a single node in the task graph."""
    def __init__(self, node_id: str, instruction: str, dependencies: list[str], task_type: TaskType, output: str, model: str):
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("task_id must be a non-empty string.")
        
        self.node_id = node_id
        self.instruction = instruction
        self.dependencies = dependencies
        self.status = TaskStatus.PENDING
        self.task_type = task_type
        self.output = output
        self.result: str | None = None
        self.action_graph = ActionGraph()
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=OPENAI_API_KEY,
            temperature=config.TEMPERATURE,
            max_completion_tokens=config.MAX_COMPLETION_TOKENS,
            timeout=config.TIMEOUT,
            max_retries=config.LLM_MAX_RETRIES
        )

    def __repr__(self):
        """Provides a string representation for the task node."""
        return (f"TaskNode(id='{self.node_id}', status='{self.status.value}', "
                f"instruction='{self.instruction[:30]}...', deps={self.dependencies})")

    def generate_action_graph(self, global_agent_state: GlobalAgentState, namespace: dict, tool_sets=[], additional_instruction: str = "" ) -> ActionGraph:
        structured_llm = self.llm.with_structured_output(PydanticActionGraph)
        data_dir = os.path.join(config.TEMP_FILEPATH, global_agent_state['sess_id'], config.DATA_FILEPATH)
        figure_dir = os.path.join(config.TEMP_FILEPATH, global_agent_state['sess_id'], config.FIGURE_FILEPATH)
        model_dir = os.path.join(config.TEMP_FILEPATH, global_agent_state['sess_id'], config.MODEL_FILEPATH)
        
        sys_prompt_template = load_prompt(agent_name='code')
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt_template),
            ("human", "Instruction: {instruction}. Additional Instruction: {additional_instruction}. Agent State: {agent_state}. namespace: {namespace}")
        ])
        chain = prompt | structured_llm
        
        pydantic_action_graph: PydanticActionGraph = chain.invoke({"instruction": self.instruction, "additional_instruction":additional_instruction, "agent_state":global_agent_state, "namespace":namespace, "data_dir":data_dir, "figure_dir":figure_dir, "model_dir":model_dir}) # type: ignore
        
        action_graph = ActionGraph()
        for p_node in pydantic_action_graph.task_nodes:
            action_node = ActionNode(
                action_id=p_node.action_id,
                description=p_node.description,
                code=p_node.code
            )
            action_graph.add_action(action_node)
        
        increase_num_steps(global_agent_state)
        return action_graph

    def refine_and_update_action_graph(self, global_agent_state: GlobalAgentState, namespace: dict):
        refined_graph: ActionGraph = self.generate_action_graph(global_agent_state=global_agent_state, namespace=namespace, additional_instruction=f"A few errors were encountered when running the actions one-by-one. The current ActionGraph is {str(self.action_graph.nodes)}. You need to debug the code using the error message: {str(self.action_graph.result)}")
        self.action_graph.nodes = refined_graph.nodes
        self.action_graph.result = None

    def iterative_refining(self, global_agent_state: GlobalAgentState, max_retries: int = config.TASK_NODE_MAX_RETRIES):
        """Generate and refine the ActionGraph within a trial limit."""
        debug_agent_state: GlobalAgentState = copy.deepcopy(global_agent_state)
        
        namespace = {'agent_state':{}}

        action_graph: ActionGraph = self.generate_action_graph(global_agent_state=debug_agent_state, namespace=namespace)
        self.action_graph = action_graph
        for _ in range(max_retries):
            namespace = {'agent_state':{}}
            self.action_graph.execute_action_graph(namespace)
            if self.action_graph.result is None:
                raise Exception("Action Graph Result is None")
            if self.action_graph.result.success is False:
                self.refine_and_update_action_graph(global_agent_state=debug_agent_state, namespace=namespace)
                increase_num_steps(global_agent_state)
            else:
                self.status = TaskStatus.SUCCESS
                break
        if self.action_graph.result is not None and self.action_graph.result.success is False:
            self.status = TaskStatus.FAILED
            self.result = str(self.action_graph.result)

    def run_action_graph(self, agent_state: GlobalAgentState) -> GlobalAgentState:
        namespace = {'agent_state':{}}
        agent_state_copy = copy.deepcopy(agent_state)

        self.action_graph.execute_action_graph(namespace)
        if self.action_graph.result is not None and self.action_graph.result.success:
            # Update current agent work to Global Agent State
            final_state = self.action_graph.result.namespace.get('agent_state',{})
            if final_state:
                agent_state_copy['agent_messages'].append({'sender': self.task_type + ' agent', 'content':final_state})
            return agent_state_copy
        else:
            raise RuntimeError(f'Failed to run ActionGraph {self.node_id}: {self.action_graph.result}')

class TaskGraph:
    """Manages the entire Directed Acyclic Graph (DAG) of tasks."""
    def __init__(self, model:str, max_retries: int = config.TASK_GRAPH_MAX_RETRIES):
        self.max_retries: int = max_retries
        self.nodes: dict[str, TaskNode] = {}
        self.sys_instructions: str = load_prompt(agent_name='master')
        self.llm: ChatOpenAI = ChatOpenAI(
            model=model,
            openai_api_key=OPENAI_API_KEY,
            temperature=config.TEMPERATURE,
            max_completion_tokens=config.MAX_COMPLETION_TOKENS,
            timeout=config.TIMEOUT,
            max_retries=config.LLM_MAX_RETRIES
          )

    def add_task(self, task: TaskNode, replace: bool = False):
        """Adds a TaskNode to the graph.
        If a task with the same id exists:
          - if replace is True, overwrite the existing TaskNode,
          - otherwise raise ValueError.
        """
        if not isinstance(task.node_id, str) or not task.node_id:
            raise ValueError("task_id must be a non-empty string.")
        if task.node_id in self.nodes:
            if replace:
                self.nodes[task.node_id] = task
                print(f"Replaced existing Task with id '{task.node_id}'.")
                return
            raise ValueError(f"Task with id '{task.node_id}' already exists. Pass replace=True to overwrite.")
        self.nodes[task.node_id] = task

    def _read_text_file(self, file_path: str) -> str:
        """Reads a text-based file (txt, csv, json, etc.) into a string."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return f"Error reading file: {file_path}"
    
    def _synthesis_dataset_info(self, filepath: str) -> str:
        import io
        import pandas as pd
        df = pd.read_csv(filepath)

        buffer = io.StringIO()

        df.info(buf=buffer)

        return buffer.getvalue()
    
    def _collect_file_list_data(self, file_list: List[str]) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        import mimetypes
        message_parts: List[Dict[str, Union[str, Dict[str, str]]]] = []

        for file_path in file_list:
            try:
                mime_type, _ = mimetypes.guess_type(file_path)
                
                if mime_type:
                    file_type = mime_type.split('/')[0]
                    
                    if file_type == 'image':
                        # It's an image: encode it and add as image_url
                        encoded_image = self.encode_image(file_path)
                        message_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_image}"
                            }
                        })
                    elif file_type == 'csv' in mime_type:
                        try:
                            dataset_info = self._synthesis_dataset_info(filepath=file_path)
                            message_parts.append({
                                "type": "text",
                                "text": f"The dataset info: {dataset_info}"
                            })
                        except Exception as e:
                            content = self._read_text_file(file_path)
                            message_parts.append({
                                "type": "text",
                                "text": f"The dataset: {content[:1000]}"
                            })
                    elif file_type == 'text' or 'json' in mime_type:
                        # It's a text file: read it and add as text
                        content = self._read_text_file(file_path)
                        # We wrap the content in markers for clarity
                        formatted_content = f"""
---
File Name: {file_path}
Content:
{content}
---
"""
                        message_parts.append({
                            "type": "text",
                            "text": formatted_content
                        })
                    else:
                        print(f"Warning: Skipping unsupported file type: {file_path} (MIME: {mime_type})")
                else:
                    # Fallback for unknown extensions
                    if file_path.endswith(('.txt', '.csv', '.json', '.py', '.md')):
                        content = self._read_text_file(file_path)
                        formatted_content = f"""
---
File Name: {file_path}
Content:
{content}
---
"""
                        message_parts.append({
                            "type": "text",
                            "text": formatted_content
                        })
                    else:
                        print(f"Warning: Skipping unknown file type: {file_path}")
                        
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        return message_parts

    def _generate_task_graph(self, human_input:str, sess_id: str, file_list: List[str]):
        """Generates a task graph"""
        structured_llm = self.llm.with_structured_output(PydanticTaskGraph)

        message_parts = self._collect_file_list_data(file_list=file_list)
        messages = [
            SystemMessage(content= self.sys_instructions),
            HumanMessage(content=message_parts),
            HumanMessage(content=f"The dataset info: {self._synthesis_dataset_info(sess_id=sess_id)}"),
            HumanMessage(content=f"User request: {human_input}")
        ]

        pydantic_response: PydanticTaskGraph = structured_llm.invoke(messages) # type: ignore
        # pydantic_response: PydanticTaskGraph = result.content # type: ignore

        for pydantic_node in pydantic_response.task_nodes:
            node_data = pydantic_node.model_dump()
            node = TaskNode(
                node_id=node_data["task_id"],
                instruction=node_data["instruction"], 
                dependencies=node_data["dependencies"], 
                task_type=node_data["task_type"],
                output=node_data["output"],
                model=config.OPENAI_MODEL
            )
            self.add_task(node)
    
    def _refine_and_update_task_graph(self, add_instructions:str):
        # TODO
        raise RuntimeError(f'failed to run Task Graph: {add_instructions}')

    def initialize_and_populate_task_graph(self, human_input:str, global_agent_state: GlobalAgentState, file_list: List[str], progress_callback: Union[callable, None] = None):
        self._generate_task_graph(human_input=human_input, sess_id=global_agent_state['sess_id'], file_list=file_list)
        increase_num_steps(global_agent_state)
        print(f'== Task Graph created with {len(self.nodes)} Nodes.')
        if progress_callback:
            progress_callback(f'Task Graph created with {len(self.nodes)} Nodes.')
        for node in self.nodes.values():
            print(f'==== Initializing Node {node.node_id}')
            if progress_callback:
                progress_callback(f'Initializing Node {node.node_id}')
            node.iterative_refining(global_agent_state=global_agent_state)
            if node.status is TaskStatus.FAILED:
                print(f'Iterate refining for Task Node {node.node_id} failed to run in limit {self.max_retries}: {node.action_graph.result}')
                if progress_callback:
                    progress_callback(f'Iterate refining for Task Node {node.node_id} failed to run in limit {self.max_retries}: {node.action_graph.result}')
                self._refine_and_update_task_graph(node.result or '')

    def _get_execution_order(self) -> list[str]:
        """
        Determines the execution order of tasks using topological sort.
        This is crucial for executing tasks in the correct sequence based on their dependencies.
        """
        graph_representation = {
            task_id: node.dependencies for task_id, node in self.nodes.items()
        }
        
        try:
            ts = TopologicalSorter(graph_representation)
            return list(ts.static_order())
        except CycleError as e:
            print(f"Error: A cycle was detected in the task graph. Cannot determine execution order. Details: {e}")
            return []

    def print_graph(self, sess_id:str, run_id: str, verbose: bool = False, ensure_dir=True, filename: str = 'code'):
        """Prints a summary of all tasks and their dependencies."""
        if not self.nodes:
            print("Graph is empty.")
            return
        
        if verbose:
            try:
                dest_dir = os.path.join(config.DATA_FILEPATH,sess_id,run_id)
                if ensure_dir:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                filepath = dest_dir / filename
                with open(filepath, "w", encoding="utf-8") as f:
                    
                    f.write(f"# RUN ID -- {run_id}\n\n")
                    for task_id, node in self.nodes.items():
                        f.write(f"# --- Task {task_id} ---\n")
                        f.write(f"# Instruction: {node.instruction}\n")
                        # if node has an action_graph, write each action's code
                        if getattr(node, 'action_graph', None) and node.action_graph.nodes:
                            for action in node.action_graph.nodes:
                                f.write(f"# Action {action.action_id}: {action.description}\n")
                                f.write(action.code + "\n\n")
                        else:
                            f.write("# No action graph available\n\n")
                print(f"Saved action graphs codes to {filepath}")
            except Exception as e:
                print(f"Failed to write action graphs to file: {e}")
        
        else:
            print("--- Task Graph ---")
            for task_id, node in self.nodes.items():
                print(f"  - ID: {task_id}, Status: {node.status.value}")
                print(f"    Instruction: {node.instruction}")
                print(f"    Dependencies: {node.dependencies or 'None'}")
            print("--------------------")

    def run_workflow(self, agent_state: GlobalAgentState, stop_on_failure: bool = True) -> GlobalAgentState:
        """
        Run tasks in topological order. For each TaskNode:
          - evaluate optional `condition` (simple eval with limited globals),
          - execute via TaskNode.iterate_refining(llm, agent_state),
          - update agent_state from action_graph result if present.
        Returns final agent_state dict.
        """
        # ensure order
        order = self._get_execution_order()
        if not order:
            raise RuntimeError("No execution order (empty graph or cycle detected).")

        current_state: GlobalAgentState = copy.deepcopy(agent_state)

        for tid in order:
            if tid not in self.nodes:
                print(f"Skipping unknown task id {tid}")
                continue

            node = self.nodes[tid]
            print(f"== Running task {tid}")

            try:
                current_state = node.run_action_graph(agent_state=current_state)
                print(f'Current State: {current_state}')
            except Exception as e:
                print(f"Exception when running task {tid}: {type(e).__name__}: {e}")
                node.status = TaskStatus.FAILED
                if stop_on_failure:
                    break
                else:
                    continue

            if node.status == TaskStatus.FAILED and stop_on_failure:
                print("Stopping workflow due to failure.")
                break

        return current_state
