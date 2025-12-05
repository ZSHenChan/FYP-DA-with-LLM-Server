from enum import Enum
from typing import Optional, TypedDict, List, Any
from pydantic import BaseModel, Field

# --- TypedDicts ---

class AgentMessage(TypedDict):
    """Represents a structured message from an agent to the shared state."""
    sender: str
    content: Any

class GlobalAgentState(TypedDict):
    """A comprehensive state for a data science pipeline."""
    sess_id: str
    run_id: str
    requirement: str
    num_steps: int
    raw_data_filenames: List[str]
    evaluation_results: List[AgentMessage]
    visualization_paths: List[str]
    agent_messages: List[AgentMessage]

# --- Enums ---

class TaskStatus(str, Enum):
    """Enum for task status values."""
    SUCCESS = "success"
    FAILED = "failed" 
    PENDING = "pending"

class TaskType(str, Enum):
    """Enum for task type values."""
    DATA_LOADING = "data_loading"
    EXPLORATION = "exploration"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    EVALUATION = "evaluation"
    VISUALIZATION = "visualization"

# --- Pydantic Schemas ---

class PydanticAnalysisResult(BaseModel):
    result: str = Field(..., description="analysis result")
    
class PydanticActionNode(BaseModel):
    action_id: int = Field(..., description="The sequential ID...")
    description: str = Field(..., description="A brief, natural language...")
    code: str = Field(..., description="A valid, executable snippet...")

class PydanticActionGraph(BaseModel):
    task_nodes: List[PydanticActionNode] = Field(default=[] ,  description="List of task nodes for the process.")

class PydanticTaskNode(BaseModel):
    task_id: str = Field(..., description="Unique id for the task...")
    dependencies: List[str] = Field(..., description="A list of unique ids...")
    instruction: str = Field(..., description="A concise instruction...")
    task_type: TaskType = Field(description="Current status of the task")
    output: str = Field(..., description="description of what data...")

class PydanticTaskGraph(BaseModel):
    task_nodes: List[PydanticTaskNode] = Field(..., description="List of task nodes...")

class PydanticEditAction(str, Enum):
    ADD = "add"       # Add a completely new task
    MODIFY = "modify" # Change instruction/dependencies of a PENDING task
    DELETE = "delete" # Remove a task (and handle its children)

class PydanticGraphEdit(BaseModel):
    action: PydanticEditAction = Field(..., description="The type of change to apply.")
    # For DELETE, we only need the ID.
    # For ADD/MODIFY, we need the full node details.
    task: Optional[PydanticTaskNode] = Field(None, description="The task details. Required for ADD and MODIFY.")
    target_task_id: Optional[str] = Field(None, description="The specific ID of the task to DELETE.")

class PydanticGraphModificationPlan(BaseModel):
    reasoning: str = Field(..., description="Brief explanation of why these changes meet the new requirement.")
    edits: List[PydanticGraphEdit] = Field(..., description="List of atomic edits to apply to the graph.")

class PydanticMasterResult(BaseModel):
    response: str = Field(..., description="analysis result summary...")