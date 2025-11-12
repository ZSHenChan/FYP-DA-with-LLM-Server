import base64, logging, copy
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import config
from .schemas import PydanticAnalysisResult, GlobalAgentState, AgentMessage
from .utils import load_prompt

OPENAI_API_KEY = config.OPENAI_API_KEY

logger = logging.getLogger(config.SESS_LOG_NAME)

class AnalysisAgent:
    """A class to perform analysis based on code outputs."""
    def __init__(self, model:str):
        self.instruction = load_prompt(agent_name='analysis')
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=OPENAI_API_KEY,
            temperature=config.TEMPERATURE,
            max_completion_tokens=config.MAX_COMPLETION_TOKENS,
            timeout=config.TIMEOUT,
            max_retries=config.LLM_MAX_RETRIES
        )
    
    def _encode_image(self, image_path: str):
        """Helper function to encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_diagram(self, prompt:str, file_path_list: List[str]) -> str:
        """
        Encodes the image and asks the vision model a question about it.
        """
       
        messages = [
            SystemMessage(content= self.instruction),
            HumanMessage(content=prompt),
            HumanMessage(content=[{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode_image(x)}"
                    }
                } for x in file_path_list])
        ]

        structured_llm = self.llm.with_structured_output(PydanticAnalysisResult)

        pydantic_response: PydanticAnalysisResult = structured_llm.invoke(messages) # type: ignore

        response_str: str = pydantic_response.result

        return response_str
    
    def analyze_all_diagrams(self, prompt:str, state: GlobalAgentState) -> GlobalAgentState:
        
        diagram_list: List[str] = state['visualization_paths']
        logger.info(f'== Analyzing {len(diagram_list)} diagrams...')

        state_copy = copy.deepcopy(state)

        analysis_result: AgentMessage = {'sender': 'Analysis Agent', 'content': []}

        # for diagram_path in diagram_list:
        analysis_result_str = self.analyze_diagram(prompt=prompt, file_path_list=diagram_list)

        analysis_result['content'].append({'analysis_result': analysis_result_str})

        if not state_copy.get('evaluation_results'):
            state_copy['evaluation_results'] = []
        state_copy['evaluation_results'].append(analysis_result)
        return state_copy
       