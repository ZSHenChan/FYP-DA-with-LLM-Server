import base64, logging, copy
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import config
from .schemas import GlobalAgentState, AgentMessage, PydanticDiagramResult, PydanticAnalysisResult
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
    
    def analyze_diagram(self, prompt:str, file_path_list: List[str]) -> PydanticAnalysisResult:
        """
        Encodes the image and asks the vision model a question about it.
        """
        import os
        
        image_content = []
        for path in file_path_list:
            filename = os.path.basename(path)
            image_content.extend([
                {
                    "type": "text", 
                    "text": f"Image Filename: {filename}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode_image(path)}"
                    }
                }
            ])

        messages = [
            SystemMessage(content=self.instruction),
            # Combine the user's specific question with the labeled images
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                *image_content  # Unpack the list of text and image blocks
            ])
        ]

        structured_llm = self.llm.with_structured_output(PydanticAnalysisResult)

        pydantic_response: PydanticAnalysisResult = structured_llm.invoke(messages) # type: ignore

        return pydantic_response
    
    def analyze_all_diagrams(self, prompt:str, state: GlobalAgentState) -> GlobalAgentState:
        
        diagram_list: List[str] = state['visualization_paths']
        logger.info(f'== Analyzing {len(diagram_list)} diagrams...')

        state_copy = copy.deepcopy(state)

        analysis_result: AgentMessage = {'sender': 'Analysis Agent', 'content': []}

        # for diagram_path in diagram_list:
        analysis_result_dict: PydanticAnalysisResult = self.analyze_diagram(prompt=prompt, file_path_list=diagram_list)

        state_copy['analysis_result'] = analysis_result_dict
        return state_copy
       