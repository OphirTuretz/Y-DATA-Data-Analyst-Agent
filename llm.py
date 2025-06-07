from app.const import MODEL_NAME, BASE_URL, TEMPERATURE, TOP_P
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()  # Load environment variables from .env file


class LLM:

    @staticmethod
    def perform_request(
        messages: List[dict],
        base_url: str = BASE_URL,
        model: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        tools: List[dict] = None,
        tool_choice: str = None,
        parallel_tool_calls: bool = False,
    ):

        # Initialize OpenAI client
        client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("NEBIUS_STUDIO_API_KEY"),
        )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            temperature=temperature,
            top_p=top_p,
        )

        return response
