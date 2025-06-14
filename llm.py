from app.const import (
    MODEL_NAME,
    BASE_URL,
    TEMPERATURE,
    TOP_P,
    DEFAULT_TOOL_CHOICE,
    DEFAULT_PARALLEL_TOOL_CALLS,
)
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List
from pydantic import BaseModel

load_dotenv()  # Load environment variables from .env file


class LLM:

    @staticmethod
    def perform_tools_request(
        messages: List[dict],
        tools: List[dict],
        base_url: str = BASE_URL,
        model: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        tool_choice: str = DEFAULT_TOOL_CHOICE,
        parallel_tool_calls: bool = DEFAULT_PARALLEL_TOOL_CALLS,
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

    @staticmethod
    def perform_structured_outputs_request(
        messages: List[dict],
        response_format: BaseModel,
        base_url: str = BASE_URL,
        model: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ):

        # Initialize OpenAI client
        # Initialize OpenAI client
        client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("NEBIUS_STUDIO_API_KEY"),
        )

        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
        )

        return response
