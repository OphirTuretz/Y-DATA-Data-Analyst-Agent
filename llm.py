from app.const import MODEL_NAME, BASE_URL, TEMPERATURE, TOP_P
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file


class LLM:

    @staticmethod
    def perform_request(
        messages,
        base_url=BASE_URL,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=False,
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
