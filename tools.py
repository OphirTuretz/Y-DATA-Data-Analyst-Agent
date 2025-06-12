from pydantic import BaseModel, Field
from typing import List, Literal
from typing import Union
import pandas as pd
from data import Dataset
from app.const import (
    SUMMARIZE_DEFAULT_BATCH_SIZE,
    SUMMARIZE_DEFAULT_N_BATCHES,
    SUMMARIZE_BATCH_PROMPT_FILE_PATH,
    SUMMARIZE_ALL_BATCHES_PROMPT_FILE_PATH,
)
from prompt import read_prompt_file
from llm import LLM


def sum(a: float, b: float) -> float:
    """
    Sum two numbers.
    Args:
        a (float): First number.
        b (float): Second number.
    Returns:
        float: The sum of a and b.
    """
    return a + b


def summarize(
    user_request: str,
    ds: Dataset,
    n_batches: int = SUMMARIZE_DEFAULT_N_BATCHES,
    batch_size: int = SUMMARIZE_DEFAULT_BATCH_SIZE,
) -> str:
    """
    Summarize a user request using the dataset.
    Args:
        user_request (str): The user request to summarize.
        ds (Dataset): The dataset to use for summarization.
        n_batches (int): Number of batches to process.
        batch_size (int): Size of each batch.
    Returns:
        str: A summary of the user request.
    """

    # Sample rows from the dataset to use for summarization
    n_rows_to_sample = min(ds.count_rows(), n_batches * batch_size)
    sampled_df = ds.show_examples(n_rows_to_sample)

    # Read the prompt files for summarization
    summarize_batch_prompt = read_prompt_file(SUMMARIZE_BATCH_PROMPT_FILE_PATH)
    summarize_all_batches_prompt = read_prompt_file(
        SUMMARIZE_ALL_BATCHES_PROMPT_FILE_PATH
    )

    # Initialize a list to hold batch summaries
    batch_summaries = []

    # Iterate over the sampled DataFrame in batches
    for i in range(0, n_rows_to_sample, batch_size):
        batch_df = sampled_df.iloc[i : i + batch_size]

        # Each batch prompt is in independent conversation in order to avoid biasing the LLM
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes user requests using a dataset.",
            },
            {
                "role": "user",
                "content": summarize_batch_prompt.format(
                    user_request=user_request, data=batch_df.to_dict(orient="records")
                ),
            },
        ]

        # Perform the initial request to the LLM
        response = LLM.perform_request(
            messages,
        )

        # Extract the assistant's message from the response
        batch_summaries.append(response.choices[0].message.content)

    # Combine all batch summaries into a final summary
    combined_summaries = "\n".join(batch_summaries)
    final_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes user requests using a dataset.",
        },
        {
            "role": "user",
            "content": summarize_all_batches_prompt.format(
                user_request=user_request, summaries=combined_summaries
            ),
        },
    ]
    # Perform the final request to the LLM
    final_response = LLM.perform_request(
        final_messages,
    )
    # Extract the final summary from the response
    final_answer = final_response.choices[0].message.content

    return final_answer


def finish(final_answer: str) -> str:
    """
    Finish the conversation with a final answer.
    Args:
        final_answer (str): The final answer to return.
    Returns:
        str: The final answer.
    """
    return final_answer


class GetPossibleIntentsInput(BaseModel):
    reasoning: str = Field(..., description="Reasoning for the function call.")
    function_type: Literal["get_possible_intents"]


class GetPossibleCategoriesInput(BaseModel):
    reasoning: str = Field(..., description="Reasoning for the function call.")
    function_type: Literal["get_possible_categories"]


class SelectSemanticIntentInput(BaseModel):
    reasoning: str = Field(..., description="Reasoning for the function call.")
    function_type: Literal["select_semantic_intent"]
    intent_names: List[str] = Field(
        ..., description="List of intent names to filter by."
    )


class SelectSemanticCategoryInput(BaseModel):
    reasoning: str = Field(..., description="Reasoning for the function call.")
    function_type: Literal["select_semantic_category"]
    category_names: List[str] = Field(
        ..., description="List of category names to filter by."
    )


class SumInput(BaseModel):
    reasoning: str = Field(..., description="Reasoning for the function call.")
    function_type: Literal["sum"]
    a: int = Field(..., description="First number to sum.")
    b: int = Field(..., description="Second number to sum.")


class CountCategoryInput(BaseModel):
    reasoning: str = Field(..., description="Reasoning for the function call.")
    function_type: Literal["count_category"]
    category: str = Field(..., description="Category to count in the DataFrame.")


class CountIntentInput(BaseModel):
    reasoning: str = Field(..., description="Reasoning for the function call.")
    function_type: Literal["count_intent"]
    intent: str = Field(..., description="Intent to count in the DataFrame.")


class ShowExamplesInput(BaseModel):
    reasoning: str = Field(..., description="Reasoning for the function call.")
    function_type: Literal["show_examples"]
    n: int = Field(..., description="Number of examples to show from the DataFrame.")


class SummarizeInput(BaseModel):
    reasoning: str = Field(..., description="Reasoning for the function call.")
    function_type: Literal["summarize"]
    user_request: str = Field(..., description="User request to summarize.")


class FinishInput(BaseModel):
    reasoning: str = Field(..., description="Reasoning for the function call.")
    function_type: Literal["finish"]
    final_answer: str = Field(..., description="Final answer to return.")


class CountRowsInput(BaseModel):
    reasoning: str = Field(..., description="Reasoning for the function call.")
    function_type: Literal["count_rows"]


FunctionType = Union[
    GetPossibleIntentsInput,
    GetPossibleCategoriesInput,
    SelectSemanticIntentInput,
    SelectSemanticCategoryInput,
    SumInput,
    CountCategoryInput,
    CountIntentInput,
    ShowExamplesInput,
    SummarizeInput,
    FinishInput,
    CountRowsInput,
]


class FunctionInput(BaseModel):
    function_call: FunctionType = Field(discriminator="function_type")


def execute_function(function_call: FunctionType, ds: Dataset):
    """
    Execute the function call on the dataset.
    Args:
        function_call (FunctionType): The function call to execute.
        ds (Dataset): The dataset to operate on.
    Returns:
        dict: A dictionary containing the dataset and the response from the function call.
    """
    output = {}
    output["dataset"] = ds

    if isinstance(function_call, GetPossibleIntentsInput):
        output["response"] = {"possible_intents": ds.get_possible_intents()}
        return output
    elif isinstance(function_call, GetPossibleCategoriesInput):
        output["response"] = {"possible_categories": ds.get_possible_categories()}
        return output
    elif isinstance(function_call, SelectSemanticIntentInput):
        ds = ds.select_semantic_intent(function_call.intent_names)
        output["dataset"] = ds
        output["response"] = {
            "selected_intents": ds.get_possible_intents(),
            "number_of_rows": ds.count_rows(),
        }
        return output
    elif isinstance(function_call, SelectSemanticCategoryInput):
        ds = ds.select_semantic_category(function_call.category_names)
        output["dataset"] = ds
        output["response"] = {
            "selected_categories": ds.get_possible_categories(),
            "number_of_rows": ds.count_rows(),
        }
        return output
    elif isinstance(function_call, CountRowsInput):
        output["response"] = {"number_of_rows": ds.count_rows()}
        return output
    elif isinstance(function_call, SumInput):
        result = sum(function_call.a, function_call.b)
        output["response"] = {"sum": result}
        return output
    elif isinstance(function_call, CountCategoryInput):
        count = ds.count_category(function_call.category)
        output["response"] = {"count": count}
        return output
    elif isinstance(function_call, CountIntentInput):
        count = ds.count_intent(function_call.intent)
        output["response"] = {"count": count}
        return output
    elif isinstance(function_call, ShowExamplesInput):
        examples = ds.show_examples(function_call.n)
        output["response"] = {"examples": examples.to_dict(orient="records")}
        return output
    elif isinstance(function_call, SummarizeInput):
        summary = summarize(function_call.user_request, ds)
        output["response"] = {"summary": summary}
        return output
    elif isinstance(function_call, FinishInput):
        final_answer = finish(function_call.final_answer)
        output["response"] = {"final_answer": final_answer}
        return output
    else:
        raise ValueError(f"Unknown function type: {function_call.function_type}")


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_possible_intents",
            "description": "Get a list of possible intents from the dataset",
            "parameters": GetPossibleIntentsInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_possible_categories",
            "description": "Get a list of possible categories from the dataset",
            "parameters": GetPossibleCategoriesInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_intent",
            "description": "Count the number of rows in the dataset that match a specific intent",
            "parameters": CountIntentInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_category",
            "description": "Count the number of rows in the dataset that match a specific category",
            "parameters": CountCategoryInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_rows",
            "description": "Count the number of rows in the dataset",
            "parameters": CountRowsInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_examples",
            "description": "Show a number of examples from the dataset",
            "parameters": ShowExamplesInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": (
                "Summarize a user request using the dataset. "
                "The user request is provided as input."
            ),
            "parameters": SummarizeInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_semantic_intent",
            "description": "Select rows from the dataset where the 'intent' column matches any of the provided intent names",
            "parameters": SelectSemanticIntentInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_semantic_category",
            "description": "Select rows from the dataset where the 'category' column matches any of the provided category names",
            "parameters": SelectSemanticCategoryInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sum",
            "description": "Sum two numbers",
            "parameters": SumInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish the conversation with a final answer. Do not include follow-up prompts, only provide the final answer to the user's original question.",
            "parameters": FinishInput.model_json_schema(),
        },
    },
]
