from pydantic import BaseModel, Field
from typing import List, Literal
from typing import Union
import pandas as pd
from data import Dataset


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


def summarize(user_request: str, ds: Dataset) -> str:
    """
    Summarize the user request.
    Args:
        user_request (str): The user request to summarize.
        ds (Dataset): The dataset to use for summarization.
    Returns:
        str: A summary of the user request.
    """
    # Placeholder for summarization logic
    return f"Summary of '{user_request}' using dataset {ds.dataset_name}"


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
    function_type: Literal["get_possible_intents"]


class GetPossibleCategoriesInput(BaseModel):
    function_type: Literal["get_possible_categories"]


class SelectSemanticIntentInput(BaseModel):
    function_type: Literal["select_semantic_intent"]
    intent_names: List[str] = Field(
        ..., description="List of intent names to filter by."
    )


class SelectSemanticCategoryInput(BaseModel):
    function_type: Literal["select_semantic_category"]
    category_names: List[str] = Field(
        ..., description="List of category names to filter by."
    )


class SumInput(BaseModel):
    function_type: Literal["sum"]
    a: int = Field(..., description="First number to sum.")
    b: int = Field(..., description="Second number to sum.")


class CountCategoryInput(BaseModel):
    function_type: Literal["count_category"]
    category: str = Field(..., description="Category to count in the DataFrame.")


class CountIntentInput(BaseModel):
    function_type: Literal["count_intent"]
    intent: str = Field(..., description="Intent to count in the DataFrame.")


class ShowExamplesInput(BaseModel):
    function_type: Literal["show_examples"]
    n: int = Field(..., description="Number of examples to show from the DataFrame.")


class SummarizeInput(BaseModel):
    function_type: Literal["summarize"]
    user_request: str = Field(..., description="User request to summarize.")


class FinishInput(BaseModel):
    function_type: Literal["finish"]
    final_answer: str = Field(..., description="Final answer to return.")


class CountRowsInput(BaseModel):
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
            "name": "count_rows",
            "description": "Count the number of rows in the dataset",
            "parameters": CountRowsInput.model_json_schema(),
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
            "name": "count_category",
            "description": "Count the number of rows in the dataset that match a specific category",
            "parameters": CountCategoryInput.model_json_schema(),
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
            "name": "show_examples",
            "description": "Show a number of examples from the dataset",
            "parameters": ShowExamplesInput.model_json_schema(),
        },
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "summarize",
    #         "description": (
    #             "Summarize a user request using the dataset. "
    #             "The user request is provided as input."
    #         ),
    #         "parameters": SummarizeInput.model_json_schema(),
    #     },
    # },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish the conversation with a final answer",
            "parameters": FinishInput.model_json_schema(),
        },
    },
]
