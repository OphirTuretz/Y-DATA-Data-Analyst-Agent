from llm import LLM
from data import Dataset
import tools
from app.const import (
    STSTEM_PROMPT_FILE_PATH,
    MAX_CALL_DEPTH,
    DEFAULT_PARALLEL_TOOL_CALLS,
    DEFAULT_TOOL_CHOICE,
)
import json
from typing import Callable
from prompt import read_prompt_file


def process_user_query(
    user_query: str,
    ds: Dataset,
    log_function: Callable[[str], None] = print,
    llm_tools=tools.tools,
    llm_tool_choice=DEFAULT_TOOL_CHOICE,
    llm_parallel_tool_calls=DEFAULT_PARALLEL_TOOL_CALLS,
):

    # Load the system prompt from the file
    system_prompt = read_prompt_file(STSTEM_PROMPT_FILE_PATH)
    # with open(STSTEM_PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
    #     system_prompt = f.read()

    # Set the initial messages for the LLM
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_query},
    ]

    # Log the initial messages
    log_function(f"Initial messages: {json.dumps(messages, indent=2)}")

    # Perform the initial request to the LLM
    response = LLM.perform_tools_request(
        messages,
        tools=llm_tools,
        tool_choice=llm_tool_choice,
        parallel_tool_calls=llm_parallel_tool_calls,
    )

    # Log the response from the LLM
    log_function(f"Response from LLM: {json.dumps(response.model_dump(), indent=2)}")

    # Extract the assistant's message from the response
    assistant_message = response.choices[0].message

    depth = 0

    # Process tool calls if they exist in the assistant's message
    while assistant_message.tool_calls:

        depth += 1
        if depth > MAX_CALL_DEPTH:
            log_function("Maximum tool call depth exceeded. Exiting.")
            return {
                "response": "Sorry, the request caused too many internal steps and could not be completed.",
                "dataset": ds,
            }

        # Add the tool calls to the messages
        tool_calls_messages = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_message.tool_calls
            ],
        }
        messages.append(tool_calls_messages)

        # Log the number of tool calls
        log_function(f"Total tool calls: {len(assistant_message.tool_calls)}")

        # Log the tool calls messages added to the conversation
        log_function(
            f"Tools calls messages added: {json.dumps(tool_calls_messages, indent=2)}"
        )

        # Process each function call in the assistant's message
        for function in assistant_message.tool_calls:

            # Extract the function call details
            arguments = json.loads(function.function.arguments)

            try:
                # Execute the function call with the provided arguments
                function_input = tools.FunctionInput(function_call=arguments)
                output = tools.execute_function(function_input.function_call, ds)

                # Extract the function output and dataset from the output
                function_output = output["response"]
                ds = output["dataset"]

                if function_output.get("final_answer"):
                    # If the function output contains a final answer, return it
                    return {
                        "response": function_output["final_answer"],
                        "dataset": ds,
                    }

                # Set the function output to be added to the messages
                function_message = {
                    "role": "tool",
                    "tool_call_id": function.id,
                    "content": json.dumps(function_output),
                }

            except Exception as e:
                # Handle any exceptions that occur during function execution
                error_msg = f"Error processing function call: {str(e)}"
                log_function(error_msg)

                # Set the function output to be added to the messages
                function_message = {
                    "role": "tool",
                    "tool_call_id": function.id,
                    "content": f"Tool call failed: {error_msg}",
                }

            # Add the function call message to the messages
            messages.append(function_message)

            # Log the function call message added to the conversation
            log_function(
                f"Tool call output message added: {json.dumps(function_message, indent=2)}"
            )

        # Perform the next request to the LLM with the updated messages
        response = LLM.perform_tools_request(
            messages,
            tools=llm_tools,
            tool_choice=llm_tool_choice,
            parallel_tool_calls=llm_parallel_tool_calls,
        )

        # Log the response from the LLM
        log_function(
            f"Response from LLM: {json.dumps(response.model_dump(), indent=2)}"
        )

        # Extract the assistant's message from the response
        assistant_message = response.choices[0].message

    # Return the final response and updated dataset
    return {
        "response": assistant_message.content,
        "dataset": ds,
    }
