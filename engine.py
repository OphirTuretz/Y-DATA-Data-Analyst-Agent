from llm import LLM
from data import Dataset
import tools
from app.const import STSTEM_PROMPT_FILE_PATH
import json
from typing import Callable


def log_to_console(message: str) -> None:
    """Logs a message to the console."""
    print(message)


def process_user_query(
    user_query: str,
    ds: Dataset,
    log_function: Callable[[str], None] = log_to_console,
    tools=tools.tools,
    tool_choice="auto",
    parallel_tool_calls=True,
):

    # Load the system prompt from the file
    with open(STSTEM_PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # Set the initial messages for the LLM
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_query},
    ]

    # Log the initial messages
    log_function(f"Initial messages: {messages}")

    # Perform the initial request to the LLM
    response = LLM.perform_request(
        messages,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )

    # Log the response from the LLM
    log_function(f"Response from LLM: {response}")

    # Extract the assistant's message from the response
    assistant_message = response.choices[0].message

    # Process tool calls if they exist in the assistant's message
    while assistant_message.tool_calls:

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
        log_function(f"Total function calls: {len(assistant_message.tool_calls)}")

        # Log the tool calls messages added to the conversation
        log_function(f"Tool calls messages added: {tool_calls_messages}")

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
            log_function(f"Function call message added: {function_message}")

            if function_output.get("final_answer"):
                # If the function output contains a final answer, return it
                return {
                    "response": function_output["final_answer"],
                    "dataset": ds,
                }

        # Perform the next request to the LLM with the updated messages
        response = LLM.perform_request(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Log the response from the LLM
        log_function(f"Response from LLM: {response}")

        # Extract the assistant's message from the response
        assistant_message = response.choices[0].message

    # Return the final response and updated dataset
    return {
        "response": assistant_message.content,
        "dataset": ds,
    }
