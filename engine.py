from llm import LLM
from data import Dataset
import tools
from app.const import STSTEM_PROMPT_FILE_PATH
import json


def process_user_query(user_query: str, ds: Dataset) -> dict:

    # Load the system prompt from the file
    with open(STSTEM_PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # Set the messages for the LLM
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_query},
    ]

    response = LLM.perform_request(
        messages, tools=tools.tools, tool_choice="auto", parallel_tool_calls=True
    )

    assistant_message = response.choices[0].message

    while assistant_message.tool_calls:
        print("total function calls: ", len(assistant_message.tool_calls))

        messages.append(
            {
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
        )

        for function in assistant_message.tool_calls:

            print(f"Processing function call: {function}")

            function_call = function.function
            arguments = json.loads(function_call.arguments)

            function_name = function.function.name

            try:
                function_input = tools.FunctionInput(function_call=arguments)
                output = tools.execute_function(function_input.function_call, ds)

                function_output = output["response"]
                ds = output["dataset"]

                print(f"User query: {user_query}")
                print(f"Function called: {function_name}")
                print(f"Arguments: {arguments}")
                print(f"Result: {json.dumps(function_output, indent=2)}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": function.id,
                        "content": json.dumps(function_output),
                    }
                )

            except Exception as e:
                error_msg = f"Error processing function call: {str(e)}"
                print(error_msg)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": function.id,
                        "content": f"Tool call failed: {error_msg}",
                    }
                )

            print(f"Updated messages: {messages}")

        response = LLM.perform_request(
            messages, tools=tools.tools, tool_choice="auto", parallel_tool_calls=True
        )
        assistant_message = response.choices[0].message

    # else:
    #     # If no function was called, return the assistant's message
    #     print(f"User query: {user_query}")
    #     print("No function was called. Assistant response:")
    #     print(assistant_message.content)
    #     results.append(assistant_message.content)

    # return results
