from llm import LLM
from data import Dataset
import tools
from app.const import STSTEM_PROMPT_FILE_PATH


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

    # Get the response
    assistant_message = response.choices[0].message
    results = []

    # Check if there's a function call
    if assistant_message.tool_calls:
        print("total function calls: ", len(assistant_message.tool_calls))
        for function in assistant_message.tool_calls:
            function_call = function.function
            function_name = function_call.name
            arguments = json.loads(function_call.arguments)

            # Add the function_type to the arguments for discriminated union
            arguments["function_type"] = function_name

            # Parse the input with Pydantic
            try:
                function_input = FunctionInput(function_call=arguments)
                result = execute_function(function_input)

                # For demonstration, show what happened
                print(f"User query: {user_query}")
                print(f"Function called: {function_name}")
                print(f"Arguments: {arguments}")
                print(f"Result: {json.dumps(result, indent=2)}")
                results.append(result)
            except Exception as e:
                error_msg = f"Error processing function call: {str(e)}"
                print(error_msg)
                results.append({"error": error_msg})
    else:
        # If no function was called, return the assistant's message
        print(f"User query: {user_query}")
        print("No function was called. Assistant response:")
        print(assistant_message.content)
        results.append(assistant_message.content)

    return results
