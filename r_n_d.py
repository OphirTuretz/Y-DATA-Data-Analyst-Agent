from llm import LLM
from data import Dataset
import engine


def main():
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a helpful assistant.",
    #     },
    #     {
    #         "role": "user",
    #         "content": "What is the capital of France?",
    #     },
    # ]

    # response = LLM.perform_request(messages)
    # print(response.choices[0].message.content)

    ds = Dataset()
    print(ds.show_examples(5))

    queries = [
        # 1. Structured:
        "What are the most frequent categories?",
        # "Show examples of Category X.",
        # "What categories exist?",
        # "Show intent distributions.",
        # # 2. Unstructured:
        # "Summarize Category account.",  # called list_column_values
        # "Summarize how agent respond to Intent Y.",  # called list_column_values
        # # 3. Out-of-scope:
        # "Who is Magnus Carlson?",
        # "What is Serj's rating?",
    ]

    for query in queries:
        print("\n" + "=" * 50)
        engine.process_user_query(query, ds)
        print("=" * 50)


if __name__ == "__main__":
    main()
