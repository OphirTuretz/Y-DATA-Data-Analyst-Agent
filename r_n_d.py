from llm import LLM
from data import Dataset


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

    dataset = Dataset()
    print(dataset.dataset.head())


if __name__ == "__main__":
    main()
