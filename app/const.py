import os

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SPLIT = "train"

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"  # "Qwen/Qwen3-30B-A3B"
BASE_URL = "https://api.studio.nebius.com/v1/"
TEMPERATURE = 0.0
TOP_P = 1.0

SYSTEM_PROMPT_FILE_NAME = "system_prompt.txt"
STSTEM_PROMPT_FILE_PATH = os.path.join("prompts", SYSTEM_PROMPT_FILE_NAME)
