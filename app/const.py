import os

DATE_TIME_PATTERN = "%H:%M:%S"

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SPLIT = "train"

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"  # "Qwen/Qwen3-30B-A3B"
BASE_URL = "https://api.studio.nebius.com/v1/"
TEMPERATURE = 0.0
TOP_P = 1.0
DEFAULT_TOOL_CHOICE = "auto"
DEFAULT_PARALLEL_TOOL_CALLS = True

MAX_CALL_DEPTH = 30

SYSTEM_PROMPT_FILE_NAME = "my_new_react_prompt.txt"
STSTEM_PROMPT_FILE_PATH = os.path.join("prompts", SYSTEM_PROMPT_FILE_NAME)

SUMMARIZE_BATCH_PROMPT_FILE_NAME = "summarize_batch_prompt.txt"
SUMMARIZE_BATCH_PROMPT_FILE_PATH = os.path.join(
    "prompts", SUMMARIZE_BATCH_PROMPT_FILE_NAME
)

SUMMARIZE_DEFAULT_N_BATCHES = 10
SUMMARIZE_DEFAULT_BATCH_SIZE = 50

SUMMARIZE_ALL_BATCHES_PROMPT_FILE_NAME = "summarize_all_batches_prompt.txt"
SUMMARIZE_ALL_BATCHES_PROMPT_FILE_PATH = os.path.join(
    "prompts", SUMMARIZE_ALL_BATCHES_PROMPT_FILE_NAME
)
