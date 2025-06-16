import os

DATE_TIME_PATTERN = "%H:%M:%S"

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SPLIT = "train"

MODEL_NAME = "gpt-4o-mini"  # "meta-llama/Meta-Llama-3.1-70B-Instruct"  # "Qwen/Qwen2.5-72B-Instruct"  # "Qwen/Qwen2.5-32B-Instruct"
BASE_URL = None  # "https://api.studio.nebius.com/v1/"
API_KEY_ENV_VAR = "OPENAI_API_KEY"  # "NEBIUS_STUDIO_API_KEY"
TEMPERATURE = 0.0
TOP_P = 1.0
DEFAULT_TOOL_CHOICE = "auto"
DEFAULT_PARALLEL_TOOL_CALLS = False

MAX_CALL_DEPTH = 150

SYSTEM_PROMPT_FILE_NAME = "gpt_4o_mini_react_agent_system_prompt.txt"
STSTEM_PROMPT_FILE_PATH = os.path.join("prompts", SYSTEM_PROMPT_FILE_NAME)

SUMMARIZE_BATCH_PROMPT_FILE_NAME = "summarize_batch_prompt.txt"
SUMMARIZE_BATCH_PROMPT_FILE_PATH = os.path.join(
    "prompts", SUMMARIZE_BATCH_PROMPT_FILE_NAME
)

SUMMARIZE_DEFAULT_N_BATCHES = 5
SUMMARIZE_DEFAULT_BATCH_SIZE = 20

SUMMARIZE_ALL_BATCHES_PROMPT_FILE_NAME = "summarize_all_batches_prompt.txt"
SUMMARIZE_ALL_BATCHES_PROMPT_FILE_PATH = os.path.join(
    "prompts", SUMMARIZE_ALL_BATCHES_PROMPT_FILE_NAME
)
