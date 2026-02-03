DATA_PATH = "data/final_dataset.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

LABEL2ID = {
    "benign": 0,
    "human_phish": 1,
    "llm_phish": 2,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE_TRANSFORMER = 2e-5
LEARNING_RATE_HEAD = 1e-4
