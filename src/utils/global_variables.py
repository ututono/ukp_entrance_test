ENCODING = "utf-8"

START_TAG = "<START>"
STOP_TAG = "<STOP>"

# <-- Data relevant configurations -->
TRAIN_FILENAME = "train.conll"
TEST_FILENAME = "test.conll"
DEV_FILENAME = "dev.conll"
DATA_DIR_NAME = "data"

# <-- Option for reading data files -->
DATA_OPT = "data-conll"

NEGLECT_TAGS = ["PAD"]

ENDING_PUNCTUATIONS = [".", "?", "!"]

# <-- Column names in the data files -->
WORD_COL = "Word"
POS_COL = "POS"
NP_COL = "NP"
NER_COL = "NER"
DATA_COL_NAMES = (WORD_COL, POS_COL, NP_COL, NER_COL)

# <-- Model relevant configurations -->
MODEL_NAME = "model.pt"
TRAIN_CONFIG_NAME = "train_config.json"

# <-- General settings -->
LOGGING_LEVEL = 'DEBUG'  # Logging level, possible values: DEBUG, INFO, WARNING, ERROR, CRITICAL
SEED = 2023
