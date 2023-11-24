ENCODING = "utf-8"

START_TAG = "<START>"
STOP_TAG = "<STOP>"

TRAIN_FILENAME = "train.conll"
TEST_FILENAME = "test.conll"
DEV_FILENAME = "dev.conll"

NEGLECT_TAGS = ["PAD"]

ENDING_PUNCTUATIONS = [".", "?", "!"]

# <-- Column names in the data files -->
WORD_COL = "Word"
POS_COL = "POS"
NP_COL = "NP"
NER_COL = "NER"
DATA_COL_NAMES = (WORD_COL, POS_COL, NP_COL, NER_COL)

# <-- Option for reading data files -->
DATA_OPT = "data-conll"
