import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 50
FEATURE_DIM = 3
NUM_CLASSES = 3

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3