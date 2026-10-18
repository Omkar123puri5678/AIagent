"""Settings for the sleep-stage model and its training."""

from pathlib import Path

DATA_DIR = Path("data/sleep_edf")
CHECKPOINT_FILE = Path("checkpoints/sleep_stager.pt")

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
N_CHANNELS = 3
EPOCH_SAMPLES = 3000

# Data split, by subject so nobody appears in more than one set.
SEED = 0
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

# Model
SEQ_LEN = 20  # consecutive 30-second epochs the model looks at together (10 minutes)
HIDDEN_SIZE = 128
DROPOUT = 0.5

# Training
BATCH_SIZE = 16  # sequences per step, so BATCH_SIZE * SEQ_LEN epochs
MAX_EPOCHS = 30
STEPS_PER_EPOCH = 600  # about one pass over the training data
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 5.0
PATIENCE = 6  # stop after this many epochs without a better validation score
