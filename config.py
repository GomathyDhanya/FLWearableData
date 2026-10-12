from pathlib import Path
import random
import numpy as np
import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_ROUNDS = 300
MIN_ROUNDS = 30
PATIENCE = 20
MIN_DELTA = 0.001
TOP_N_CLASSES = 5
WINDOW_SIZE = 5
WINDOW_STRIDE = 5
ALIGN_TOLERANCE = "250ms"
INPUT_DIM = 60
N_CLASSES = 5
MAX_GRAD_NORM = 1.0
HE_DP_SIGMA = 0.5
RESULTS_DIR = Path("Final_Results")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
