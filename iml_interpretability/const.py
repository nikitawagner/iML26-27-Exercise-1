from pathlib import Path
from typing import Literal

DEFAULT_TASK: Literal["classification", "regression"] = "classification"
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_MAX_DEPTH = 5
DEFAULT_MIN_SAMPLES_SPLIT = 2
DEFAULT_MIN_SAMPLES_LEAF = 5
DEFAULT_CCP_ALPHA = 0.0
DEFAULT_SEED = 0
DEFAULT_TEST_SIZE = 0.25

TREE_FIGURE_SIZE = (24, 12)
TREE_FONT_SIZE = 8
TREE_IMAGE_DPI = 150
