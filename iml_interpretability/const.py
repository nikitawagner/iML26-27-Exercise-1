from pathlib import Path
from typing import Literal

DEFAULT_TASK: Literal["classification", "regression"] = "classification"
DEFAULT_MODEL: Literal["decision_tree", "random_forest"] = "decision_tree"
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_MAX_DEPTH = 5
DEFAULT_MIN_SAMPLES_SPLIT = 2
DEFAULT_MIN_SAMPLES_LEAF = 5
DEFAULT_CCP_ALPHA = 0.0
DEFAULT_SEED = 0
DEFAULT_TEST_SIZE = 0.25
DEFAULT_N_ESTIMATORS = 100
DEFAULT_N_JOBS = 1
DEFAULT_MAX_FEATURES: Literal["sqrt", "log2", "all"] | None = None

TREE_FONT_NAME = "Helvetica"
TREE_FONT_SIZE = 12
