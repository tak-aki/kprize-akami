import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_json(data: Any, path: str | Path) -> None:
    """Save data to a JSON file.

    Args:
        data: The data to save.
        path (str | Path): The path to the JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
