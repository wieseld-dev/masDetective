import os
import random
import runpy
import sys

import numpy as np
import torch

SEED = 1337

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]

runpy.run_path(
    script,
    run_name="__main__",
)
