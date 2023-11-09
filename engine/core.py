from collections import Counter
import torch
import numpy as np
import random


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dict_add(dict1, dict2):
    return dict(Counter(dict1) + Counter(dict2))
