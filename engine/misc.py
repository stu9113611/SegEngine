import random
from collections import Counter

import numpy as np
import torch
from typing import TypeVar


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


T = TypeVar('T')


def to_tuple(value: T, number: int) -> tuple[T]:
    return tuple([value for _ in range(number)])