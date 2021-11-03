import numpy as np
import torch
import pandas as pd


def convertToTensor(data):
    data = pd.DataFrame(data, dtype=np.float32)
    return torch.Tensor(np.array(data))
