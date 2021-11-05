import numpy as np
import torch
import pandas as pd


def convertToTensor(data):
    data = pd.DataFrame(data)
    torch.set_printoptions(
        precision=6,  # 精度，保留小数点后几位，默认4
        threshold=1000,
        edgeitems=3,
        linewidth=150,  # 每行最多显示的字符数，默认80，超过则换行显示
        profile=None,
        sci_mode=False  # 用科学技术法显示数据，默认True
    )
    return torch.tensor(np.array(data), dtype=torch.float32)



