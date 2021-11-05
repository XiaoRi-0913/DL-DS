from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer = SummaryWriter("../logs")
image_path = "dataset/train/ants/0013035.jpg"
img_PIL = Image.open(image_path)
# JpegImageFile类型转为numpy
img_array = np.array(img_PIL)
print(img_array.shape)
# (512, 768, 3) Height Weight channels
#  Args:
#             tag (string): Data identifier
#             img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
#             global_step (int): Global step value to record
#             walltime (float): Optional override default walltime (time.time())
#               seconds after epoch of event
# dataformats='HWC' 输入图片的  HWC 要指定。
writer.add_image("test", img_array, 1, dataformats="HWC")
for i in range(100):
    # param 1 Y  param 2 x
    writer.add_scalar("y=2x", 3*i, i)

writer.close()