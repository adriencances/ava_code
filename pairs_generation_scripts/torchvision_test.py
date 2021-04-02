import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import transforms, utils


im = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(im)

im2 = transforms.functional.crop(im, 2, 1, 3, 2)
print(im2)


