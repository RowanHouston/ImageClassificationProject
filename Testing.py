import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

import cv2
import shutil
import random

dataset = "c:/Users/tilli/OneDrive/Desktop/ImageClassificationProject/bloodcells_dataset"
img_path_end = os.listdir((dataset + "/" + str(os.listdir(dataset)[0])))[0]
image_path = dataset + '/basophil' + '/' + img_path_end
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor = torch.from_numpy((np.asarray(image)))
print(image_tensor.size()) 

