import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from testing_lib import class_to_idx

import cv2
import shutil
import random


def __getitem__(self, index): #returns one datapoint at a time
        image_filepath = self.image_paths[index]
        image = Image.open(image_filepath).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((359, 359))
        ])
        image = transform(image)
        
        label = (image_filepath.split("\\"))[-2]
        label = class_to_idx(label)
        if self.transform != None:
            image = self.transform(image = image)["image"] # our transforms when called must return the transformed image 
            
        return image, label