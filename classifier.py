import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True


num_classes = 101

class Classifier(nn.Module):

    def __init__(self, use_gpu=True):
        super(Classifier, self).__init__()
        if use_gpu:
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)).cuda()
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)).cuda()
            self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)).cuda()
            self.fc1 = nn.Linear(32768, num_classes).cuda()
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc1 = nn.Linear(32768, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out

    def predict(self, x):
        return self.forward(x)
