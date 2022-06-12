import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.nn.functional as F

from torchvision.models import vgg16

from utils import gaussian

# Define Gaussian function
g = gaussian(25, 11.2)
kernel = torch.matmul(g.unsqueeze(-1), g.unsqueeze(-1).t())
kernel = kernel[None, None, :] # Remove extra dimensions

# Define center bias
center_bias = torch.from_numpy(np.load("../data/center_bias_density.npy"))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.center_bias = nn.Parameter(torch.log(center_bias), requires_grad=False)

        self.vgg16 = vgg16(pretrained=True)
        self.vgg16 = self.vgg16.features[:30]
        
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 1, 1, padding=0)
        )

    def forward(self, x):
        x = self.vgg16(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear")
        x = self.classifier(x)
        x = F.conv2d(x, self.kernel, padding="same")
        x = x + self.center_bias
        return x