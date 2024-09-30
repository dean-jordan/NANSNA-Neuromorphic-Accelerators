import torch
import torchvision
import torch.nn as nn

import snntorch
from snntorch import spikeplot as splt
from snntorch import spikegen

from snntorch import backprop
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

from torch.utils.data import dataloader
from torchvision import datasets, transforms
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import itertools

class ReluActivation(nn.Module):
    def __init__(self):
        super(ReluActivation, self).__init__()

    def forward(self):
        x = torch.maximum(x, 0.0)
        return x