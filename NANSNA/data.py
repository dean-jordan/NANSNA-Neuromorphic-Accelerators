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

class DataPipeline(nn.Module):
    def __init__(self):
        super(DataPipeline, self).__init__()