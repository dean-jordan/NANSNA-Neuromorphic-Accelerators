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

from Encoder import encoder
from Decoder import decoder
from Subnetwork import subnetwork_ensemble
from Activation import activation
from Loss import loss
from Adapters import adapters

class NANSNA(nn.Module):
    def __init__(self):
        super(NANSNA, self).__init__()