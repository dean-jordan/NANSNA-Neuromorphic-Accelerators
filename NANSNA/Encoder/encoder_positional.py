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

import math

class EncoderPositionalEncoding(nn.Module):
    def __init__(self, model_dimensionality, max_sequence_length):
        super(EncoderPositionalEncoding, self).__init__()

        encoding_tensors = torch.zeros(max_sequence_length, model_dimensionality)
        positonal_tensors = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        divided_term = torch.exp(torch.arange(0, model_dimensionality, 2).float() * -(math.log(10000.0) / model_dimensionality))

        encoding_tensors[:, 0::2] = torch.sin(positonal_tensors * divided_term)
        encoding_tensors[:, 1::2] = torch.cos(positonal_tensors * divided_term)

        self.register_buffer('encoding_tensors', encoding_tensors.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding_tensors[:, :x.size(1)]