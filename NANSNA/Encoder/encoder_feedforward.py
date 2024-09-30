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

from Layers import encoder_recurrent

class EncoderFeedForwardNetwork(nn.Module):
    def __init__(self, model_dimensionality, feedforward_dimensionality):
        super(EncoderFeedForwardNetwork, self).__init__()
        
        self.recurrent1 = encoder_recurrent.EncoderRecurrentLayer(model_dimensionality, feedforward_dimensionality,
                                                                  feedforward_dimensionality)
        self.recurrent2 = encoder_recurrent.EncoderRecurrentLayer(model_dimensionality, feedforward_dimensionality,
                                                                  feedforward_dimensionality)
        self.recurrent3 = encoder_recurrent.EncoderRecurrentLayer(model_dimensionality, feedforward_dimensionality,
                                                                  feedforward_dimensionality)