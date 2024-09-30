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
from Activation import sigmoid
from Activation import relu
from Activation import leakyrelu
from Activation import softmax
from Loss import loss
from Adapters import adapters

class NANSNA(nn.Module):
    def __init__(self, max_sequence_length, model_dimensionality, num_heads):
        super(NANSNA, self).__init__()

        self.encoder = encoder.EncoderBlock()
        self.decoder = decoder.DecoderBlock()
        self.subnetwork = subnetwork_ensemble.SubnetworkEnsemble()

        self.sigmoid = sigmoid.SigmoidActivation()
        self.relu = relu.ReluActivation()
        self.leakyrelu = leakyrelu.LeakyReluActivation()
        self.softmax = softmax.SoftmaxActivation()