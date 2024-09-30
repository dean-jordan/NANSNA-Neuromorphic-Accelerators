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

class EncoderRecurrentLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, batch_size: int) -> None:
        super(EncoderRecurrentLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False)
    
    def forward(self, x, hidden_state) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_to_hidden(x)

        hidden_state = self.hidden_to_hidden(x)
        hidden_state = torch.tanh(x + hidden_state)

        output = self.hidden_to_hidden(x)

        return output, hidden_state