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

from Layers import decoder_recurrent
from Activation import relu

class DecoderFeedForwardNetwork(nn.Module):
    def __init__(self, model_dimensionality, feedforward_dimensionality):
        super(DecoderFeedForwardNetwork, self).__init__()

        self.activation = relu.ReluActivation()
        
        self.recurrent1 = decoder_recurrent.DecoderRecurrentLayer(model_dimensionality, feedforward_dimensionality,
                                                                  feedforward_dimensionality)
        self.recurrent2 = decoder_recurrent.DecoderRecurrentLayer(model_dimensionality, feedforward_dimensionality,
                                                                  feedforward_dimensionality)
        self.recurrent3 = decoder_recurrent.DecoderRecurrentLayer(model_dimensionality, feedforward_dimensionality,
                                                                  feedforward_dimensionality)
        self.recurrent4 = decoder_recurrent.DecoderRecurrentLayer(model_dimensionality, feedforward_dimensionality,
                                                                  feedforward_dimensionality)
        self.recurrent5 = decoder_recurrent.DecoderRecurrentLayer(model_dimensionality, feedforward_dimensionality,
                                                                  feedforward_dimensionality)
        self.recurrent6 = decoder_recurrent.DecoderRecurrentLayer(model_dimensionality, feedforward_dimensionality,
                                                                  feedforward_dimensionality)
        self.recurrent7 = decoder_recurrent.DecoderRecurrentLayer(model_dimensionality, feedforward_dimensionality,
                                                                  feedforward_dimensionality)
        self.recurrent8 = decoder_recurrent.DecoderRecurrentLayer(model_dimensionality, feedforward_dimensionality,
                                                                  feedforward_dimensionality)
        self.recurrent9 = decoder_recurrent.DecoderRecurrentLayer(model_dimensionality, feedforward_dimensionality,
                                                                  feedforward_dimensionality)
        self.recurrent10 = decoder_recurrent.DecoderRecurrentLayer(model_dimensionality, feedforward_dimensionality,
                                                                  feedforward_dimensionality)
        
    def forward(self, x):
        return self.recurrent10(
            self.activation(
                self.recurrent9(
                    self.activation(
                        self.recurrent8(
                            self.activation(
                                self.recurrent7(
                                    self.activation(
                                        self.recurrent6(
                                            self.activation(
                                                self.recurrent5(
                                                    self.activation(
                                                        self.recurrent4(
                                                            self.activation(
                                                                self.recurrent3(
                                                                    self.activation(
                                                                        self.recurrent2(
                                                                            self.activation(
                                                                                self.recurrent1(
                                                                                    self.activation(x)
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )