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
from Activation import softmax

class DecoderMultiHeadAttention(nn.Module):
    def __init__(self, model_dimensionality, num_heads):
        super(DecoderMultiHeadAttention, self).__init__()

        assert model_dimensionality % num_heads == 0

        self.model_dimensionality = model_dimensionality
        self.num_heads = num_heads
        self.d_k = model_dimensionality / num_heads

        self.query = nn.Linear(model_dimensionality, model_dimensionality)
        self.key = nn.Linear(model_dimensionality, model_dimensionality)
        self.value = nn.Linear(model_dimensionality, model_dimensionality)
        self.output = nn.Linear(model_dimensionality, model_dimensionality)

    def dot_product_attention(self, query, key, value, mask=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_probabilities = softmax.SoftmaxActivation(x=attention_scores, dim=-1)

        output = torch.matmul(attention_probabilities, value)

        return output
    
    def combine_heads(self, x):
        batch_size = x.size()
        sequence_length = x.size()
        d_k = x.size()

        return x.view(batch_size, sequence_length, d_k, self.num_heads).transpose(1, 2)
    
    def split_heads(self, x):
        batch_size = x.size()
        _ = x.size()
        sequence_length = x.size()

        return x.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.model_dimensionality)
    
    def forward(self, query, key, value, mask=None):
        query = self.split_heads(self.query(query))
        key = self.split_heads(self.key(key))
        value = self.split_heads(self.value(value))

        attention_output = self.dot_product_attention(query, key, value, mask)

        output = self.output(self.combine_heads(attention_output))