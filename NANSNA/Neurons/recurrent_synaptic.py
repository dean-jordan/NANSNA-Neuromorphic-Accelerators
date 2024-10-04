import torch
import torchvision
import torch.nn as nn

import snntorch as snn
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

alpha = 0.9
beta = 0.8
num_steps = 200

lif_recurrent = snn.Synaptic(alpha=alpha, beta=beta)

w = 0.2
spk_period = torch.cat((torch.ones(1)*w, torch.zeros(9)), 0)
spk_in = spk_period.repeat(20)

syn, mem = lif_recurrent.init_synaptic()
spk_out = torch.zeros(1)
syn_rec = []
mem_rec = []
spk_rec = []

for step in range(num_steps):
  spk_out, syn, mem = lif_recurrent(spk_in[step], syn, mem)
  spk_rec.append(spk_out)
  syn_rec.append(syn)
  mem_rec.append(mem)

spk_rec = torch.stack(spk_rec)
syn_rec = torch.stack(syn_rec)
mem_rec = torch.stack(mem_rec)

plt.plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec,
                     "Synaptic Conductance-based Neuron Model With Input Spikes")