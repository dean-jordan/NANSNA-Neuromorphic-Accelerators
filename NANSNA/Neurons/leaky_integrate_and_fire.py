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

delta_t = torch.tensor(1e-3)
tau = torch.tensor(5e-3)
beta = torch.exp(-delta_t/tau)

w = 0.4
beta = 0.819

def leaky_integrate_and_fire_neuron(membrane, x, weight, beta, threshold=1):
    spike = (membrane > threshold)

    membrane = beta * membrane + weight*x - spike*threshold

    return spike, membrane

num_steps = 200

x = torch.cat((torch.zeros(10), torch.ones(190)*0.5), 0)
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec = []
spk_rec = []

for step in range(num_steps):
  spk, mem = leaky_integrate_and_fire_neuron(mem, x[step], w=w, beta=beta)
  mem_rec.append(mem)
  spk_rec.append(spk)

mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plt.plot_cur_mem_spk(x*w, mem_rec, spk_rec, thr_line=1,ylim_max1=0.5,
                 title="LIF Neuron Model With Weighted Step Voltage")