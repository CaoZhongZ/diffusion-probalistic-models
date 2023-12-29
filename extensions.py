"""
Extensions called during training to generate samples and diagnostic plots and printouts.
"""

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import torch as T

import viz
import sampler

def decay_learning_rate(iteration, old_value):
    min_value = 1e-4

    decay_rate = T.exp(T.log(0.1)/1000.)
    new_value = decay_rate * old_value
    if new_value < min_value:
        new_value = min_value
    return new_value.to(torch.float32)
