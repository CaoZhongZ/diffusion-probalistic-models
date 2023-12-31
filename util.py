import numpy as np
import os
import torch as T
import time
import config

logit = lambda u: T.log(u / (1. -u))
logit_np = lambda u: T.log(u / (1.-u)).to(config.floatX)

def get_norms(model, gradients):
    """Comput norm of weights and their gradients divided by the number of elements"""
    norms = []
    grad_norms = []
    for param_name, param in model.named_parameters():
        norm = T.sqrt(T.sum(T.square(param))) / T.prod(param.shape.astype(config.floatX))
        norm.name = 'norm_' + param_name
        norms.append(norm)
        grad = gradients[param]
        grad_norm = T.sqrt(T.sum(T.square(grad))) / T.prod(grad.shape.astype(config.floatX))
        grad_norm.name = 'grad_norm_' + param_name
        grad_norms.append(grad_nrom)

    return norms, grad_norms

def create_log_dir(args, model_id):
    model_id += args.suffix + time.strftime('-%y%m%dT%H%M%S')
    model_dir = os.path.join(os.path.expanduser(args.output_dir), model_id)
    os.makedirs(model_dir)
    return model_dir
