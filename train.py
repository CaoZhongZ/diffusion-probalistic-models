import argparse
import numpy as np
import os
import warnings

import torch
import torchvision.transforms as T
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader

import model
import util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=512, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Initial learning rate. ' + \
                        'Will be decayed until it\'s 1e-5.')
    parser.add_argument('--resume_file', default=None, type=str,
                        help='Name of saved model to continue training')
    parser.add_argument('--suffix', default='', type=str,
                        help='Optional descriptive suffix for model')
    parser.add_argument('--output-dir', type=str, default='./',
                        help='Output directory to store trained models')
    parser.add_argument('--ext-every-n', type=int, default=25,
                        help='Evaluate training extensions every N epochs')
    parser.add_argument('--model-args', type=str, default='',
                        help='Dictionary string to be eval()d containing model arguments.')
    parser.add_argument('--dropout_rate', type=float, default=0.,
                        help='Rate to use for dropout during training+testing.')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Name of dataset to use.')
    parser.add_argument('--plot_before_training', type=bool, default=False,
                        help='Save diagnostic plots at epoch 0, before any training.')
    parser.add_argument('--use-mps', type=bool, default=False,
                        help='Use Apple MPS shader')
    parser.add_argument('--use-cuda', type=bool, default=False,
                        help='Use Nvidia GPU')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log loss during interval')
    args = parser.parse_args()

    model_args = eval('dict(' + args.model_args + ')')
    print(model_args)

    if not os.path.exists(args.output_dir):
        raise IOError("Output directory '%s' does not exist. "%args.output_dir)
    return args, model_args

def train(args, model, device, train_loader, optimizer, epoch):
    ## set up optimization
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.test()

    for batch_idx, (data, target) in enumerate(test_loader):
        pass

def checkpoint(args, epoch, optimizer, loss):
    model_dir = util.create_log_dir(args, dpm.name + '_' + args.dataset)
    model_save_name = os.path.join(model_dir, 'model.pt')
    torch.save({'epoch':epoch, 'model_state_dict': dpm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':loss}, model_save_name)

if __name__ == '__main__':
    __spec__ = None
    # TODO batches_per_epoch should not be hard coded
    batches_per_epoch = 500
    import sys
    sys.setrecursionlimit(10000000)

    args, model_args = parse_args()

    if args.use_cuda and args.use_mps:
        raise TypeError("Specify only single acceleration option")

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        args.use_cuda = False
        args.use_mps = False

    if args.resume_file is not None:
        print ("Resuming training from " + args.resume_file)
        from blocks.scripts import continue_training
        continue_training(args.resume_file)

    Flatten = T.Compose([
        T.ToTensor(),
        T.Normalize([0,], [1,]),
        torch.flatten])

    ## load the training data
    if args.dataset == 'MNIST':
        from torchvision.datasets import MNIST
        dataset_train = MNIST('.data', train=True, download=True,
                            transform=Flatten)
        dataset_test = MNIST('.data', train=False, download=True,
                            transform=Flatten)
        n_colors = 1
        spatial_width = 28
    elif args.dataset == 'CIFAR10':
        from torchvision.datasets import CIFAR10
        dataset_train = CIFAR10('.data', train=True, download=True,
                            transform=Flatten)
        dataset_test = CIFAR10('.data', train=False, download=True,
                            transform=Flatten)
        n_colors = 3
        spatial_width = 32
    elif args.dataset == 'IMAGENET':
        from torchvision.datasets import ImageNet
        dataset_train = ImageNet('.data', train=True, download=True,
                            transform=Flatten)
        dataset_test = ImageNet('.data', train=False, download=True,
                            transform=Flatten)
        n_colors = 3
        spatial_width = 128
    else:
        raise ValueError("Unknown dataset %s."%args.dataset)

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.batch_size, 'shuffle': True}

    if args.use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    elif args.use_mps:
        mps_kwargs = {'num_workers': 1}
        train_kwargs.update(mps_kwargs)
        test_kwargs.update(mps_kwargs)

    train_stream = DataLoader(dataset_train, **train_kwargs)
    test_stream = DataLoader(dataset_test, **test_kwargs)

    # make the training data 0 mean and variance 1
    # scale is applied before shift
    baseline_uniform_noise = 1./255. # appropriate for MNIST and CIFAR10 Fuel datasets, which are scaled [0,1]
    # scl = 1./np.sqrt(np.mean((Xbatch-np.mean(Xbatch))**2))
    scl = 1
    uniform_noise = baseline_uniform_noise/scl

    ## initialize the model
    dpm = model.DiffusionModel(spatial_width, n_colors, uniform_noise=uniform_noise, **model_args).to(device)
    optimizer = RMSprop(dpm.parameters(), lr=args.lr, eps=1e-10)
    scheduler = StepLR(optimizer, step_size=1, gamma=np.exp(np.log(0.1)/1000))

    for epoch in range(1, 100000 + 1):
        train(args, dpm, device, train_stream, optimizer, epoch)
        test(args, dpm, device, test_stream)
        scheduler.step()

