import torch
from model import NeRF
from data import NeRFDataset
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, PeakSignalNoiseRatio
from nerf import train_nerf
import matplotlib.pyplot as plt
import os
import numpy as np



nerf = NeRF()
dataset = NeRFDataset(train = True)
dataloader = DataLoader(dataset, batch_size = 4)


if torch.cuda.is_available():
    device = torch.device('cuda')
else :
    device = torch.device('cpu')

nerf = nerf.to(device)
train_nerf(dataloader, nerf, device, epochs = 2000, lr = 0.001)



