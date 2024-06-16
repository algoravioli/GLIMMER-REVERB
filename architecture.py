import glob
import torch
import auraloss
import torchaudio
import numpy as np
import dasp_pytorch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List


import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x.clone())))
        out = torch.relu(self.bn2(self.conv2(out.clone())))
        return out


class ParameterNetwork(nn.Module):
    def __init__(self, num_control_params: int):
        super(ParameterNetwork, self).__init__()
        self.num_control_params = num_control_params

        # Use a simple TCN to estimate the parameters
        self.blocks = nn.ModuleList()
        self.blocks.append(TCNBlock(1, 16, 3, dilation=1))
        self.blocks.append(TCNBlock(16, 32, 3, dilation=2))

        # Initialize all params to random values
        for block in self.blocks:
            nn.init.kaiming_normal_(block.conv1.weight)
            nn.init.kaiming_normal_(block.conv2.weight)
            nn.init.zeros_(block.conv1.bias)
            nn.init.zeros_(block.conv2.bias)
            nn.init.ones_(block.bn1.weight)
            nn.init.zeros_(block.bn1.bias)
            nn.init.ones_(block.bn2.weight)
            nn.init.zeros_(block.bn2.bias)

        self.linear = nn.Linear(32, num_control_params)
        nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x.clone())
        x = x.clone().mean(dim=-1)  # Aggregate over time
        p = torch.relu(self.linear(x.clone()))  # Map to parameters
        return p
