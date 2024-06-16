# %%

import torch
import numpy as np
import dasp_pytorch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List

from diff_reverb import NNFreeverbModule
from nontrainable_diff_reverb import NNFreeverbModuleClone
from architecture import ParameterNetwork
from plot_stuff import *
from dataset_things import AudioEffectDataset
from custom_loss import CustomLoss

num_epochs = 2
reverb = NNFreeverbModuleClone()
for param in reverb.parameters():
    param.requires_grad = True
neural_network = ParameterNetwork(53)
for param in neural_network.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(neural_network.parameters(), lr=1e-4)

dataset = AudioEffectDataset("data_mini")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
torch.autograd.set_detect_anomaly(True)


criterion = CustomLoss()
# torch.autograd.set_detect_anomaly(True)
for epoch in range(num_epochs):
    # iterate over the dataset
    print(f"Epoch {epoch}")
    loss_history = []
    pbar = tqdm(dataloader)
    last_y = None  # To store the last batch y
    last_y_hat = None  # To store the last batch y_hat
    for batch, data in enumerate(pbar):
        x, y = data
        y_in = y.mean(dim=1, keepdim=True)

        # x = x.cuda()
        # y = y.cuda()

        p_hat = neural_network(y_in)
        p_hat = p_hat.squeeze().clone()
        reverb.update_params(p_hat)
        y_hat = reverb(x)
        y_hat = y_hat.unsqueeze(0).clone()
        print(y_hat.shape)
        print(y.shape)
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # Print message if any gradient is None
        for name, param in neural_network.named_parameters():
            if param.grad is None:
                print(f"Gradient of {name} is None")

        optimizer.step()

        loss_history.append(loss.item())
        pbar.set_description(f"Loss: {loss.item()}")
        # Store the last batch y and y_hat
        last_y = y
        last_y_hat = y_hat
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # Create the logs directory if it doesn't exist
    plot_loss(log_dir, loss_history)

    if epoch % 5 == 0 and last_y is not None and last_y_hat is not None:
        # plot the frequency response of the first output, and first predicted output
        y_plot = last_y[0].clone().squeeze().detach()[0]
        y_hat_plot = last_y_hat[0].clone().squeeze().detach()[0]
        print(y_plot.shape)
        print(y_hat_plot.shape)
        fft_y = torch.fft.rfft(y_plot)
        fft_y_hat = torch.fft.rfft(y_hat_plot)

        fig, ax = plt.subplots(2, 1, figsize=(6, 4))
        # first subplot
        ax[0].plot(y_plot)
        ax[0].plot(y_hat_plot)
        ax[0].legend(["y", "y_hat"])
        # second subplot
        ax[1].plot(20 * torch.log10(torch.abs(fft_y) + 1e-8), label="y")
        ax[1].plot(20 * torch.log10(torch.abs(fft_y_hat) + 1e-8), label="y_hat")
        ax[1].legend()
        plt.savefig(f"logs/freq_response_{epoch}.png")
        plt.close()

# %%
