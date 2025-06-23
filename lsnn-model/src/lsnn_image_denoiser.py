import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.pyt_spk_encoder import SpikeEncoder  # Your spike encoder
from utils.pyt_surr_grad_spike import SpikeFunction  # Surrogate gradient for spikes

import torch
import torch.nn as nn
from pyt_spk_encoder import TensorEncoder


class LSNNImageDenoiser(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=1024, order=64, debug=False):
        super(LSNNImageDenoiser, self).__init__()
        self.debug = debug

        # Spike encoder layer: encodes input into spike trains over ORDER timesteps
        self.encoder = SpikeEncoder(input_dim, order, debug=debug)

        # Recurrent spiking layer (simple version for now)
        self.rnn = nn.RNN(input_size=order, hidden_size=hidden_dim, batch_first=True)

        # Decoder to project hidden state back to patch
        self.decoder = nn.Linear(hidden_dim, output_dim)

        # Surrogate gradient spike function
        self.spike_fn = SpikeFunction.apply

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 1024)

        Returns:
            Tensor of shape (batch_size, 1024) representing the denoised image patch
        """
        # Encode to spike input over time
        spike_input = self.encoder.encode_inp(x)  # shape: (batch, time, order)

        # Pass through recurrent layer
        rnn_out, _ = self.rnn(spike_input)  # shape: (batch, time, hidden_dim)

        # Use last time step output to decode
        last_hidden = rnn_out[:, -1, :]  # shape: (batch, hidden_dim)

        out = self.decoder(last_hidden)  # shape: (batch, 1024)
        return out
    


class SpikeEncoder2D(nn.Module):
    def __init__(self, shape=(1, 32, 32), timesteps=10):
        super().__init__()
        self.timesteps = timesteps
        self.shape = shape
        self.encoder = TensorEncoder(tensor_size=(1, shape[1] * shape[2]))

    def forward(self, x):
        # x: [B, 1, 32, 32]
        B = x.size(0)
        x_flat = x.view(B, -1)  # [B, 1024]
        spike_seq = []

        # Encode into T time steps
        for _ in range(self.timesteps):
            spikes = self.encoder.encode(x_flat)  # [B, 1024]
            spike_seq.append(spikes.unsqueeze(1))  # [B, 1, 1024]

        return torch.cat(spike_seq, dim=1)  # [B, T, 1024]

