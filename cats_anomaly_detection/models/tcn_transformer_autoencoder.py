from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class CausalConv1d(nn.Module):
    """1D convolution with left-only padding to preserve causal semantics."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padded_inputs = functional.pad(inputs, (self.left_padding, 0))
        return self.convolution(padded_inputs)


class TemporalBlock(nn.Module):
    """Canonical residual TCN block with causal and dilated convolutions."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.network(inputs)


class TCNBlock(nn.Module):
    """Stacked TCN blocks with exponentially increasing dilation schedule."""

    def __init__(self, channels: int, kernel_size: int, levels: int, dropout: float) -> None:
        super().__init__()
        temporal_blocks = [
            TemporalBlock(
                channels=channels,
                kernel_size=kernel_size,
                dilation=2**level_index,
                dropout=dropout,
            )
            for level_index in range(levels)
        ]
        self.network = nn.Sequential(*temporal_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TCNTransformerAutoencoder(nn.Module):
    """Autoencoder that combines canonical TCN and transformer context.

    Pipeline:
    1. Project input features to hidden channels.
    2. Extract causal temporal patterns via a dilated residual TCN stack.
    3. Model longer-range dependencies with a Transformer encoder.
    4. Decode back to input feature space for reconstruction loss training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        transformer_heads: int,
        transformer_layers: int,
        tcn_levels: int,
        tcn_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # First, map input features to hidden size.
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        # Then learn short and mid-range time patterns with TCN.
        self.tcn = TCNBlock(
            channels=hidden_dim,
            kernel_size=tcn_kernel_size,
            levels=tcn_levels,
            dropout=dropout,
        )

        # Transformer helps the model use longer-range context.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Decode hidden features back to original input size.
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: [batch, seq_len, features]
        # Conv1d expects channels first: [batch, features, seq_len].
        channel_first_inputs = inputs.transpose(1, 2)
        projected_features = self.input_projection(channel_first_inputs)
        tcn_features = self.tcn(projected_features)
        # Transformer expects sequence first: [batch, seq_len, hidden_dim].
        sequence_features = tcn_features.transpose(1, 2)
        transformer_features = self.transformer(sequence_features)
        outputs = self.decoder(transformer_features)
        return outputs
