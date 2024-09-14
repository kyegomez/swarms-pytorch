"""
Swarmalators with transformer models, SUPER EXPERIMENTAL, NEEDS WORK
"""

import torch
from torch import nn


class SwarmalatorModel(nn.Module):
    """
    # Example
    N = 100  # number of swarmalators
    D = 3  # dimensions

    model = SwarmalatorModel(N, D)
    positions, orientations = model()
    print(positions, orientations)
    """

    def __init__(
        self, N, D, nhead=8, num_encoder_layers=6, num_decoder_layers=6
    ):
        super(SwarmalatorModel, self).__init__()
        self.N = N
        self.D = D

        self.positions = nn.Parameter(torch.randn(N, D))
        self.orientations = nn.Parameter(torch.randn(N, D))

        # Transformer encoder to process positions and orientations
        encoder_layer = nn.TransformerEncoderLayer(d_model=D, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Transformer decoder to produce updated positions and orientations
        decoder_layer = nn.TransformerDecoderLayer(d_model=D, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

    def forward(self, src_mask=None, tgt_mask=None, memory_mask=None):
        # Using transformer encoder to get memory
        position_memory = self.transformer_encoder(
            self.positions.unsqueeze(1), mask=src_mask
        )
        orientation_memory = self.transformer_encoder(
            self.orientations.unsqueeze(1), mask=src_mask
        )
        # Using transformer decoder to get updated positions and orientations
        updated_positions = self.transformer_decoder(
            self.positions.unsqueeze(1),
            position_memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        updated_orientations = self.transformer_decoder(
            self.orientations.unsqueeze(1),
            orientation_memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )

        return updated_positions.squeeze(1), updated_orientations.squeeze(1)
