"""The sleep-stage model: a CNN reads each 30-second epoch, a GRU reads the sequence of epochs."""

import torch
import torch.nn as nn

import config


def _conv_block(in_channels: int, out_channels: int, kernel_size: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )


class EpochEncoder(nn.Module):
    """Turns one epoch of raw signal, (channels, 3000 samples), into a feature vector."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # A wide first filter (half a second at 100 Hz) picks up the slow sleep rhythms.
            _conv_block(config.N_CHANNELS, 64, 50, stride=6, padding=22),
            nn.MaxPool1d(8),
            nn.Dropout(dropout),
            _conv_block(64, 128, 8, padding=4),
            _conv_block(128, 128, 8, padding=4),
            _conv_block(128, 128, 8, padding=4),
            nn.MaxPool1d(4),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            self.out_features = self.net(torch.zeros(2, config.N_CHANNELS, config.EPOCH_SAMPLES)).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SleepStager(nn.Module):
    def __init__(self, hidden_size: int = config.HIDDEN_SIZE, dropout: float = config.DROPOUT) -> None:
        super().__init__()
        self.encoder = EpochEncoder(dropout)
        # Bidirectional, so each epoch is scored using what came before and after it.
        self.gru = nn.GRU(self.encoder.out_features, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_size, len(config.STAGE_NAMES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, sequence, channels, samples) -> logits: (batch, sequence, stages)."""
        batch, sequence = x.shape[:2]
        features = self.encoder(x.flatten(0, 1)).view(batch, sequence, -1)
        output, _ = self.gru(features)
        return self.head(self.dropout(output))
