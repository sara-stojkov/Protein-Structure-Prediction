import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim=21, n_filters=64, kernel_size=5,
                 lstm_units=128, dropout=0.3, n_classes=3):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # "same" padding, čuva seq_len
        )
        self.bn = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=n_filters,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(dropout)

        self.fc = nn.Linear(lstm_units * 2, n_classes)  # *2 zbog bidirectional

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)          # -> (batch, input_dim, seq_len) za Conv1d
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = x.transpose(1, 2)          # -> (batch, seq_len, n_filters) za LSTM

        x, _ = self.lstm(x)            # -> (batch, seq_len, lstm_units*2)
        x = self.dropout2(x)

        out = self.fc(x)               # -> (batch, seq_len, n_classes)
        return out