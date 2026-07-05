import torch
import torch.nn as nn
import numpy as np
from helpers.model import CNNLSTMModel
from helpers.metrics import q3_accuracy


def train_model(hyperparams, X_train, y_train, mask_train,
                 X_val, y_val, mask_val, max_epochs=5, device="cpu",
                 input_dim=21, return_model=False):
    """
    hyperparams: dictionary with keys:
        n_filters, kernel_size, lstm_units, dropout, lr, batch_size
    Returns: val_q3_accuracy (float), or (float, model) if return_model=True
    """
    model = CNNLSTMModel(
        input_dim=input_dim,
        n_filters=int(hyperparams["n_filters"]),
        kernel_size=int(hyperparams["kernel_size"]),
        lstm_units=int(hyperparams["lstm_units"]),
        dropout=hyperparams["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    batch_size = int(hyperparams["batch_size"])
    n_samples = X_train_t.shape[0]

    model.train()
    for epoch in range(max_epochs):
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = permutation[i:i + batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)

            optimizer.zero_grad()
            out = model(xb)                          # (batch, seq_len, 3)
            loss = criterion(out.reshape(-1, 3), yb.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # evaluate on validation set
    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        out = model(X_val_t)
        preds = out.argmax(dim=-1).cpu().numpy()

    val_acc = q3_accuracy(preds, y_val, mask_val)

    if return_model:
        return val_acc, model
    return val_acc