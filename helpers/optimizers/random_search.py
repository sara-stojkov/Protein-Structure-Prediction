import numpy as np
from helpers.train import train_model

PARAM_RANGES = {
    "n_filters": (16, 128),
    "kernel_size": (3, 7),
    "lstm_units": (32, 256),
    "dropout": (0.1, 0.5),
    "lr": (1e-4, 1e-2),          # log-scale
    "batch_size_options": [16, 32, 64],
}


def sample_random_config():
    return {
        "n_filters": np.random.randint(*PARAM_RANGES["n_filters"]),
        "kernel_size": np.random.choice([3, 5, 7]),
        "lstm_units": np.random.randint(*PARAM_RANGES["lstm_units"]),
        "dropout": np.random.uniform(*PARAM_RANGES["dropout"]),
        "lr": 10 ** np.random.uniform(np.log10(PARAM_RANGES["lr"][0]),
                                       np.log10(PARAM_RANGES["lr"][1])),
        "batch_size": np.random.choice(PARAM_RANGES["batch_size_options"]),
    }


def random_search(n_iter, X_train, y_train, mask_train, X_val, y_val, mask_val,
                   max_epochs=3, device="cpu"):
    history = []
    best_acc = -1
    best_config = None

    for i in range(n_iter):
        config = sample_random_config()
        acc = train_model(config, X_train, y_train, mask_train,
                           X_val, y_val, mask_val, max_epochs=max_epochs, device=device)
        history.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_config = config
        print(f"[RandomSearch {i+1}/{n_iter}] acc={acc:.4f} config={config}")

    return best_config, best_acc, history