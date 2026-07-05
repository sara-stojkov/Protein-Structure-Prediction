import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from helpers.train import train_model
from helpers.optimizers.random_search import PARAM_RANGES

SEARCH_SPACE = [
    Integer(PARAM_RANGES["n_filters"][0], PARAM_RANGES["n_filters"][1], name="n_filters"),
    Categorical([3, 5, 7], name="kernel_size"),
    Integer(PARAM_RANGES["lstm_units"][0], PARAM_RANGES["lstm_units"][1], name="lstm_units"),
    Real(PARAM_RANGES["dropout"][0], PARAM_RANGES["dropout"][1], name="dropout"),
    Real(PARAM_RANGES["lr"][0], PARAM_RANGES["lr"][1], prior="log-uniform", name="lr"),
    Categorical(PARAM_RANGES["batch_size_options"], name="batch_size"),
]


def bayesian_optimization(n_calls, X_train, y_train, mask_train,
                           X_val, y_val, mask_val, max_epochs=3, device="cpu"):

    history = []

    def objective(params):
        config = {
            "n_filters": params[0],
            "kernel_size": params[1],
            "lstm_units": params[2],
            "dropout": params[3],
            "lr": params[4],
            "batch_size": params[5],
        }
        acc = train_model(config, X_train, y_train, mask_train,
                           X_val, y_val, mask_val, max_epochs=max_epochs, device=device)
        history.append(acc)
        print(f"[Bayes] eval {len(history)}/{n_calls} | acc={acc:.4f} | config={config}")
        return -acc  # gp_minimize minimizes, while we maximize accuracy

    result = gp_minimize(
        objective, SEARCH_SPACE,
        n_calls=n_calls, random_state=42, verbose=False
    )

    best_config = {
        "n_filters": result.x[0], "kernel_size": result.x[1],
        "lstm_units": result.x[2], "dropout": result.x[3],
        "lr": result.x[4], "batch_size": result.x[5],
    }
    best_acc = -result.fun

    # history do sad je "trenutna" tačnost po evaluaciji; za krivu konvergencije
    # nam treba "najbolja do sad" (running max) da bude uporedivo sa GA/PSO
    running_best = np.maximum.accumulate(history)

    return best_config, best_acc, running_best.tolist()