import numpy as np
from helpers.train import train_model
from helpers.optimizers.random_search import PARAM_RANGES

BOUNDS = np.array([
    PARAM_RANGES["n_filters"],
    (3, 7),                                  # kernel_size
    PARAM_RANGES["lstm_units"],
    PARAM_RANGES["dropout"],
    (np.log10(PARAM_RANGES["lr"][0]), np.log10(PARAM_RANGES["lr"][1])),
    (0, len(PARAM_RANGES["batch_size_options"]) - 1),  
], dtype=np.float64)


def position_to_config(pos):
    n_filters = int(np.clip(pos[0], *BOUNDS[0]))
    kernel_size = int(np.clip(round(pos[1]), *BOUNDS[1]))
    if kernel_size % 2 == 0:  
        kernel_size += 1
    lstm_units = int(np.clip(pos[2], *BOUNDS[2]))
    dropout = float(np.clip(pos[3], *BOUNDS[3]))
    lr = float(10 ** np.clip(pos[4], *BOUNDS[4]))
    bs_idx = int(np.clip(round(pos[5]), *BOUNDS[5]))
    batch_size = PARAM_RANGES["batch_size_options"][bs_idx]

    return {
        "n_filters": n_filters, "kernel_size": kernel_size,
        "lstm_units": lstm_units, "dropout": dropout,
        "lr": lr, "batch_size": batch_size,
    }


def pso(n_particles, n_iterations, X_train, y_train, mask_train,
        X_val, y_val, mask_val, max_epochs=3, device="cpu",
        w=0.6, c1=1.5, c2=1.5):

    dim = len(BOUNDS)
    positions = np.array([
        np.random.uniform(BOUNDS[d][0], BOUNDS[d][1], n_particles) for d in range(dim)
    ]).T
    velocities = np.zeros((n_particles, dim))

    pbest_positions = positions.copy()
    pbest_scores = np.full(n_particles, -np.inf)

    gbest_position = None
    gbest_score = -np.inf
    history = []

    for it in range(n_iterations):
        for i in range(n_particles):
            config = position_to_config(positions[i])
            score = train_model(config, X_train, y_train, mask_train,
                                 X_val, y_val, mask_val, max_epochs=max_epochs, device=device)

            if score > pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i].copy()

            if score > gbest_score:
                gbest_score = score
                gbest_position = positions[i].copy()

        history.append(gbest_score)
        print(f"[PSO] Iter {it+1}/{n_iterations} | best so far: {gbest_score:.4f}")

        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_positions - positions)
            + c2 * r2 * (gbest_position - positions)
        )
        positions = positions + velocities

        for d in range(dim):
            positions[:, d] = np.clip(positions[:, d], BOUNDS[d][0], BOUNDS[d][1])

    best_config = position_to_config(gbest_position)
    return best_config, gbest_score, history