import numpy as np
from helpers.train import train_model
from helpers.optimizers.random_search import sample_random_config, PARAM_RANGES


def mutate(config, mutation_rate=0.3):
    new_config = config.copy()
    if np.random.rand() < mutation_rate:
        new_config["n_filters"] = np.random.randint(*PARAM_RANGES["n_filters"])
    if np.random.rand() < mutation_rate:
        new_config["kernel_size"] = np.random.choice([3, 5, 7])
    if np.random.rand() < mutation_rate:
        new_config["lstm_units"] = np.random.randint(*PARAM_RANGES["lstm_units"])
    if np.random.rand() < mutation_rate:
        new_config["dropout"] = np.random.uniform(*PARAM_RANGES["dropout"])
    if np.random.rand() < mutation_rate:
        new_config["lr"] = 10 ** np.random.uniform(
            np.log10(PARAM_RANGES["lr"][0]), np.log10(PARAM_RANGES["lr"][1]))
    if np.random.rand() < mutation_rate:
        new_config["batch_size"] = np.random.choice(PARAM_RANGES["batch_size_options"])
    return new_config


def crossover(parent1, parent2):
    """Uniform crossover: svaki gen (hiperparametar) nasumično dolazi od jednog roditelja."""
    child = {}
    for key in parent1.keys():
        child[key] = parent1[key] if np.random.rand() < 0.5 else parent2[key]
    return child


def tournament_selection(population, fitnesses, k=3):
    """Bira najboljeg od k nasumično izabranih jedinki."""
    idx = np.random.choice(len(population), size=k, replace=False)
    best_idx = idx[np.argmax([fitnesses[i] for i in idx])]
    return population[best_idx]


def genetic_algorithm(pop_size, n_generations, X_train, y_train, mask_train,
                       X_val, y_val, mask_val, max_epochs=3, device="cpu",
                       mutation_rate=0.3, elitism=1):
    # 1. Inicijalizacija populacije
    population = [sample_random_config() for _ in range(pop_size)]
    fitnesses = [
        train_model(cfg, X_train, y_train, mask_train, X_val, y_val, mask_val,
                    max_epochs=max_epochs, device=device)
        for cfg in population
    ]

    best_overall_acc = max(fitnesses)
    best_overall_config = population[int(np.argmax(fitnesses))]
    history = [best_overall_acc]

    print(f"[GA] Gen 0 | best so far: {best_overall_acc:.4f}")

    for gen in range(1, n_generations + 1):
        # 2. Elitizam: zadrži najbolje jedinke direktno
        sorted_idx = np.argsort(fitnesses)[::-1]
        new_population = [population[i] for i in sorted_idx[:elitism]]

        # 3. Popuni ostatak populacije selekcijom + ukrštanjem + mutacijom
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        fitnesses = [
            train_model(cfg, X_train, y_train, mask_train, X_val, y_val, mask_val,
                        max_epochs=max_epochs, device=device)
            for cfg in population
        ]

        gen_best_acc = max(fitnesses)
        if gen_best_acc > best_overall_acc:
            best_overall_acc = gen_best_acc
            best_overall_config = population[int(np.argmax(fitnesses))]

        history.append(best_overall_acc)
        print(f"[GA] Gen {gen} | best this gen: {gen_best_acc:.4f} | best overall: {best_overall_acc:.4f}")

    return best_overall_config, best_overall_acc, history