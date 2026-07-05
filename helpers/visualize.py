import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_convergence(histories_dict, title="Comparison of optimization methods by Q3 accuracy"):
    """histories_dict: {"GA": [...], "PSO": [...], "Bayes": [...], "Random Search": [...]}"""
    plt.figure(figsize=(8, 5))
    for name, history in histories_dict.items():
        plt.plot(range(1, len(history) + 1), history, marker="o", label=name)
    plt.xlabel("Number of evaluations")
    plt.ylabel("Best Q3 accuracy so far")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_confusion_matrix(cm, class_names=("H", "E", "C"), title="Confusion matrix - final model on CB513"):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()