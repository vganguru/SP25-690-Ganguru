import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy(train, val, name):
    plt.plot(train)
    plt.plot(val)
    plt.title(name)
    plt.savefig(f"outputs/{name}.png")
    plt.close()

def plot_confusion(cm, name):
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig(f"outputs/{name}.png")
    plt.close()

def plot_bar(a, b):
    plt.bar(["MLP", "Transformer"], [a, b])
    plt.savefig("outputs/model_comparison.png")
    plt.close()