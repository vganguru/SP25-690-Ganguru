import torch
import torch.nn as nn
import torch.optim as optim

from config import EPOCHS
from data.data_loader import get_loaders
from models.mlp import MLP
from models.transformer import TransformerModel
from training.trainer import Trainer
from utils.metrics import get_predictions, print_report, get_conf_matrix
from utils.plots import plot_accuracy, plot_confusion, plot_bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get_loaders()

def run(model, name):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device)

    train_acc = []
    val_acc = []

    for _ in range(EPOCHS):
        t = trainer.train_epoch(train_loader)
        v = trainer.evaluate(val_loader)
        train_acc.append(t)
        val_acc.append(v)

    plot_accuracy(train_acc, val_acc, name)

    preds, labels = get_predictions(model, test_loader, device)
    print_report(labels, preds)

    cm = get_conf_matrix(labels, preds)
    plot_confusion(cm, f"confusion_{name}")

    return val_acc[-1]

mlp_acc = run(MLP(), "mlp_accuracy")
trans_acc = run(TransformerModel(), "transformer_accuracy")

plot_bar(mlp_acc, trans_acc)

print("done")