import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def get_predictions(model, loader, device):
    model.eval()
    preds = []
    labels = []

    import torch
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = out.argmax(dim=1).cpu().numpy()
            preds.extend(p)
            labels.extend(y.numpy())

    return np.array(preds), np.array(labels)

def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))

def get_conf_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
