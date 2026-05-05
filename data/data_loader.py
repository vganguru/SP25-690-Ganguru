import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from data.synthetic_data import generate_dataset
from config import BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT

def get_loaders():
    X, y = generate_dataset()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)

    n = len(dataset)
    train_size = int(TRAIN_SPLIT * n)
    val_size = int(VAL_SPLIT * n)
    test_size = n - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE),
        DataLoader(test_ds, batch_size=BATCH_SIZE)
    )
