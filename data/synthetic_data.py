import numpy as np
from config import SEQ_LEN, FEATURE_DIM

def generate_sample(label):
    base_speed = [0.3, 0.6, 0.9][label]
    noise = np.random.normal(0, 0.05, (SEQ_LEN, FEATURE_DIM))
    signal = np.ones((SEQ_LEN, FEATURE_DIM)) * base_speed
    return signal + noise

def generate_dataset(n=2000):
    X = []
    y = []
    for i in range(n):
        label = np.random.randint(0, 3)
        X.append(generate_sample(label))
        y.append(label)
    return np.array(X), np.array(y)