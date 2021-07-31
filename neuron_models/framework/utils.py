import numpy as np
from torch.utils.data import Dataset, DataLoader

class NeuronDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

def neuron_dataloader(X, Y, batch_size=1, out_dimension=(1)):
    dataset = NeuronDataset(X,Y)
    
    def collate_fn(l):
        X = np.array([])
        Y = np.array([])
        for x, y in l:
            X = np.append(X, x)
            Y = np.append(Y, y)
        X = X.reshape(*out_dimension)
        Y = Y.reshape(*out_dimension)
        return X, Y

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)