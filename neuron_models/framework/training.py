import torch
import torch.nn as nn
import numpy as np

class Trainer(object):
    '''...'''
    def __init__(self, model, dataloader, device=None, dtype=None, *args, **kwargs):
        ### Hardware parameters.
        if device is None:
            self.device = torch.device("cpu") # "cpu", or "cuda:0".
        else:
            self.device = device

        if dtype is None:
            self.dtype = torch.float64
        else:
            self.dtype = dtype

        ### Model.
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.model.train()

        ### Dataloader
        self.dataloader = dataloader

    def _data_convert(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        #x = x.to(device=self.device, dtype=self.dtype)
        #y = y.to(device=self.device, dtype=self.dtype)

        return x, y

    def run(self, *args, **kwargs):
        ### Training parameters.
        if 'loss_fn' in kwargs:
            loss_fn = kwargs['loss_fn']
        else:
            loss_fn = nn.functional.nll_loss

        if 'lr' in kwargs:
            lr = kwargs['lr']
        else:
            lr = 1e-2

        if 'optimizer' in kwargs:
            optimizer_fn = kwargs['optimizer']
        else:
            optimizer_fn = torch.optim.SGD

        loss_rec  = torch.Tensor().to(device=self.device, dtype=self.dtype)
        optimizer = optimizer_fn(self.model.parameters(), lr=lr)

        ### Train loop.
        for x, y in self.dataloader:
            x, y = self._data_convert(x, y)

            optimizer.zero_grad()
            out  = self.model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            loss_rec = np.append(loss_rec, loss.item())

        return self.model, loss_rec