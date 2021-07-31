import torch
import torch.nn as nn

class Tester(object):
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
        self.model = model
        self.model.eval()
        
        self.dataloader = dataloader
    
    def run(self, *args, **kwargs):
        ### Testing parameters.
        if 'loss_fn' in kwargs:
            loss_fn = kwargs['loss_fn']
        else:
            loss_fn = nn.functional.nll_loss

        test_loss = 0  # Total loss.
        avg_loss  = 0  # Average loss.
        correct   = 0  # Total correct predictions.
        
        ### Testing loop.
        with torch.no_grad():
            for x, y in self.dataloader:
                #data, target = data.to(device), target.to(device)
                out = self.model(x)
                test_loss += loss_fn(out, y, reduction='sum').item()
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()

        ### Output.
        total = len(self.dataloader.dataset)
        avg_loss = test_loss/len(self.dataloader.dataset)
        accuracy = 100.*correct/len(self.dataloader.dataset)

        return avg_loss, accuracy, (correct, total)