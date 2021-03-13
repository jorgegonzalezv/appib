import torch 
from model import NeuralNetwork
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

"""
    Podriamos incluir:
    - custom transformation que capture info chula de la imagen
"""

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == '__main__':

    # data loader
    t_ = transforms.Compose([
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            ])

    train_path = "Task_1/development"
    train_ds = ImageFolder(train_path, transform=t_)


    train_ds_loader = DataLoader(train_ds)

    #for im, tag in train_ds_loader:
    #    print(tag)

    # model
    model = NeuralNetwork()

    # model loss
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


