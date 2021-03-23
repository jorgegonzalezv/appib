import torch 
from torch import nn
from model import NeuralNetwork, NeuralNetworkAdaptative, device
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

def train(dataloader, model, loss_fn, optimizer):
    threshold = 0.5
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y= y.float()
        # Compute prediction error
        pred = model(X)
        pred = pred.unsqueeze(-1)
        #print("pred:", pred)
        #print("target:",y)
        loss = loss_fn(pred, y)

        #print("loss", loss)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += ((pred > threshold) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct /= size
    print(f"Train: \n Accuracy: {(100*correct):>0.1f}%")

def test(dataloader, model, loss_fn, threshold=0.5):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.float()
            pred = model(X)
            pred = pred.unsqueeze(-1)
            test_loss += loss_fn(pred, y).item()
            correct += ((pred > threshold) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    epochs = 200
    batch_size = 1
    resize = (256, 256)

    # check path for weights
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    # train data loader
    # transfomations to data
    t_train = transforms.Compose([
            #transforms.RandomRotation(180),
            transforms.ToTensor(),
            #transforms.Resize(resize) #,interpolation='bilinear'),
            ])

    # create train dataloader
    train_path = "refactor_Task_1/development"
    train_ds = ImageFolder(train_path, transform=t_train)
    train_ds_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print(">>.",len(train_ds_loader.dataset))

    # evaluation (test) dataloader
    t_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Resize(resize)# ,interpolation=transforms.InterpolationMode.BILINEAR),
        ])
    test_path = "refactor_Task_1/evaluation"
    test_ds = ImageFolder(test_path,transform=t_test)
    test_ds_dataloader = DataLoader(test_ds, batch_size=batch_size)

    # model
    model = NeuralNetworkAdaptative()
    
    # model loss
    loss_fn = nn.BCELoss()

    # optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.8)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    
    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        train(train_ds_loader, model, loss_fn, optimizer)
        test(test_ds_dataloader, model, loss_fn)
        
        # save model each epoch 
        torch.save(model.state_dict(), "weights/model_e"+str(e)+".pth")
        print("Saved PyTorch Model State to weights")

    torch.save(model.state_dict(), "weights/model_e"+str(200)+".pth")
