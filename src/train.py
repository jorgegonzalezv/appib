"""
    Training code based on 
    https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

import torch 
from torch import nn
from model import NeuralNetwork, device
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

"""
    Podriamos incluir:
    - custom transformation que capture info chula de la imagen
    - rgb -> un solo channel con info interesante (ver papels y tal)
    - a las malas, captura de features con PCA, DCT o algo y cambiamos a un modelo MLP
    - optical flow, aplicar al dataset (opencv)
    - otra es entrenar un modelo de AE y utilizar el latent space como input a un MLP, random forest, ... etc
    - quitar data-augmentation para no liarla?
    - split train dataset para validation?
    - TODO: probar modelo que este aprendiendo con otro dataset!!!!!
    - AWS: cuando sepa lo anterior meterlo chicha con AWS al entreno
    - añadir dropout
    - TODO: sacar features manualemente y entrenar MLP
    - momentum!
    - TODO: monitorizar train accuracy!
"""

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y= y.float()
        # Compute prediction error
        pred = model(X)
        #print("pred:", pred)
        #print("target:",y)
        loss = loss_fn(pred, y)

        #print("loss", loss)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, threshold=0.5):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.float()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += ((pred > threshold) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':

    epochs = 200
    batch_size = 4
    resize = (256, 256)

    # TODO: normalizar colores, resize mas pequeño... etc
    # train data loader
    t_train = transforms.Compose([
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Resize(resize) #,interpolation='bilinear'),
            ])

    train_path = "Task_1/development"
    train_ds = ImageFolder(train_path, transform=t_train)
    train_ds_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for im, tag in train_ds_loader:
        print(im.shape)
        print(torch.max(im))
        #plt.imshow(im[0].permute(1,2,0))
        #plt.show()
        break

    # evaluation (test) dataloader
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize)# ,interpolation=transforms.InterpolationMode.BILINEAR),
        ])
    test_path = "Task_1/evaluation"
    test_ds = ImageFolder(test_path,transform=t_test)
    test_ds_dataloader = DataLoader(test_ds, batch_size=batch_size)

    # model
    model = NeuralNetwork()
    #print(model)
    model.load_state_dict(torch.load("weights/model_e20.pth"))
    
    #pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print("Numero de parametros: " ,pytorch_total_params)
    
    # model loss
    # cross-entropy, ya que vamos a predecir probabilidades
    # de clasificar en fake o no fake
    loss_fn = nn.BCELoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    
    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        train(train_ds_loader, model, loss_fn, optimizer)
        test(test_ds_dataloader, model, loss_fn)
        
        if e%10 == 0:
            torch.save(model.state_dict(), "weights/model_e"+str(e)+".pth")
            print("Saved PyTorch Model State to weights")

    torch.save(model.state_dict(), "weights/model_e"+str(200)+".pth")
