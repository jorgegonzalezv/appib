import torch 
from src.model import NeuralNetwork, NeuralNetworkAdaptative, device
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def roc(threshold):
    # evaluation (test) dataloader
    t_test = transforms.Compose([
        transforms.ToTensor(),
        ])
    test_path = "refactor_Task_2_3/evaluation"
    test_ds = ImageFolder(test_path,transform=t_test)
    test_ds_dataloader = DataLoader(test_ds, batch_size=1)

    # model
    model = NeuralNetworkAdaptative()

    # set weights from saved file
    model.load_state_dict(torch.load("weights/model_e20.pth"))
    # roc
    size = len(test_ds_dataloader.dataset)
    model.eval()

    TPR = 0.
    FPR = 0.
    P = 0.
    N = 0.
    correct = 0.
    with torch.no_grad():
        for X, y in test_ds_dataloader:
            X, y = X.to(device), y.to(device)
            y = y.float()
            pred = model(X)
            pred = pred.unsqueeze(-1)
            if y.item() == 0:
                N += 1.
            if y.item() == 1:
                P += 1.

            if ((pred > threshold) == y):
                correct +=1

            if ((pred > threshold) == y).type(torch.float).sum().item() and (y.item() == 1):
                TPR += 1.

            if ((pred > threshold) == y).type(torch.float).sum().item() and (y.item() == 0):
                FPR += 1.
    print("Accuracy: ", correct/(P+N))

    print(TPR, P)
    print(FPR, N)
    TPR /= P
    FPR /= N
    return TPR, FPR

if __name__== '__main__':
    N =30
    thresholds = np.linspace(0,1,N)

    # roc
    tprs = []
    fprs = []
    auc_aprox = 0
    print("calculando roc...")
    for t in thresholds:  
        tpr, fpr = roc(t)
        print(tpr,fpr)
        tprs.append(tpr)
        fprs.append(fpr)
        auc_aprox += tpr

    print("...ya!")
    # auc
    auc_aprox /= N

    plt.title('ROC, AUC(aprox)='+str(auc_aprox))    
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fprs,tprs)
    #plt.show()
    plt.savefig('plot.png')