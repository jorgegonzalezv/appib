# eliminar imagen sin cara!
# mejorar/simplificar red!! mucho!!! 100.000 params que si no se aprende el dataset

# face extraction
# data augmentation
# batch = 8
# resize con padding lo minimo posible

# preprocess dataset
# 1. face extraction/background
# 2. fast fourier
# 3. simple model

# simple models
# pca/freqs/colors...
# knn

# orange baseline

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np 

from scipy.fft import fft, ifft

train_path = "Task_1/development"
train_ds = ImageFolder(train_path)
#train_ds_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

"""for im, tag in train_ds:
    im = np.array(im)
    print(im)
    plt.imshow(im)
    plt.show()
    f_im = fft(im)
    print(f_im)
    plt.imshow(np.real(f_im) /255)
    plt.show()
    break
"""
for im, tag in train_ds: