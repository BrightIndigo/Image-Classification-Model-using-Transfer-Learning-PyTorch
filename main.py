import torchvision.datasets as sets
import torch
import torchvision.transforms as trans
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
import random

transform = trans.Compose([
    trans.ToTensor(), 
    trans.Normalize((0.5,), (0.5,)) 
])

mnist = sets.MNIST(root="./data", download=True, train=True, transform=transform)

train_loader = DataLoader(mnist, batch_size=64, shuffle=True)

rand_int = random.randint(0, 50000)
image, label = mnist[50000]

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.show()