import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import multiprocessing
multiprocessing.set_start_method('spawn', True)
scale = 0.1
torch.manual_seed(190234995)
transform = transforms.Compose(
    [transforms.ToTensor(),
     ])





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    pass
    net = Net().cuda()
    testset = torchvision.datasets.CIFAR10(root="/home/twsf/data/Cifar/", train=False,
                                            download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=4)
    correct = 0
    total = 0
    time_used=0
    temptime=0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.cuda())
            time_start_predict=time.time()
            _, predicted = torch.max(outputs.data, 1)
            time_finish_predict=time.time()
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()
            temptime+=time_finish_predict-time_start_predict
            time_used=temptime
            mat=confusion_matrix(labels,predicted)
            print(mat)
    print(temptime)
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))