import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
multiprocessing.set_start_method('spawn', True)
scale = 0.1
torch.manual_seed(190234995)
transform = transforms.Compose(
    [transforms.ToTensor(),
     ])
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

trainset_src = torchvision.datasets.CIFAR10(root="/home/twsf/data/Cifar/", train=True,
                                            download=True, transform=transform)

testset_src = torchvision.datasets.CIFAR10(root="/home/twsf/data/Cifar/", train=False,
                                           download=True, transform=transform)

testset_gt = torch.tensor(testset_src.data).permute(0, 3, 1, 2).float() / 255
noise = (scale * torch.randn(len(testset_gt), 3, 32, 32))
testset_noise = testset_gt + noise

def addsalt_pepper(imgs):
    C1, C2, C3, C4 = imgs.shape
    limit = 256 if imgs.max() > 1 else 1
    noise_salt = np.random.randint(0, limit, (C1, C2, C3, C4))
    noise_pepper = np.random.randint(0, limit, (C1, C2, C3, C4))
    rand = 0.1
    noise_salt = torch.tensor(np.where(noise_salt < rand * limit, limit, 0)).type_as(imgs)
    noise_pepper = torch.tensor(np.where(noise_pepper < rand * limit, -limit, 0)).type_as(imgs)
    imgs_noise = imgs + noise_salt + noise_pepper
    return imgs_noise.clamp(min=0, max=limit)

def addGauss(imgs):
    C1, C2, C3, C4 = imgs.shape
    limit = 256 if imgs.max() > 1 else 1
    Gauss_noise = torch.tensor(np.random.normal(0, 50/(257-limit), (C1, C2, C3, C4))).type_as(imgs)
    imgs_noise = imgs + Gauss_noise
    return imgs_noise.clamp(min=0, max=limit)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # 1 input image channel, 16 output channel, 3x3 square convolution
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  #to range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def test(model, epoch):
    print("Test.................")
    model.eval()
    with torch.no_grad():
        criterion = nn.MSELoss()
        losses = []
        outputs = []
        for index, input in enumerate(tqdm(testset_noise)):
            target = testset_gt[index].unsqueeze(0).to(device)
            input = input.unsqueeze(0).to(device)
            pred = model(input)

            loss = criterion(target, pred)
            losses += [float(loss.cpu())]
            outputs.append((target, pred),)
        temp = sum(losses) / len(losses)
        print("Test mean loss: {}".format(temp))
    print('Test done..................')
    # save model
    check = {
            'epoch': epoch + 1,
            'model': model.state_dict()
        }
    filename = "epoch_{}.pth".format(epoch+1)
    torch.save(check, filename)
    if epoch >= max_epochs - 1:
        plt.figure(figsize=(20, 12))
        indexes = np.argsort(-np.array(losses))[:30].tolist()
        for ii, index in enumerate(indexes):
            plt.subplot(6, 10, 2 * ii + 1)
            gt, pred = outputs[index]
            # show ground thruth
            gt = gt.cpu().squeeze(0).permute(1, 2, 0).numpy()
            plt.imshow(gt)
            # show predict
            pred = pred.cpu().squeeze(0).permute(1, 2, 0).numpy()
            pred = np.clip(pred, 0, 1)
            plt.subplot(6, 10, 2 * ii + 2)
            plt.imshow(pred)
        plt.savefig('result1.png')
        plt.show()
    return temp


def train(model, num_epochs=5, batch_size=64, learning_rate=1e-3):
    model.train()
    trainset_gt = torch.tensor(trainset_src.data).permute(0, 3, 1, 2).float() / 255
    trainset_noise = addGauss(trainset_gt)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    test_result = []
    for epoch in range(num_epochs):
        for index in tqdm(range(len(trainset_noise) // batch_size)):
            start = index * batch_size
            end = (index + 1) * batch_size
            targets = trainset_gt[start:end].to(device)
            inputs = trainset_noise[start:end].to(device)

            recon = model(inputs)

            targets = targets.to(device)
            loss = criterion(recon, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Train loss:{:.4f}'.format(epoch+1, float(loss.cpu())))

        temp = test(model, epoch)
        test_result.append(temp)
    return test_result


max_epochs = 30
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
model = Autoencoder().to(device)
# test(model, 30)
test_result = train(model, num_epochs=max_epochs)
x = [i+1 for i in range(len(test_result))]
plt.plot(x, test_result, 'b^-', label='mse')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.savefig("result2.png")
plt.show()
