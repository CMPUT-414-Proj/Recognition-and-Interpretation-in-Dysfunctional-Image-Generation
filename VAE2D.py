from __future__ import print_function
import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from os.path import join as oj
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from torch import Tensor
from sklearn import manifold

from visuals import imscatter

is3D = True

N = 2
MODEL = 'weights/vae_2d_epoch_100.pth'
# if is3D:
#     N = 3
#     MODEL = 'weights/vae_3d_epoch_100.pth'

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=1,
                    metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='enables CUDA training')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval',
                    type=int,
                    default=10,
                    metavar='N',
                    help='how many batches to wait before logging training status')


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, N)  # 20
        self.fc22 = nn.Linear(400, N)  # 20
        self.fc3 = nn.Linear(N, 400)  # 20
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        # return mu, logvar
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, log_var = self.encode(x.reshape(-1, 784))
        z = self.reparameterize(mu, log_var)
        x_out = self.decode(z)
        return x_out, mu, log_var, z

    def getZ(self, x):
        mu, log_var = self.encode(x.reshape(-1, 784))
        z = self.reparameterize(mu, log_var)
        return z


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta=1):
    BCE_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE_loss + beta * KLD


def _removeOutliers(data, r=10):
    outlierRatio = r / 2
    IQR = np.percentile(data, 100 - outlierRatio) - np.percentile(data, outlierRatio)
    lower_threshold = np.percentile(data, outlierRatio) - 1.5 * IQR
    upper_threshold = np.percentile(data, 100 - outlierRatio) + 1.5 * IQR

    filtered = deepcopy(data)
    filtered[(filtered < lower_threshold)] = 0
    filtered[(filtered > upper_threshold)] = 0
    return filtered


def _toImg(x):
    return x.cpu().detach().numpy().reshape((28, 28))


def _plot2DLatentSpacePlus(Z):
    trainLabels = trainingData.targets[:60000].reshape((-1, 1))
    labels = np.array([int(item[0]) for item in trainLabels.numpy()])

    plt.figure(figsize=(6, 6))
    SCOPE = 10
    plt.xlim((-SCOPE, SCOPE))
    plt.ylim((-SCOPE, SCOPE))
    a = Z[:, 0].detach().numpy()
    b = Z[:, 1].detach().numpy()
    plt.scatter(a, b, edgecolors="black", c=labels, alpha=0.7, s=20)

    samples = [
        # Cluster 1 (Left Up) #todo
        sample_1 := np.array([np.random.normal(-3.5, 0.3), np.random.normal(2.5, 0.3)]).reshape(1, 2),
        sample_2 := np.array([np.random.normal(-2.5, 0.3), np.random.normal(1.5, 0.3)]).reshape(1, 2),
        # Cluster 2 (Left Down) #todo
        sample_3 := np.array([np.random.normal(-3, 0.3), np.random.normal(-2.3, 0.3)]).reshape(1, 2),
        sample_4 := np.array([np.random.normal(-2.5, 0.3), np.random.normal(-1.5, 0.3)]).reshape(1, 2),
        # Cluster 3  (Middle Up - blue) #todo
        sample_5 := np.array([np.random.normal(-0.5, 0.1), np.random.normal(3.5, 0.2)]).reshape(1, 2),
        sample_6 := np.array([np.random.normal(-0.5, 0.1), np.random.normal(2.5, 0.2)]).reshape(1, 2),
        # Cluster 4  (Middle Up - yellow) #todo
        sample_7 := np.array([np.random.normal(0, 0.1), np.random.normal(3.5, 0.2)]).reshape(1, 2),
        sample_8 := np.array([np.random.normal(0, 0.1), np.random.normal(2.5, 0.2)]).reshape(1, 2),
        # Cluster 5   (Middle Down) #todo
        sample_9 := np.array([np.random.normal(0, 0.2), np.random.normal(-2, 0.3)]).reshape(1, 2),
        sample_10 := np.array([np.random.normal(0, 0.2), np.random.normal(-3, 0.3)]).reshape(1, 2),
        # Cluster 6   (Right Up) #todo
        sample_11 := np.array([np.random.normal(2, 0.3), np.random.normal(3, 0.3)]).reshape(1, 2),
        sample_12 := np.array([np.random.normal(2.5, 0.3), np.random.normal(2.6, 0.3)]).reshape(1, 2),
        # Cluster 7   (Right Down) #todo
        sample_13 := np.array([np.random.normal(2.5, 0.3), np.random.normal(-2.8, 0.15)]).reshape(1, 2),
        sample_14 := np.array([np.random.normal(3, 0.3), np.random.normal(0, 0.15)]).reshape(1, 2),
        # Cluster 8   (Right Down) #todo
        sample_15 := np.array([np.random.normal(1.3, 0.1), np.random.normal(-1.5, 0.15)]).reshape(1, 2),
        sample_16 := np.array([np.random.normal(1.3, 0.1), np.random.normal(-1, 0.15)]).reshape(1, 2),
        sample_17 := np.array([np.random.normal(0, 0.3), np.random.normal(0, 0.15)]).reshape(1, 2),
        sample_18 := np.array([np.random.normal(0, 0.3), np.random.normal(0, 0.15)]).reshape(1, 2),
    ]
    # samples = []
    print(samples)
    print("VAE decoding...")
    for idx, sample in enumerate(samples):
        plt.scatter(sample[:, 0], sample[:, 1], color='white', edgecolor="black", alpha=0.7, s=20)
        plt.annotate(f"{str(idx + 1)}", (sample[:, 0] + 0.1, sample[:, 1] + 0.1), color="red")
    plt.show()

    # Decode and display each sample
    for idx, sample in enumerate(samples):
        sampleRec = model.decode(Tensor(sample))
        plt.figure()
        plt.imshow(_toImg(sampleRec), cmap='gray')
        plt.title(f'{str(idx + 1)}')
        plt.show()


def _plot3DLatentSpacePlus(Z):
    trainLabels = trainingData.targets[:60000].reshape((-1, 1))
    labels = np.array([int(item[0]) for item in trainLabels.numpy()])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    SCOPE = 6
    ax.set_xlim((-SCOPE, SCOPE))
    ax.set_ylim((-SCOPE, SCOPE))
    ax.set_zlim((-SCOPE, SCOPE))

    print(Z.shape)
    print(len(Z[:, 0]), len(Z[:, 1]), len(Z[:, 2]))
    a = np.array(Z[:, 0].detach().reshape((-1, 1)), dtype=list)
    b = np.array(Z[:, 1].detach().reshape((-1, 1)), dtype=list)
    c = np.array(Z[:, 2].detach().reshape((-1, 1)), dtype=list)
    ax.scatter(a, b, c, edgecolors="black", c=labels)
    samples = []
    """
    samples = [
        sample_1 := np.array([np.random.normal(0.6, 0.1), np.random.normal(-2.8, 0.15), np.random.normal(0, 0.15)]).reshape(1, 3),
        sample_2 := np.array([np.random.normal(0.6, 0.1), np.random.normal(-2.8, 0.15), np.random.normal(0, 0.15)]).reshape(1, 3),

        sample_3 := np.array([np.random.normal(-1, 0.15), np.random.normal(-3, 0.15), np.random.normal(0, 0.15)]).reshape(1, 3),
        sample_4 := np.array([np.random.normal(-1, 0.15), np.random.normal(-3, 0.15), np.random.normal(0, 0.15)]).reshape(1, 3),

        sample_5 := np.array([np.random.normal(-4, 0.3), np.random.normal(1, 0.3), np.random.normal(0, 0.15)]).reshape(1, 3),
        sample_6 := np.array([np.random.normal(-4, 0.3), np.random.normal(1, 0.3), np.random.normal(0, 0.15)]).reshape(1, 3),

        sample_7 := np.array([np.random.normal(-1.75, 0.3), np.random.normal(3.75, 0.3), np.random.normal(0, 0.15)]).reshape(1, 3),
        sample_8 := np.array([np.random.normal(-1.75, 0.3), np.random.normal(3.75, 0.3), np.random.normal(0, 0.15)]).reshape(1, 3),

        sample_9 := np.array([np.random.normal(3.75, 0.3), np.random.normal(2, 0.3), np.random.normal(0, 0.15)]).reshape(1, 3),
        sample_10 := np.array([np.random.normal(3.75, 0.3), np.random.normal(2, 0.3), np.random.normal(0, 0.15)]).reshape(1, 3),
    ]"""

    print(samples)
    print("VAE decoding...")
    for idx, sample in enumerate(samples):
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], color="white", edgecolor="black")
        ax.text(sample[:, 0], sample[:, 1], sample[:, 2], f"{str(idx + 1)}", color="red")
    plt.show()

    # Decode and display each sample
    for idx, sample in enumerate(samples):
        sampleRec = model.decode(Tensor(sample))
        plt.figure()
        plt.imshow(_toImg(sampleRec), cmap='gray')  # Assuming _toImg creates a 2D image from the sample
        plt.title(f'{str(idx + 1)}')
        plt.show()


def train(epoch):
    model.train()
    training_loss = 0
    if epoch == 1:
        with open("trainingset.txt", "w") as file:
            for item in trainingData.test_labels:
                file.write(f"{item}\n")
    Z = None
    X = []
    for batch_index, (data, _) in enumerate(train_loader):
        X.append(data)
        data = data.to(device)
        optimizer.zero_grad()
        batch, mu, log_var, z = model(data)
        Z = z if batch_index == 0 else torch.vstack((Z, z))
        loss = loss_function(batch, data, mu, log_var)
        loss.backward()
        training_loss += loss.item()
        optimizer.step()
        if batch_index % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                       100. * batch_index / len(train_loader),
                       loss.item() / len(data)))
    GMM = GaussianMixture(n_components=N)
    GMM.fit(Z.detach().numpy())

    # _plot3DLatentSpacePlus(Z)
    _plot2DLatentSpacePlus(Z)

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, training_loss / len(train_loader.dataset)))
    torch.save(model.state_dict(), oj("./weights", f'vae_2d_epoch_n.pth'))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if epoch == 1:
                with open("testset.txt", "w") as file:
                    for item in testData.test_labels:
                        file.write(f"{item}\n")

            data = data.to(device)
            batch, mu, log_var, _ = model(data)
            test_loss += loss_function(batch, data, mu, log_var).item()
            if i == 0:
                n = min(data.size(0), data.size(1))
                comparison = torch.cat([data[:n], batch.view(args.batch_size, 1, 28, 28)[:n]])
                if epoch == 1:
                    save_image(comparison.cpu(), oj(out_dir, 'reconstruction_' + str(epoch) + '.png'), nrow=n)
                    a = "2d"
                    # if is3D:
                    #     a = "3d"
                    torch.save(model.state_dict(), oj("./weights", f'vae_{a}_epoch_{epoch}.pth'))

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    trainingData = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    testData = datasets.MNIST('data', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainingData, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(testData, batch_size=args.batch_size, shuffle=False, **kwargs)
    out_dir = 'samples2'
    os.makedirs(out_dir, exist_ok=True)

    model = VAE().to(device)
    model.load_state_dict(torch.load(MODEL))
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # actually do training
    for epoch in range(1, 501):
        # train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(60, N).to(device)
            sample = model.decode(sample).cpu()
            idx = 0
            for img in sample:
                print('new_' + str(epoch) + '_' + str(idx) + '.png')
                save_image(img.view(1, 28, 28), oj(out_dir, 'new_' + str(epoch) + '_' + str(idx) + '.png'))
                idx += 1
