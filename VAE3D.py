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



N = 3
MODEL = 'weights/vae_3d_epoch_100.pth'

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

# construct a VAE
class VAE(nn.Module):
    """
        Construct a VAE
    """
    def __init__(self):
        """
            Construct the architecture of VAE
        """
        super(VAE, self).__init__()
        # input layer
        self.fc1 = nn.Linear(784, 400)
        # output layer of mu
        self.fc21 = nn.Linear(400, N)  # 20
        # output layer of mu
        self.fc22 = nn.Linear(400, N)  # 20
        # input layer of decoder
        self.fc3 = nn.Linear(N, 400)  # 20
        # input layer of decoder
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        """
            Define the encoding process
            :param:
            - x, input data of the model;
            :return:
            - self.fc21(h1), mean of the latent Gaussian;
            - self.fc22(h2), standard deviation of the latent Gaussian;
        """
        h1 = F.relu(self.fc1(x))
        # return mu, logvar
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, log_var):
        """
            Reparameterization trick to sample from N(mu, var) from N(0,1)
            :param:
            - mu, mean of the latent Gaussian;
            - log_var, standard deviation of the latent Gaussian;
            :return:
            - mu + eps * std, latent variable
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
            Define the decoding process; Map the given latent codes onto the image space.
            :param:
            - z, latent variable;
            :return:
            - torch.sigmoid(self.fc4(h3)), out put data of decoder
        """
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """
            Forward propagation function
            :param:
            - x, input data of the model;
            :return:
            - x_out, output data of decoder;
            - mu, mean of the latent Gaussian;
            - log_var, standard deviation of the latent Gaussian;
            - z, latent variable;
        """
        mu, log_var = self.encode(x.reshape(-1, 784))
        z = self.reparameterize(mu, log_var)
        x_out = self.decode(z)
        return x_out, mu, log_var, z

    def getZ(self, x):
        """
            Get the sampling variable Z
            :param:
            - x, input data of the model;
            :return:
            - z, latent variable;
        """
        mu, log_var = self.encode(x.reshape(-1, 784))
        z = self.reparameterize(mu, log_var)
        return z


def loss_function(recon_x, x, mu, logvar, beta=1):
    """
        Reconstruction + KL divergence losses summed over all elements and batch
        :param:
        - recon_x, decoder-generated reconstructed data。
        - x, input data;
        - mu, mean of the latent Gaussian;
        - log_var, standard deviation of the latent Gaussian;
        - beta, Weighting factors for weighing reconstruction error and KL dispersion;
        :return:
        - BCE_loss + beta * KLD, weighted sum of reconstruction error and KL scattering;
    """
    BCE_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE_loss + beta * KLD

def _removeOutliers(data, r=10):
    """
        Remove outliers
        :param:
        - data, input data;
        - r, parameters for outlier detection, to control the percentage of outliers
        :return:
        - filtered: data after removal of outliers;
    """
    outlierRatio = r / 2
    # IQR = Q3 - Q1
    IQR = np.percentile(data, 100 - outlierRatio) - np.percentile(data, outlierRatio)
    # lower = Q1 - 1.5 * IQR
    lower = np.percentile(data, outlierRatio) - 1.5 * IQR
    # lower = Q3 + 1.5 * IQR
    upper = np.percentile(data, 100 - outlierRatio) + 1.5 * IQR

    filtered = deepcopy(data)
    filtered[(filtered < lower)] = 0
    filtered[(filtered > upper)] = 0
    return filtered


def _toImg(x):
    """
        Convert tensor x to a NumPy array representing an image
        :param:
        - x, input data;
        :return:
        - image, image with the form of array;
    """
    return x.cpu().detach().numpy().reshape((28, 28))


def _plot2DLatentSpacePlus(Z):
    """
        Plot the latent space of 2D VAE
    """
    # Get the label
    trainLabels = trainingData.targets[:60000].reshape((-1, 1))
    labels = np.array([int(item[0]) for item in trainLabels.numpy()])

    plt.figure(figsize=(6, 6))
    SCOPE = 10
    plt.xlim((-SCOPE, SCOPE))
    plt.ylim((-SCOPE, SCOPE))
    a = Z[:, 0].cpu().detach().numpy()
    b = Z[:, 1].cpu().detach().numpy()
    plt.scatter(a, b, edgecolors="black", c=labels, alpha=0.7, s=20)
    # Generate Samples
    samples = [
        sample_1 := np.array(
            [np.random.normal(0.6, 0.1), np.random.normal(-2.8, 0.15), np.random.normal(0, 0.15)]).reshape(1, 3),
        sample_2 := np.array(
            [np.random.normal(0.6, 0.1), np.random.normal(-2.8, 0.15), np.random.normal(0, 0.15)]).reshape(1, 3),

        sample_3 := np.array(
            [np.random.normal(-1, 0.15), np.random.normal(-3, 0.15), np.random.normal(0, 0.15)]).reshape(1, 3),
        sample_4 := np.array(
            [np.random.normal(-1, 0.15), np.random.normal(-3, 0.15), np.random.normal(0, 0.15)]).reshape(1, 3),

        sample_5 := np.array([np.random.normal(-4, 0.3), np.random.normal(1, 0.3), np.random.normal(0, 0.15)]).reshape(
            1, 3),
        sample_6 := np.array([np.random.normal(-4, 0.3), np.random.normal(1, 0.3), np.random.normal(0, 0.15)]).reshape(
            1, 3),

        sample_7 := np.array(
            [np.random.normal(-1.75, 0.3), np.random.normal(3.75, 0.3), np.random.normal(0, 0.15)]).reshape(1, 3),
        sample_8 := np.array(
            [np.random.normal(-1.75, 0.3), np.random.normal(3.75, 0.3), np.random.normal(0, 0.15)]).reshape(1, 3),

        sample_9 := np.array(
            [np.random.normal(3.75, 0.3), np.random.normal(2, 0.3), np.random.normal(0, 0.15)]).reshape(1, 3),
        sample_10 := np.array(
            [np.random.normal(3.75, 0.3), np.random.normal(2, 0.3), np.random.normal(0, 0.15)]).reshape(1, 3),
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
    """
        Plot the latent space of 3D VAE
    """
    # Get the label
    trainLabels = trainingData.targets[:60000].reshape((-1, 1))
    labels = np.array([int(item[0]) for item in trainLabels.numpy()])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Set the axis range
    SCOPE = 6
    ax.set_xlim((-SCOPE, SCOPE))
    ax.set_ylim((-SCOPE, SCOPE))
    ax.set_zlim((-SCOPE, SCOPE))

    #print(Z.shape)
    #print(len(Z[:, 0]), len(Z[:, 1]), len(Z[:, 2]))
    a = np.array(Z[:, 0].detach().cpu().reshape((-1, 1)), dtype=list)
    b = np.array(Z[:, 1].detach().cpu().reshape((-1, 1)), dtype=list)
    c = np.array(Z[:, 2].detach().cpu().reshape((-1, 1)), dtype=list)
    # Scatterplotting
    ax.scatter(a, b, c, edgecolors="black", c=labels)
    # Generate Samples
    samples = []
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
    ]
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

# Traning
def train(epoch):
    """
        Train VAE models
    """
    model.train()
    training_loss = 0
    if epoch == 1:
        with open("trainingset.txt", "w") as file:
            for item in trainingData.test_labels:
                file.write(f"{item}\n")
    Z = None
    X = []
    # Training loop
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
    # GMM fitting
    GMM = GaussianMixture(n_components=N)
    GMM.fit(Z.cpu().detach().numpy())

    #_plot3DLatentSpacePlus(Z)
    _plot2DLatentSpacePlus(Z)

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, training_loss / len(train_loader.dataset)))
    torch.save(model.state_dict(), oj("./weights", f'vae_3d_epoch_{epoch}.pth'))
    print("save successfully!!!")

# Testing
def test(epoch):
    """
        Evaluate the performance of the model on the test set at the end of each training cycle
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if epoch == 1:
                # Save test set label
                with open("testset.txt", "w") as file:
                    for item in testData.test_labels:
                        file.write(f"{item}\n")

            data = data.to(device)
            batch, mu, log_var, _ = model(data)
            # Calculate test loss
            test_loss += loss_function(batch, data, mu, log_var).item()
            # Compare and save images
            if i == 0:
                n = min(data.size(0), data.size(1))
                comparison = torch.cat([data[:n], batch.view(args.batch_size, 1, 28, 28)[:n]])
                if epoch == 1:
                    save_image(comparison.cpu(), oj(out_dir, 'reconstruction_' + str(epoch) + '.png'), nrow=n)
                    dimension = "3d"
                    # Save weights
                    torch.save(model.state_dict(), oj("./weights", f'vae_{dimension}_epoch_{epoch}.pth'))
    # Calculate the average loss
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # Load training data
    trainingData = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    # load test data
    testData = datasets.MNIST('data', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainingData, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(testData, batch_size=args.batch_size, shuffle=False, **kwargs)
    # Set the output directory to save the generated image
    out_dir = 'samples3'
    os.makedirs(out_dir, exist_ok=True)

    file_path = "./weights/vae_3d_epoch_100.pth"
    if os.path.exists(file_path):
        print("aaaaaaaaaaaa")
        MODEL = 'weights/vae_3d_epoch_100.pth'
        model = VAE().to(device)
        model.load_state_dict(torch.load(MODEL))
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    else:
        print("bbbbbbbbbbbbb")
        model = VAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        for epoch in range(1,501):
            train(epoch)
        MODEL = 'weights/vae_3d_epoch_100.pth'
        model.load_state_dict(torch.load(MODEL))


    model = VAE().to(device)
    model.load_state_dict(torch.load(MODEL))
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # actually do training
    for epoch in range(1, 501):
        #train(epoch)
        test(epoch)
        # Save images
        with torch.no_grad():
            sample = torch.randn(60, N).to(device)
            sample = model.decode(sample).cpu()
            idx = 0
            for img in sample:
                print('new_' + str(epoch) + '_' + str(idx) + '.png')
                save_image(img.view(1, 28, 28), oj(out_dir, 'new_' + str(epoch) + '_' + str(idx) + '.png'))
                idx += 1
