from __future__ import print_function

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from torchvision import datasets, transforms


class VAE(nn.Module):
    def __init__(self):
        """
            Initialize the Variational Autoencoder (VAE) model with fully connected layers;
        """
        super(VAE, self).__init__()

        # Input layer to hidden layer:
        self.fc1 = nn.Linear(784, 400)
        #  Hidden layer to mean of latent space 2-dim space:
        self.fc21 = nn.Linear(400, N)
        # Hidden layer to log variance of latent space 2-dim space:
        self.fc22 = nn.Linear(400, N)
        # Latent space to hidden layer for decoding:
        self.fc3 = nn.Linear(N, 400)
        # # Hidden layer to output layer (reconstruction)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        """
            Encodes the input by passing it through the encoder network and returning the mean and log variance of the
        latent space distribution;
        :param:
            - x: Tensor, the input image data;
        :return:
            - self.fc21(h1): the mean of the encoded input;
            - self.fc22(h1): the log variance of the encoded input;
        """
        # Apply a rectified linear unit activation function, then return the mean and log variance;
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """
            Applies the reparameterization trick by sampling from the standard normal distribution and scaling it by the
        standard deviation and shifting it by the mean.
        :param:
            - mu: Tensor, the mean of the latent space;
            - logvar: Tensor, the log variance of the latent space;
        :return:
            - (mu + eps * std), Tensor, the sampled latent vector;
        """
        # Compute the parameters based on the VAE design
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
            Decodes the latent space vector by passing it through the decoder network and returning the reconstructed input.
        :param:
            - z: Tensor, the latent space vector;
        :return:
            - torch.sigmoid(self.fc4(h3)), Tensor, the probability distribution of the reconstructed input;
        """
        # Apply a rectified linear unit activation function, then apply a sigmoid activation function;
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """
            Defines the forward pass of the VAE;
        :param
            - x: Tensor, the input image data;
        :return:
            - (self.decode(z), mu, logvar, z), tuple, containing the reconstructed image, mean and log variance of the
            latent space, and the sampled latent vector;
          """
        # Encode the input, then reparameterize to sample from latent space:
        mu, logvar = self.encode(x.reshape(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def getC(self, x):
        """
            Encodes the input data and returns the sampled latent vector;
        :param:
            - x: Tensor, the input image data;
        :return:
            - C, Tensor, the sampled latent vector C;
        """
        # Encode the input, then reparameterize to sample from latent space;
        mu, logvar = self.encode(x.reshape(-1, 784))
        C = self.reparameterize(mu, logvar)
        return C

def _lossFunction(x_bar, x, mu, logvar, beta=1):
    """
        Compute the loss function for a VAE which includes a reconstruction loss term (BCE) and a KL divergence term (KLD),
    weighted by a factor beta;
    :param
        - x_bar: Tensor, reconstructed images;
        - x: Tensor, original images;
        - mu: Tensor, mean of the latent space distribution;
        - logvar: Tensor, log variance of the latent space distribution;
        - beta: float, default=1, weighting factor for KLD;
    :return:
        - Tensor, the computed loss value;
    """
    # Compute the overall loss based on each part of BCE and KLD (weighted, to avoid vanishing):
    BCE = F.binary_cross_entropy(x_bar, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

def _removeOutliers(data, r=10):
    """
        Optional helper function, that help remove outliers from data using IQR method;
    :params:
        - data: np.array, input data;
        - r: float, default=10, percentage to determine the outlier ratio;
    :return:
        - filtered, np.array, data with outliers removed;
    """
    # Compute the outlier thresholds:
    outlierRatio = r / 2
    Q1 = np.percentile(data, outlierRatio)
    Q3 = np.percentile(data, 100-outlierRatio)
    IQR = Q3 - Q1
    lower_threshold = Q1 - 1.5 * IQR
    upper_threshold = Q3 + 1.5 * IQR

    # Process the filtering:
    filtered = deepcopy(data)
    filtered[(filtered < lower_threshold)] = 0
    filtered[(filtered > upper_threshold)] = 0
    return filtered

def _toImg(x):
    """
        Transform a flattened tensor back to an image shape.
    :param
        - x: Tensor, flattened image data;
    :return:
        - x.cpu().detach().numpy().reshape((28, 28)), np.array, image data reshaped to (28x28);
    """
    return x.cpu().detach().numpy().reshape((28, 28))


def train(epoch):
    """
        Train the VAE model for one epoch based on a specific epoch number;
    :param:
        - epoch: int, the current epoch number;
    """
    # Set the model to training and initialize the loss to zero;
    model.train()
    train_loss = 0
    Z = None
    X = []
    # Loop over each batch of data using the train_loader;
    for batch_idx, (data, _) in enumerate(train_loader):
        X.append(data)
        data = data.to(device)
        # Zero the gradients of the model parameters;
        optimizer.zero_grad()
        # Forward pass through the model to get the reconstructed images, the mean (mu) and log variance (logvar) of the
        #   latent variables, and the latent code (z);
        recon_batch, mu, logvar, z = model(data)
        Z = z if batch_idx == 0 else torch.vstack((Z, z))

        # Compute the loss using the custom loss function which includes the reconstruction loss and KL divergence;
        loss = _lossFunction(recon_batch, data, mu, logvar)
        # Backpropagate the loss to compute the gradients of the model parameters.
        loss.backward()
        # Add the current batch loss to the total training loss.
        train_loss += loss.item()
        # Update the model parameters using the optimizer.
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    # Print the average training loss for the epoch and save the training results;
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    try:
        if N == 2:
            torch.save(model.state_dict(), 'vae_2d.pth')
        else:
            torch.save(model.state_dict(), 'vae_3d.pth')
    except:
        print(">> SavingError: Fail to store the trained model to the parent directory of `\proj`;")



if __name__ == "__main__":
    """ Learning Settings """
    # Basic settings
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        metavar='N')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        metavar='N')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False)
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S')
    parser.add_argument('--log-interval',
                        type=int,
                        default=10,
                        metavar='N')
    # Device settings;
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


    """ Data Preparation """
    # Input data and set output path;
    trainingData = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    testData = datasets.MNIST('data', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainingData, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(testData, batch_size=args.batch_size, shuffle=False, **kwargs)


    """ Mode Selection """
    print("Please select the operation. \n\t * 2 --> Build 2D VAE; \n\t * 3 --> Build 3D VAE;")
    MODE = str(input("> ")).lower().strip()
    # Training Mode
    if MODE == '2':
        print("* The mode you selected is '2 - Build 2D VAE'")
        """ Model Preparation for 2D VAE """
        N = 2
        model = VAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # Visualization & Sampling Mode
    elif MODE == '3':
        print("* The mode you selected is '3 - Build 3D VAE")
        """ Model Preparation for 3D VAE """
        N = 3
        model = VAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    else:
        print(">> Incorrect command, please try again;")
        exit()

    # Start training for 100 epochs by default;
    input(">> *Warning: The measure may replace the existing pre-trained model with the same name; [press to continue]\n\n")
    epochNum = 100
    for epoch in range(0, epochNum):
        train(epoch)