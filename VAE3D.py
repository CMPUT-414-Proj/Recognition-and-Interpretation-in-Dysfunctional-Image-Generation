from __future__ import print_function

import argparse
import torch
import torch.utils.data
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor



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


def _plot3DLatentSpace(Z):
    """
         Plot the 3D latent space  distributions of the latent variable Z and the corresponding decoded marginal/central samples;
    :param
        - Z, Tensor, input latent variable of data points taken from the VAE latent space;
    """
    # Extract the first 60000 labels from the training dataset and reshape to a 2D array;
    trainLabels = trainingData.targets[:60000].reshape((-1, 1))
    labels = np.array([int(item[0]) for item in trainLabels.numpy()])
    # Initialize a 3D plot with a (8x8) inch figure;
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Define the plotting range for the axes;
    SCOPE = 6
    ax.set_xlim((-SCOPE, SCOPE))
    ax.set_ylim((-SCOPE, SCOPE))
    ax.set_zlim((-SCOPE, SCOPE))

    # Size Checking;
    print(Z.shape)
    print(len(Z[:, 0]), len(Z[:, 1]), len(Z[:, 2]))
    # Extract the first three dimensions of Z and convert to NumPy arrays for plotting;
    a = np.array(Z[:, 0].detach().reshape((-1, 1)), dtype=list)
    b = np.array(Z[:, 1].detach().reshape((-1, 1)), dtype=list)
    c = np.array(Z[:, 2].detach().reshape((-1, 1)), dtype=list)
    # Create a 3D scatter plot of the latent variables with labels as colors;
    ax.scatter(a, b, c, edgecolors="black", c=labels)
    samples = []

    #print(samples)
    print("VAE decoding...")
    # If sample points are available, plot and annotate them in the 3D plot;
    for idx, sample in enumerate(samples):
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], color="white", edgecolor="black")
        ax.text(sample[:, 0], sample[:, 1], sample[:, 2], f"{str(idx+1)}", color="red")
    plt.show()

    # Decode each sample point and display the corresponding image
    for idx, sample in enumerate(samples):
        sampleRec = model.decode(Tensor(sample))
        # Assuming _toImg creates a 2D image from the sample
        plt.figure()
        plt.imshow(_toImg(sampleRec), cmap='gray')
        plt.title(f'{str(idx+1)}')
        plt.show()

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

    if MODE == '3':
        print(">> Do you want to get the GMM fitting outcome? [(Y)es, (N)o]")
        G = str(input("> ")).lower().strip()
        if G == "y" or G == "yes":
            # After all batches are processed, fit a Gaussian Mixture Model (GMM) to the latent variables.
            GMM = GaussianMixture(n_components=10)
            GMM.fit(Z.detach().numpy())
            # Prepare the grid for 3D data
            x = np.linspace(Z.detach().numpy()[:, 0].min(), Z.detach().numpy()[:, 0].max(), num=100)
            y = np.linspace(Z.detach().numpy()[:, 1].min(), Z.detach().numpy()[:, 1].max(), num=100)
            z = np.linspace(Z.detach().numpy()[:, 2].min(), Z.detach().numpy()[:, 2].max(), num=100)
            X, Y, Z_grid = np.meshgrid(x, y, z)
            XYZ = np.array([X.ravel(), Y.ravel(), Z_grid.ravel()]).T

            # Get the score_samples from GMM and reshape for visualization
            Z_pred = np.exp(GMM.score_samples(XYZ))
            Z_pred = Z_pred.reshape(X.shape)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # Scatter plot of the actual data points in latent space
            ax.scatter(Z.detach().numpy()[:, 0], Z.detach().numpy()[:, 1], Z.detach().numpy()[:, 2])
            # Plot the mean of each GMM component as a point
            for i in range(GMM.n_components):
                mean = GMM.means_[i]
                ax.scatter(mean[0], mean[1], mean[2], c='red', marker='x')
            ax.set_title('GMM components in 3D')
            plt.show()

        print(">> Do you want to get the kNN fitting outcome? [(Y)es, (N)o]")
        K = str(input("> ")).lower().strip()
        if K == "y" or K == "yes":
            knn = KNeighborsClassifier(n_neighbors=3)
            trainLabels = trainingData.targets[:60000].reshape((-1, 1))
            labels = np.array([int(item[0]) for item in trainLabels.numpy()])
            knn.fit(Z.detach().numpy(), labels)
            # Plotting only the data points in 3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # Scatter plot of the data points with true labels
            scatter = ax.scatter(Z.detach().numpy()[:, 0], Z.detach().numpy()[:, 1], Z.detach().numpy()[:, 2], c=labels,
                                 edgecolors='k', cmap=plt.cm.coolwarm)

            # Creating a legend with the unique labels
            unique_labels = np.unique(labels)
            legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend1)
            plt.title('kNN decision boundaries in 3D')
            plt.show()

        # Prompt the user to decide if they want to perform marginal sampling;
        print(">> Do you want to process visualization (and sampling, *may cause potential error to label points in 3D space)? [(Y)es/(N)o]")
        marginal = input("> ").lower().strip()
        if marginal == "y" or marginal == "yes":
            _plot3DLatentSpace(Z)

    # Print the average training loss for the epoch and save the training results;
    print('>> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    try:
        torch.save(model.state_dict(), 'vae_3d.pth')
    except:
        print(">> SavingError: Fail to store the trained model (vae_3d.pth) to the parent directory of `\proj`;")


def test():
    """
        Evaluate the VAE model on the test set and report the final loss;
    """
    # # Set the model to evaluation mode which turns off dropout and batch normalization layers.
    model.eval()
    test_loss = 0
    # Disable gradient calculation as it is not needed for evaluation, which saves memory and computations.
    with torch.no_grad():
        # Iterate over the test dataset. The loader returns batches of data and their respective labels
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, _ = model(data)
            test_loss += _lossFunction(recon_batch, data, mu, logvar).item()
            # (optional) For the first batch, perform additional operations to visualize the reconstruction quality.
            if i == 0:
                n = min(data.size(0), data.size(1))
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
    # Calculate the average test loss by dividing the total test loss by the number of samples in the test dataset.
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def generate(roundNum):
    """
        (Optional) Generate new images from random latent space samples. Used for Data Composition Analysis (DCA);
    :param:
        - roundNum: int, the dimensionality of the latent space;
    """
    with torch.no_grad():
        sample = torch.randn(60, roundNum).to(device)
        sample = model.decode(sample).cpu()
        idx = 0
        for img in sample:
            print(f'3d_{str(idx)}.png')
            save_image(img.view(1, 28, 28), f'3d_{str(idx)}.png')
            idx += 1



if __name__ == "__main__":
    """ Learning Settings """
    # Basic settings
    N = 3
    MODEL = 'vae_3d.pth'
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


    """ Model Preparation """
    model = VAE().to(device)
    try:
        # Reload pre-trained model if possible;
        model.load_state_dict(torch.load(MODEL))
        print(">> Model Reloaded Successfully.")
    except:
        print(">> ReloadError: Incorrect model path or no valid model exists;")
        exit()
    # Adam optimizer ready;
    optimizer = optim.Adam(model.parameters(), lr=1e-5)


    """ Mode Selection """
    print("Please select the operation. \n\t * 1 --> Continue Training; \n\t * 2 --> Tests; \n\t * 3 --> Visualization & Sampling;")
    MODE = str(input("> ")).lower().strip()
    # Training Mode
    if MODE == '1':
        print(">> The mode you selected is '1 - Continue Training', the default training #epoch is 100")
        input(">> The mode is designed to continue training a pretrained model, using VAE.py if no pretrained model (vae_2d.pth) exists [press to continue]\n\n")
        epochNum = 100
        for epoch in range(0, epochNum):
            train(epoch)
    # Test Mode
    elif MODE == '2':
        print("* The mode you selected is '2 - Test'")
        test()
    # Visualization & Sampling Mode
    elif MODE == '3':
        print("* The mode you selected is '3 - Visualization & Sampling'")
        epochNum = 1
        for epoch in range(0, epochNum):
            train(epoch)
    else:
        print(">> Incorrect command, please try again;")
        exit()
