from __future__ import print_function

import argparse
import torch
import torch.utils.data
from sklearn.neighbors import KNeighborsClassifier
from torch import nn, optim, Tensor
from torch.nn import functional as F
import numpy as np
from scipy.stats import multivariate_normal
from copy import deepcopy
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from torchvision import datasets, transforms
from torchvision.utils import save_image



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


def _plot2DMarginals(Z):
    """
         Plot the 2D latent space distributions of the latent variable Z and the corresponding decoded marginal samples;
    :param
        - Z, Tensor, input latent variable of data points taken from the VAE latent space;
    """
    # Extract the first 60000 labels from the training dataset and reshape to a 2D array;
    trainLabels = trainingData.targets[:60000].reshape((-1, 1))
    labels = np.array([int(item[0]) for item in trainLabels.numpy()])

    # Set up a 6x6 inch figure for plotting;
    plt.figure(figsize=(6, 6))
    SCOPE = 10
    plt.xlim((-SCOPE, SCOPE))
    plt.ylim((-SCOPE, SCOPE))
    a = Z[:, 0].detach().numpy()
    b = Z[:, 1].detach().numpy()
    plt.scatter(a, b, edgecolors="black", c=labels, alpha=0.7, s=20)

    # Taking marginal samples with different means and standard deviations;
    samples = [
        sample_1 := np.array([np.random.normal(0.6, 0.1), np.random.normal(-2.8, 0.15)]).reshape(1, 2),
        sample_2 := np.array([np.random.normal(0.6, 0.1), np.random.normal(-2.8, 0.15)]).reshape(1, 2),

        sample_3 := np.array([np.random.normal(-1, 0.15), np.random.normal(-3, 0.15)]).reshape(1, 2),
        sample_4 := np.array([np.random.normal(-1, 0.15), np.random.normal(-3, 0.15)]).reshape(1, 2),

        sample_5 := np.array([np.random.normal(-4, 0.3), np.random.normal(1, 0.3)]).reshape(1, 2),
        sample_6 := np.array([np.random.normal(-4, 0.3), np.random.normal(1, 0.3)]).reshape(1, 2),

        sample_7 := np.array([np.random.normal(-1.75, 0.3), np.random.normal(3.75, 0.3)]).reshape(1, 2),
        sample_8 := np.array([np.random.normal(-1.75, 0.3), np.random.normal(3.75, 0.3)]).reshape(1, 2),

        sample_9 := np.array([np.random.normal(3.75, 0.3), np.random.normal(2, 0.3)]).reshape(1, 2),
        sample_10 := np.array([np.random.normal(3.75, 0.3), np.random.normal(2, 0.3)]).reshape(1, 2)
    ]
    print(samples)
    print("VAE decoding...")
    # Plot each sample on the scatter plot with white color and a black edge and annotate the samples;
    for idx, sample in enumerate(samples):
        plt.scatter(sample[:, 0], sample[:, 1], color='white', edgecolor="black", alpha=0.7, s=20)
        plt.annotate(f"{str(idx+1)}", (sample[:, 0] + 0.1, sample[:, 1] + 0.1), color="red")
    plt.show()

    # Decode each sample using the model's decode function and display the result;
    for idx, sample in enumerate(samples):
        sampleRec = model.decode(Tensor(sample))
        plt.figure()
        plt.imshow(_toImg(sampleRec), cmap='gray')
        plt.title(f'{str(idx+1)}')
        plt.show()


def _plot2DCentrals(Z):
    """
         Plot the 2D latent space distributions of the latent variable Z and the corresponding decoded central samples;
    :param
        - Z, Tensor, input latent variable of data points taken from the VAE latent space;
    """
    # Extract the first 60000 labels from the training dataset and reshape to a 2D array;
    trainLabels = trainingData.targets[:60000].reshape((-1, 1))
    labels = np.array([int(item[0]) for item in trainLabels.numpy()])

    # Set up a 6x6 inch figure for plotting;
    plt.figure(figsize=(6, 6))
    SCOPE = 10
    plt.xlim((-SCOPE, SCOPE))
    plt.ylim((-SCOPE, SCOPE))
    a = Z[:, 0].detach().numpy()
    b = Z[:, 1].detach().numpy()
    plt.scatter(a, b, edgecolors="black", c=labels, alpha=0.7, s=20)

    # Taking central samples with different means and standard deviations;
    samples = [
    # Cluster 1 (Left Up)
        sample_1 := np.array([np.random.normal(-3.5, 0.3), np.random.normal(2.5, 0.3)]).reshape(1, 2),
        sample_2 := np.array([np.random.normal(-2.5, 0.3), np.random.normal(1.5, 0.3)]).reshape(1, 2),
    # Cluster 2 (Left Down)
        sample_3 := np.array([np.random.normal(-3, 0.3), np.random.normal(-2.3, 0.3)]).reshape(1, 2),
        sample_4 := np.array([np.random.normal(-2.5, 0.3), np.random.normal(-1.5, 0.3)]).reshape(1, 2),
    # Cluster 3  (Middle Up - blue)
        sample_5 := np.array([np.random.normal(-0.5, 0.1), np.random.normal(3.5, 0.2)]).reshape(1, 2),
        sample_6 := np.array([np.random.normal(-0.5, 0.1), np.random.normal(2.5, 0.2)]).reshape(1, 2),
    # Cluster 4  (Middle Up - yellow)
        sample_7 := np.array([np.random.normal(0, 0.1), np.random.normal(3.5, 0.2)]).reshape(1, 2),
        sample_8 := np.array([np.random.normal(0, 0.1), np.random.normal(2.5, 0.2)]).reshape(1, 2),
    # Cluster 5   (Middle Down)
        sample_9 := np.array([np.random.normal(0, 0.2), np.random.normal(-2, 0.3)]).reshape(1, 2),
        sample_10 := np.array([np.random.normal(0, 0.2), np.random.normal(-3, 0.3)]).reshape(1, 2),
    # Cluster 6   (Right Up)
        sample_11 := np.array([np.random.normal(2, 0.3), np.random.normal(3, 0.3)]).reshape(1, 2),
        sample_12 := np.array([np.random.normal(2.5, 0.3), np.random.normal(2.6, 0.3)]).reshape(1, 2),
    # Cluster 7   (Right Down)
        sample_13 := np.array([np.random.normal(2.5, 0.3), np.random.normal(-2.8, 0.15)]).reshape(1, 2),
        sample_14 := np.array([np.random.normal(3, 0.3), np.random.normal(0, 0.15)]).reshape(1, 2),
    # Gray Area 8   (Right Down)
        sample_15 := np.array([np.random.normal(1.3, 0.1), np.random.normal(-1.5, 0.15)]).reshape(1, 2),
        sample_16 := np.array([np.random.normal(1.3, 0.1), np.random.normal(-1, 0.15)]).reshape(1, 2),
        sample_17 := np.array([np.random.normal(0, 0.3), np.random.normal(0, 0.15)]).reshape(1, 2),
        sample_18 := np.array([np.random.normal(0, 0.3), np.random.normal(0, 0.15)]).reshape(1, 2),
    ]
    print(samples)
    print("VAE decoding...")
    # Plot each sample on the scatter plot with white color and a black edge and annotate the samples;
    for idx, sample in enumerate(samples):
        plt.scatter(sample[:, 0], sample[:, 1], color='white', edgecolor="black", alpha=0.7, s=20)
        plt.annotate(f"{str(idx+1)}", (sample[:, 0] + 0.1, sample[:, 1] + 0.1), color="red")
    plt.show()

    # Decode each sample using the model's decode function and display the result;
    for idx, sample in enumerate(samples):
        sampleRec = model.decode(Tensor(sample))
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
            x = np.linspace(Z.detach().numpy()[:, 0].min(), Z.detach().numpy()[:, 0].max(), num=100)
            y = np.linspace(Z.detach().numpy()[:, 1].min(), Z.detach().numpy()[:, 1].max(), num=100)
            X, Y = np.meshgrid(x, y)
            XX = np.array([X.ravel(), Y.ravel()]).T
            Z_pred = np.exp(GMM.score_samples(XX))
            Z_pred = Z_pred.reshape(X.shape)

            # Plot the density and contour of the GMM components over the latent space.
            plt.contour(X, Y, Z_pred)
            plt.scatter(Z.detach().numpy()[:, 0], Z.detach().numpy()[:, 1])
            for i in range(GMM.n_components):
                mean = GMM.means_[i]
                covar = GMM.covariances_[i]
                multi_normal = multivariate_normal(mean, covar)
                Z_multi = multi_normal.pdf(np.dstack((X, Y)))
                plt.contour(X, Y, Z_multi, colors='k', alpha=0.5)
            plt.title('GMM components')
            plt.show()

        print(">> Do you want to get the kNN fitting outcome? [(Y)es, (N)o]")
        K = str(input("> ")).lower().strip()
        if K == "y" or K == "yes":
            # Preparing the model of knn and feed data into it;
            knn = KNeighborsClassifier(n_neighbors=3)
            trainLabels = trainingData.targets[:60000].reshape((-1, 1))
            labels = np.array([int(item[0]) for item in trainLabels.numpy()])
            knn.fit(Z.detach().numpy(), labels)
            x_min, x_max = Z.detach().numpy()[:, 0].min() - 1, Z.detach().numpy()[:, 0].max() + 1
            y_min, y_max = Z.detach().numpy()[:, 1].min() - 1, Z.detach().numpy()[:, 1].max() + 1
            # Mapping the knn space expression onto a plot, then print it;
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                                 np.linspace(y_min, y_max, 300))
            Z_pred = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            plt.contourf(xx, yy, Z_pred, alpha=0.5, cmap=plt.cm.coolwarm)
            plt.scatter(Z.detach().numpy()[:, 0], Z.detach().numpy()[:, 1], c=labels, edgecolors='k', cmap=plt.cm.coolwarm)
            plt.title('kNN decision boundaries in 2D')
            plt.show()

        # Prompt the user to decide if they want to perform marginal sampling;
        print(">> Do you want to process marginal sampling? [(Y)es/(N)o]")
        marginal = input("> ").lower().strip()
        if marginal == "y" or marginal == "yes":
            _plot2DMarginals(Z)

        # Prompt the user to decide if they want to perform central sampling;
        print(">> Do you want to process central sampling?  [(Y)es/(N)o]")
        central = input("> ").lower().strip()
        if central == "y" or central == "yes":
            _plot2DCentrals(Z)

    # Print the average training loss for the epoch and save the training results;
    print('>> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    try:
        torch.save(model.state_dict(), 'vae_2d.pth')
    except:
        print(">> SavingError: Fail to store the trained model (vae_2d.pth) to the current directory of `\proj`;")


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
            print(f'2d_{str(idx)}.png')
            save_image(img.view(1, 28, 28), f'2d_{str(idx)}.png')
            idx += 1



if __name__ == "__main__":
    """ Learning Settings """
    # Basic settings
    N = 2
    MODEL = 'vae_2d.pth'
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
        print(">> The mode you selected is '1 - Training', the default training #epoch is 100")
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
