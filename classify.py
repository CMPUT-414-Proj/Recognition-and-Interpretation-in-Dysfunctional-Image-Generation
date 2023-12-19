import torch
import torchvision
from torchvision.transforms import ToTensor
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import os


# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=ToTensor(), download=True)
# Create data loader for training set
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Extract features and labels from the data loader
X_train, y_train = next(iter(train_loader))
X_train = X_train.reshape(X_train.shape[0], -1).numpy()
y_train = y_train.numpy()

# Train KNN model
k = 3  # Number of neighbors to consider
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)


def predict_labels(test_dir):
    """
        This is a helper function that helps classify images inside the given directory. It assigns labels for each test
    image based on a ``training set`` from the MNIST original data;
    :param
        - test_dir: The directory of the image set for classification;
    :return
        - predictions, the categories defined;
    """
    predictions = []
    for filename in os.listdir(test_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(test_dir, filename)
            # Convert image to grayscale;
            image = Image.open(image_path).convert("L")
            # Transform image to a PyTorch tensor;
            image = ToTensor()(image)
            # Flatten the tensor and convert to numpy array;
            image = image.reshape(1, -1).numpy()
            # Make label prediction by kNN;
            prediction = knn.predict(image)
            predictions.append(prediction[0])
    return predictions

# Provide the path to your test set directory
print(">> Please insert the image directory path")
test_dir = input("> ").strip().lower()

# Get predictions for test images
predictions = predict_labels(test_dir)

# Output the predicted labels
print("Predicted Labels:")
for label in predictions:
    print(label)