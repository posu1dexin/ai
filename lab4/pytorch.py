import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

# Set random seed for reproducibility
torch.manual_seed(42)


# Define the augmentation transformations
augmentation_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # Random rotation by up to 10 degrees
    transforms.RandomHorizontalFlip(),       # Random horizontal flipping
    transforms.RandomVerticalFlip(),         # Random vertical flipping
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # Random crop and resize
])

# Define the transformation pipeline including augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    augmentation_transform  # Include augmentation transformations
])


train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(train_data))
valid_size = len(train_data) - train_size
train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

# Create data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


#Creation of CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Instantiate the model
model = CNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs for training
n_epochs = 10

# Path to save the best model
model_path = 'best_model.pth'


#Model training
def train(model, train_loader, valid_loader, criterion, optimizer, n_epochs, model_path):
    train_losses = []
    valid_losses = []
    train_accuracies = []  # Store training accuracy for each epoch
    valid_accuracies = []  # Store validation accuracy for each epoch
    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        correct_train = 0
        correct_valid = 0
        total_train = 0
        total_valid = 0

        # Training
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

        # Validation
        model.eval()
        for data, target in valid_loader:
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total_valid += target.size(0)
            correct_valid += (predicted == target).sum().item()

        # Average losses and accuracies
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        train_accuracy = correct_train / total_train
        valid_accuracy = correct_valid / total_valid
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        # Save the model if validation loss has decreased
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)

        print(f'Epoch {epoch+1}/{n_epochs}, '
              f'Train Loss: {train_loss:.6f}, '
              f'Valid Loss: {valid_loss:.6f}, '
              f'Train Accuracy: {train_accuracy:.2%}, '
              f'Valid Accuracy: {valid_accuracy:.2%}')

    # Plotting the training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, n_epochs + 1), valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    return train_losses, valid_losses

# Train the model
train_losses, valid_losses = train(model, train_loader, valid_loader, criterion, optimizer, n_epochs, model_path)


# Evaluate model performance with the testing dataset
def test(model, test_loader, criterion):
    test_loss = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    accuracy = correct / total

    print(f'Test Loss: {test_loss:.6f}, Accuracy: {accuracy:.2%}')

# Load the best model and evaluate on the test set
best_model = CNN()
best_model.load_state_dict(torch.load(model_path))
test(best_model, test_loader, criterion)

# Visualize dataset images
# Display sample images from the MNIST dataset
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Iterate over the train_loader to obtain batches of data
for images, labels in train_loader:
    # Show images
    imshow(torchvision.utils.make_grid(images))
    # Exit the loop after showing the first batch of images
    break

# Display prediction results of the best-performing model

def predict(model, test_loader, num_predictions=10):
    model.eval()
    predictions = []

    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Store prediction results
        for i in range(len(predicted)):
            predictions.append((predicted[i], labels[i]))
            if len(predictions) >= num_predictions:
                break
        if len(predictions) >= num_predictions:
            break

    # Display prediction results
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, (pred, actual) in enumerate(predictions):
        ax = axes[i // 5, i % 5]
        ax.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        ax.set_title(f'Predicted: {pred}, Actual: {actual}')
        ax.axis('off')
        if i == num_predictions - 1:
            break
    plt.tight_layout()
    plt.show()

# Call the predict function with only the first 10 prediction results
predict(best_model, test_loader, num_predictions=10)


