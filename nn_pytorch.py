import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import matplotlib.pyplot as plt


# Function to load and preprocess the data
def load_data(file_name, num_samples=None):
    features = []
    labels = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if num_samples is not None and i >= num_samples:
                break
            label = int(row[0])
            pixels = [float(pix) / 255.0 for pix in row[1:]]  # Normalize pixel values to [0,1]
            features.append(pixels)
            labels.append(label)
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    return features, labels

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        # Input layer to hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function
        self.relu = nn.ReLU()
        # Hidden layer to output layer
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # Note: CrossEntropyLoss applies softmax internally

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out  # No softmax here because CrossEntropyLoss does it

def main():
    # Load the training and test data
    print("Loading data...")
    train_features, train_labels = load_data('emnist-byclass-train.csv', num_samples=10000)
    test_features, test_labels = load_data('emnist-byclass-test.csv', num_samples=2000)

    # Convert data to PyTorch tensors
    train_features = torch.from_numpy(train_features)
    train_labels = torch.from_numpy(train_labels)
    test_features = torch.from_numpy(test_features)
    test_labels = torch.from_numpy(test_labels)

    # Lists to store loss and epochs for plotting
    epochs_list = []
    average_losses = []

    # Hyperparameters
    input_size = 784         # 28x28 images
    hidden_size = 128        # Number of neurons in hidden layer
    num_classes = 62         # Number of output classes
    num_epochs = 200           # Number of training epochs
    batch_size = 64          # Batch size
    learning_rate = 0.001    # Learning rate

    # Create datasets and data loaders
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # Initialize the neural network model
    model = SimpleNN(input_size, hidden_size, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Combines Softmax and NLLLoss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / total_step 

        # Store data for plotting
        epochs_list.append(epoch + 1)
        average_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    # Evaluate the model
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # Get class with highest probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'\nTest Accuracy: {100 * correct / total:.2f}%')

     # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, average_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
