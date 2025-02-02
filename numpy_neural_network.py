import numpy as np


# Applies the sigmoid activation function
def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

# Calculates the sigmoid derivative for backpropagation
def sigmoid_gradient(x):
    return x * (1 - x)

# Converts labels to one-hot encoding
def encode_one_hot(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    for idx, label in enumerate(labels):
        one_hot[idx, label] = 1
    return one_hot

# Implements a neural network with one hidden layer
class NumpyCharacterNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def run_forward_pass(self, features):
        # Forward pass
        self.input = features
        self.hidden_layer = sigmoid_activation(
            np.dot(features, self.weights_input_hidden) + self.bias_hidden
        )
        self.output_layer = sigmoid_activation(
            np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output
        )
        return self.output_layer

    def run_backward_pass(self, features, targets, learning_rate):
        # Compute error
        output_error = targets - self.output_layer
        output_delta = output_error * sigmoid_gradient(self.output_layer)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_gradient(self.hidden_layer)

        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_layer.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(features.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def fit(self, features, targets, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward and backward pass
            self.run_forward_pass(features)
            self.run_backward_pass(features, targets, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(targets - self.output_layer))
                print(f"Epoch {epoch}, Loss: {loss}")

    def classify(self, features):
        # Prediction
        return np.argmax(self.run_forward_pass(features), axis=1)

# Reads and preprocesses a dataset
def load_numpy_samples(file_name, num_samples=1000):
    import csv
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        data = [row for i, row in enumerate(reader) if i < num_samples]
    labels = [int(row[0]) for row in data]
    features = np.array([[float(val) / 255.0 for val in row[1:]] for row in data])  # Normalize pixel values
    return features, labels

# Trains and evaluates the neural network
def run_numpy_network():
    print("Reading and preprocessing data...")
    train_features, train_labels = load_numpy_samples(
        'emnist-byclass-train.csv', num_samples=1000
    )
    test_features, test_labels = load_numpy_samples(
        'emnist-byclass-test.csv', num_samples=200
    )

    num_classes = 62  # 10 digits, 26 uppercase letters, and 26 lowercase letters
    input_size = 784  # Number of features (28×28 pixels)
    hidden_size = 16  # Arbitrary choice for hidden neurons

    # One-hot encode labels
    train_labels_one_hot = encode_one_hot(train_labels, num_classes)

    print("Initializing Neural Network...")
    network = NumpyCharacterNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_classes
    )

    print("Training Neural Network...")
    network.fit(train_features, train_labels_one_hot, epochs=1000, learning_rate=0.1)

    print("Evaluating Neural Network...")
    predictions = network.classify(test_features)
    accuracy = np.mean(predictions == np.array(test_labels))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    run_numpy_network()
