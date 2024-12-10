import numpy as np

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Convert labels to one-hot encoding
def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    for idx, label in enumerate(labels):
        one_hot[idx, label] = 1
    return one_hot

# Neural Network class
class BasicNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        # Forward pass
        self.input = X
        self.hidden_layer = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output)
        return self.output_layer

    def backward(self, X, y, learning_rate):
        # Compute error
        output_error = y - self.output_layer
        output_delta = output_error * sigmoid_derivative(self.output_layer)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer)

        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_layer.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward and backward pass
            self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.output_layer))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        # Prediction
        return np.argmax(self.forward(X), axis=1)

# Read dataset and preprocess
def read_and_preprocess_data(file_name, num_samples=1000):
    import csv
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        data = [row for i, row in enumerate(reader) if i < num_samples]
    labels = [int(row[0]) for row in data]
    features = np.array([[float(val) / 255.0 for val in row[1:]] for row in data])  # Normalize pixel values
    return features, labels

# Main function to train and evaluate
def main():
    print("Reading and preprocessing data...")
    train_features, train_labels = read_and_preprocess_data('emnist-byclass-train.csv', num_samples=1000)
    test_features, test_labels = read_and_preprocess_data('emnist-byclass-test.csv', num_samples=200)

    num_classes = 62  # Adjust based on your dataset
    input_size = 784  # Number of features (28x28 pixels)
    hidden_size = 16  # Arbitrary choice for hidden neurons

    # One-hot encode labels
    train_labels_one_hot = one_hot_encode(train_labels, num_classes)

    print("Initializing Neural Network...")
    nn = BasicNeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=num_classes)

    print("Training Neural Network...")
    nn.train(train_features, train_labels_one_hot, epochs=1000, learning_rate=0.1)

    print("Evaluating Neural Network...")
    predictions = nn.predict(test_features)
    accuracy = np.mean(predictions == np.array(test_labels))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
