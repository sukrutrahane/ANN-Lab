import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights = []
        self.biases = []

        # Initialize weights and biases for hidden layers
        for _ in range(hidden_layers):
            self.weights.append(np.random.randn(input_size, hidden_size))
            self.biases.append(np.zeros((1, hidden_size)))
            input_size = hidden_size

        # Initialize weights and biases for output layer
        self.weights.append(np.random.randn(input_size, output_size))
        self.biases.append(np.zeros((1, output_size)))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward_propagation(self, inputs):
        self.activations = []
        layer_output = inputs
        for i in range(len(self.weights)):
            layer_output = np.dot(layer_output, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                layer_output = self.relu(layer_output)
            else:
                layer_output = self.softmax(layer_output)
            self.activations.append(layer_output)
        return layer_output

    def backward_propagation(self, inputs, outputs):
        error = self.activations[-1] - outputs
        for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1:
                delta = error
            else:
                delta = np.dot(delta, self.weights[i+1].T) * (self.activations[i] > 0)
            self.weights[i] -= np.dot(self.activations[i].T, delta) * self.learning_rate
            self.biases[i] -= np.sum(delta, axis=0, keepdims=True) * self.learning_rate
            error = delta

    def train(self, inputs, outputs, epochs):
        for epoch in range(epochs):
            self.forward_propagation(inputs)
            self.backward_propagation(inputs, outputs)
            if epoch % 100 == 0:
                loss = np.mean(-np.log(self.activations[-1][np.arange(len(outputs)), np.argmax(outputs, axis=1)]))
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage
if __name__ == "__main__":
    # Sample data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # Labels (two classes)

    # Define the neural network architecture
    neural_net = NeuralNetwork(input_size=2, hidden_layers=1, hidden_size=100, output_size=2, learning_rate=0.1)

    # Train the neural network
    neural_net.train(X, y, epochs=1000)

    # Test the trained neural network
    predictions = neural_net.forward_propagation(X)
    print("Predictions:")
    print(predictions)
