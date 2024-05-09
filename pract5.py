import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)]
        self.biases = [np.random.randn(1, self.layers[i+1]) for i in range(len(self.layers)-1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, inputs):
        self.activations = [inputs]
        for i in range(len(self.weights)):
            inputs = self.sigmoid(np.dot(inputs, self.weights[i]) + self.biases[i])
            self.activations.append(inputs)
        return inputs

    def backward_propagation(self, inputs, outputs):
        error = outputs - self.activations[-1]
        deltas = [error * self.sigmoid_derivative(self.activations[-1])]
        for i in range(len(self.activations) - 2, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            deltas.append(error * self.sigmoid_derivative(self.activations[i]))
        deltas.reverse()
        
        for i in range(len(self.weights)):
            self.weights[i] += self.activations[i].T.dot(deltas[i]) * self.learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * self.learning_rate

    def train(self, inputs, outputs, epochs):
        for epoch in range(epochs):
            self.forward_propagation(inputs)
            self.backward_propagation(inputs, outputs)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(outputs - self.activations[-1]))
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage
if __name__ == "__main__":
    # Define the neural network architecture
    neural_net = NeuralNetwork(layers=[2, 3, 1], learning_rate=0.1)

    # Input data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Output data
    y = np.array([[0], [1], [1], [0]])

    # Train the neural network
    neural_net.train(X, y, epochs=10000)

    # Test the trained neural network
    for i in range(len(X)):
        print(f"Input: {X[i]}, Predicted Output: {neural_net.forward_propagation(X[i])}")
