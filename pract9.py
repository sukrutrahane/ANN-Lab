import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_input_hidden = np.zeros((1, hidden_size))
        
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, inputs):
        # Input to hidden layer
        self.hidden_output = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden)
        
        # Hidden to output layer
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output)
        
        return self.output

    def backward_propagation(self, inputs, outputs):
        # Compute gradients for the output layer
        error_output = outputs - self.output
        delta_output = error_output * self.sigmoid_derivative(self.output)
        
        # Adjust weights and biases for the output layer
        self.weights_hidden_output += np.dot(self.hidden_output.T, delta_output) * self.learning_rate
        self.bias_hidden_output += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
        
        # Compute gradients for the hidden layer
        error_hidden = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)
        
        # Adjust weights and biases for the hidden layer
        self.weights_input_hidden += np.dot(inputs.T, delta_hidden) * self.learning_rate
        self.bias_input_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate

# Example usage
if __name__ == "__main__":
    # Define neural network parameters
    input_size = 2
    hidden_size = 3
    output_size = 1
    learning_rate = 0.1

    # Create and initialize the neural network
    neural_net = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

    # Input data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Output data
    y = np.array([[0], [1], [1], [0]])

    # Train the neural network
    epochs = 10000
    for epoch in range(epochs):
        neural_net.forward_propagation(X)
        neural_net.backward_propagation(X, y)
        if epoch % 1000 == 0:
            loss = np.mean(np.square(y - neural_net.output))
            print(f"Epoch {epoch}, Loss: {loss}")

    # Test the trained neural network
    predictions = neural_net.forward_propagation(X)
    print("Predictions after training:")
    print(predictions)
