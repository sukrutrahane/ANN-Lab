import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Define the architecture
        self.input_size = 2
        self.hidden_size = 2
        self.output_size = 1
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))
        
        # Learning rate
        self.learning_rate = 0.1

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

# XOR input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and initialize the neural network
neural_net = NeuralNetwork()

# Perform backpropagation for one iteration
neural_net.forward_propagation(X)
neural_net.backward_propagation(X, y)

# Updated weights and biases after backpropagation
print("Updated weights and biases after one iteration of backpropagation:")
print("Weights (Input to Hidden):")
print(neural_net.weights_input_hidden)
print("Biases (Input to Hidden):")
print(neural_net.bias_input_hidden)
print("Weights (Hidden to Output):")
print(neural_net.weights_hidden_output)
print("Biases (Hidden to Output):")
print(neural_net.bias_hidden_output)
