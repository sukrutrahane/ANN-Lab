import numpy as np

# Training data for ASCII representation of 0 to 9
training_data = {
    '0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '1': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '2': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    '3': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    '4': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    '5': [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    '6': [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    '7': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    '8': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    '9': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
}

# Target labels for even and odd
target_labels = {
    '0': 0,  # Even
    '1': 1,  # Odd
    '2': 0,  # Even
    '3': 1,  # Odd
    '4': 0,  # Even
    '5': 1,  # Odd
    '6': 0,  # Even
    '7': 1,  # Odd
    '8': 0,  # Even
    '9': 1,  # Odd
}

# Convert data and labels to numpy arrays
X = np.array(list(training_data.values()))
y = np.array(list(target_labels.values()))

# Simple Perceptron
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0

    def predict(self, x):
        return 1 if np.dot(self.weights, x) + self.bias > 0 else 0

    def train(self, X, y, epochs=100, learning_rate=0.1):
        for epoch in range(epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                self.weights += learning_rate * (y[i] - prediction) * X[i]
                self.bias += learning_rate * (y[i] - prediction)

# Train the Perceptron
perceptron = Perceptron(input_size=len(X[0]))
perceptron.train(X, y)

# Take user input for ASCII representation of a number
user_input = input("Enter the ASCII representation of a number (0 to 9): ")

# Convert user input to the input format expected by the perceptron
user_input_formatted = [int(char) for char in user_input]

# Use the trained perceptron to predict whether the number is even or odd
result = perceptron.predict(user_input_formatted)

# Print the result
if result == 0:
    print("The number is even.")
else:
    print("The number is odd.")
