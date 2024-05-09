import numpy as np

class ART1:
    def __init__(self, input_size, vigilance_parameter):
        self.input_size = input_size
        self.vigilance_parameter = vigilance_parameter
        self.activation_threshold = 0.5
        self.reset()

    def reset(self):
        self.weights = np.random.rand(self.input_size)
        self.weights /= np.linalg.norm(self.weights)

    def vigilance_test(self, input_pattern):
        norm_input = np.linalg.norm(input_pattern)
        similarity = np.dot(input_pattern, self.weights) / (norm_input + 0.001)
        return similarity >= self.vigilance_parameter

    def train(self, input_patterns, epochs=100):
        for epoch in range(epochs):
            for pattern in input_patterns:
                while True:
                    norm_pattern = np.linalg.norm(pattern)
                    output = pattern / (norm_pattern + 0.001)
                    if self.vigilance_test(output):
                        break
                    self.weights = np.maximum(self.activation_threshold * self.weights, pattern)
            self.reset()

# Example usage
if __name__ == "__main__":
    # Define parameters
    input_size = 4
    vigilance_parameter = 0.9

    # Create ART1 neural network
    art = ART1(input_size, vigilance_parameter)

    # Input patterns
    input_patterns = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    # Train the ART1 network
    art.train(input_patterns)

    # Test the trained network
    for pattern in input_patterns:
        norm_pattern = np.linalg.norm(pattern)
        output = pattern / (norm_pattern + 0.001)
        print("Input pattern:", pattern)
        print("Output pattern:", output)
        print("Vigilance test result:", art.vigilance_test(output))
