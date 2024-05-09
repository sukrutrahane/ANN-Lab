import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            # Convert pattern to a column vector
            pattern = pattern.reshape(-1, 1)
            # Update weights
            self.weights += np.dot(pattern, pattern.T)
            # Zero out diagonal elements (self-connections)
            np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, max_iter=100):
        pattern = pattern.reshape(-1, 1)
        for _ in range(max_iter):
            # Update neurons' state asynchronously
            new_pattern = np.sign(np.dot(self.weights, pattern))
            if np.array_equal(new_pattern, pattern):
                return new_pattern.flatten()
            pattern = new_pattern
        print("Max iterations reached. Network did not converge.")
        return pattern.flatten()

# Example usage
if __name__ == "__main__":
    # Define patterns to be stored
    patterns = np.array([[1, -1, 1, -1],
                         [1, 1, -1, -1],
                         [-1, 1, -1, 1],
                         [-1, -1, 1, 1]])

    # Create a Hopfield network
    hopfield_net = HopfieldNetwork(num_neurons=len(patterns[0]))

    # Train the network with the patterns
    hopfield_net.train(patterns)

    # Test the network by recalling the stored patterns
    for i, pattern in enumerate(patterns):
        print(f"Original pattern {i+1}: {pattern}")
        recalled_pattern = hopfield_net.recall(pattern)
        print(f"Recalled pattern {i+1}: {recalled_pattern}")
        print()
