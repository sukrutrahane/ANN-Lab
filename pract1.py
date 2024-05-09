import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

# Generate x values
x = np.linspace(-5, 5)


# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, tanh(x), label='Tanh')
plt.title('Hyperbolic Tangent (tanh) Activation Function')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, relu(x), label='ReLU')
plt.title('Rectified Linear Unit (ReLU) Activation Function')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, linear(x), label='Linear')
plt.title('Linear Activation Function')
plt.legend()

plt.tight_layout()
plt.show()
