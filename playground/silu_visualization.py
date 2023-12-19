import matplotlib.pyplot as plt
import numpy as np


# SiLU (Sigmoid-weighted Linear Unit) activation function
def silu(x):
    return x * (1 / (1 + np.exp(-x)))


# Generate inputs and calculate SiLU outputs
input_values = np.linspace(-10, 10, 100)
output_values = silu(input_values)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Scatter plot of SiLU outputs
ax.scatter(input_values, output_values, input_values, c=output_values, cmap="viridis")
ax.set_xlabel("Input")
ax.set_ylabel("Output")
ax.set_zlabel("Input")
ax.set_title("3D Visualization of SiLU Activation Function")

plt.show()
