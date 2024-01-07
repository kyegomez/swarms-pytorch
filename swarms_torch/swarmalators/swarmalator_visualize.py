import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from swarms_torch.swarmalators.swarmalator_base import simulate_swarmalators

# Example usage:
N = 100
J, alpha, beta, gamma, epsilon_a, epsilon_r, R = [0.1] * 7
D = 3  # Ensure D is an integer
xi, sigma_i = simulate_swarmalators(
    N, J, alpha, beta, gamma, epsilon_a, epsilon_r, R, D
)
print(xi[-1], sigma_i[-1])


def visualize_swarmalators(results_xi):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    # Initialize the scatter plot
    scatter = ax.scatter([], [], [])

    def init():
        scatter._offsets3d = ([], [], [])
        return (scatter,)

    def update(num):
        ax.view_init(30, 0.3 * num)
        x_data, y_data, z_data = results_xi[num].t()
        scatter._offsets3d = (x_data, y_data, z_data)
        return (scatter,)

    FuncAnimation(
        fig, update, frames=len(results_xi), init_func=init, blit=False
    )

    plt.show()


# # Call the visualization function
# visualize_swarmalators(xi)
