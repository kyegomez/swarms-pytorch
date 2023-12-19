import torch
import pytest
from swarms_torch.swarmalators.swarmalator_base import (
    pairwise_distances,
    function_for_x,
    function_for_sigma,
    simulate_swarmalators,
)

# Define global constants for testing
N = 10
J = 1.0
alpha = 0.1
beta = 0.2
gamma = 0.3
epsilon_a = 0.01
epsilon_r = 0.02
R = 0.5
D = 3
T = 100
dt = 0.1

# === Test pairwise_distances ===


def test_pairwise_distances_shape():
    x = torch.randn(N, D)
    distances = pairwise_distances(x)
    assert distances.shape == (N, N)


def test_pairwise_distances_identity():
    x = torch.randn(N, D)
    distances = pairwise_distances(x)
    for i in range(N):
        assert distances[i, i] == pytest.approx(0.0, abs=1e-6)


def test_pairwise_distances_symmetry():
    x = torch.randn(N, D)
    distances = pairwise_distances(x)
    for i in range(N):
        for j in range(i + 1, N):
            assert distances[i, j] == pytest.approx(distances[j, i], abs=1e-6)


# === Test function_for_x ===


def test_function_for_x_shape():
    xi = torch.randn(N, D)
    sigma_i = torch.randn(N, D)
    dx = function_for_x(
        xi, sigma_i, N, J, alpha, beta, gamma, epsilon_a, epsilon_r, R, D
    )
    assert dx.shape == (N, D)


def test_function_for_x_output_range():
    xi = torch.randn(N, D)
    sigma_i = torch.randn(N, D)
    dx = function_for_x(
        xi, sigma_i, N, J, alpha, beta, gamma, epsilon_a, epsilon_r, R, D
    )
    assert (dx >= -1.0).all() and (dx <= 1.0).all()


def test_function_for_x_zero_at_equilibrium():
    xi = torch.zeros(N, D)
    sigma_i = torch.zeros(N, D)
    dx = function_for_x(
        xi, sigma_i, N, J, alpha, beta, gamma, epsilon_a, epsilon_r, R, D
    )
    assert (dx == 0.0).all()


# === Test function_for_sigma ===


def test_function_for_sigma_shape():
    xi = torch.randn(N, D)
    sigma_i = torch.randn(N, D)
    d_sigma = function_for_sigma(
        xi, sigma_i, N, J, alpha, beta, gamma, epsilon_a, epsilon_r, R, D
    )
    assert d_sigma.shape == (N, D)


def test_function_for_sigma_output_range():
    xi = torch.randn(N, D)
    sigma_i = torch.randn(N, D)
    d_sigma = function_for_sigma(
        xi, sigma_i, N, J, alpha, beta, gamma, epsilon_a, epsilon_r, R, D
    )
    assert (d_sigma >= -1.0).all() and (d_sigma <= 1.0).all()


def test_function_for_sigma_zero_at_equilibrium():
    xi = torch.zeros(N, D)
    sigma_i = torch.zeros(N, D)
    d_sigma = function_for_sigma(
        xi, sigma_i, N, J, alpha, beta, gamma, epsilon_a, epsilon_r, R, D
    )
    assert (d_sigma == 0.0).all()


# === Test simulate_swarmalators ===


def test_simulate_swarmalators_output_shape():
    results_xi, results_sigma_i = simulate_swarmalators(
        N, J, alpha, beta, gamma, epsilon_a, epsilon_r, R, D, T=T, dt=dt
    )
    assert len(results_xi) == T
    assert len(results_sigma_i) == T
    assert results_xi[0].shape == (N, D)
    assert results_sigma_i[0].shape == (N, D)


def test_simulate_swarmalators_convergence():
    results_xi, results_sigma_i = simulate_swarmalators(
        N, J, alpha, beta, gamma, epsilon_a, epsilon_r, R, D, T=T, dt=dt
    )
    for i in range(1, T):
        assert torch.allclose(results_xi[i], results_xi[i - 1], atol=1e-6)
        assert torch.allclose(results_sigma_i[i], results_sigma_i[i - 1], atol=1e-6)


def test_simulate_swarmalators_non_zero_initial_condition():
    xi = torch.randn(N, D)
    sigma_i = torch.randn(N, D)
    results_xi, results_sigma_i = simulate_swarmalators(
        N,
        J,
        alpha,
        beta,
        gamma,
        epsilon_a,
        epsilon_r,
        R,
        D,
        T=T,
        dt=dt,
        xi=xi,
        sigma_i=sigma_i,
    )
    assert not torch.allclose(results_xi[0], xi, atol=1e-6)
    assert not torch.allclose(results_sigma_i[0], sigma_i, atol=1e-6)


# Add more tests as needed...
