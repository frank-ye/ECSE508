import numpy as np
import matplotlib.pyplot as plt


def replicator_dynamics(x_init, num_iter, matrix, learning_rate=1e-3, early_stopping=True):
    """ Function that returns list of replicator dynamic coordinates given an initial point

    Args:
        x_init: Starting point
        max_iter: Number of iterations
        matrix: symmetric game matrix

    Returns:
        List of x coordinates of length num_iter

    """
    x_current = x_init
    positions = []
    
    for i in range(num_iter - 1):
        positions.append(x_current)
        f = np.matmul(matrix, x_current)
        avg_payoff = np.matmul(np.transpose(x_current), f)  # Average payoff
        f_bar = np.multiply(avg_payoff, np.ones(3))  # Average payoff vector
        x_dot = np.multiply(x_current, (f - f_bar))  # Displacement vector
        x_current += learning_rate * x_dot  # Update position
        
        
    
    return positions


def main():
    # Utility matrix for 2 player symmetric game
    A = np.array([[0, 5, 4],
                 [4, 0, 5],
                 [5, 4, 0]])
    # Initial point
    x_initial = np.array([0.25, 0.5, 0.25])

    # List of parameter nabla for logit dynamics
    eta = np.array([0.01, 0.1, 1])

    x0 =  x_initial
    max_iter = 500
    coords = replicator_dynamics(x_initial, max_iter, A)
    print(coords)

if __name__ == '__main__':
    main()