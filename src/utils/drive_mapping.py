# src/utils/drive_mapping.py

import numpy as np

def delta_l_to_diff4(delta_l):
    """Converts 8D motor displacements to 4D differential drives."""
    dx_short = 0.5 * (delta_l[0] - delta_l[2])
    dy_short = 0.5 * (delta_l[1] - delta_l[3])
    dx_long  = 0.5 * (delta_l[4] - delta_l[6])
    dy_long  = 0.5 * (delta_l[5] - delta_l[7])
    return np.array([dx_short, dy_short, dx_long, dy_long])

def diff4_to_delta_l(diff4):
    """Converts 4D differential drives back to 8D motor displacements."""
    dx_short, dy_short, dx_long, dy_long = diff4
    delta_l = np.zeros(8)
    # Short cable group
    delta_l[0], delta_l[2] = dx_short, -dx_short
    delta_l[1], delta_l[3] = dy_short, -dy_short
    # Long cable group
    delta_l[4], delta_l[6] = dx_long, -dx_long
    delta_l[5], delta_l[7] = dy_long, -dy_long
    return delta_l

def diff4_to_delta_l_jacobian():
    """Computes the constant 8x4 Jacobian matrix d(delta_l)/d(diff4)."""
    M = np.zeros((8, 4))
    M[0, 0] = 1.0
    M[2, 0] = -1.0
    M[1, 1] = 1.0
    M[3, 1] = -1.0
    M[4, 2] = 1.0
    M[6, 2] = -1.0
    M[5, 3] = 1.0
    M[7, 3] = -1.0
    return M
