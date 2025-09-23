import numpy as np
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cosserat.kinematics import (
    forward_kinematics, 
    forward_kinematics_with_sensitivities, 
    discretize_robot
)

def verify_position_jacobian():
    """
    Verifies the analytical position jacobian J_p from `forward_kinematics_with_sensitivities`
    against a numerical approximation based on `forward_kinematics`.
    """
    print("\n" + "="*80)
    print("  VERIFICATION: POSITION JACOBIAN (J_p)")
    print("="*80)

    # --- Setup ---
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    with open(config_path, 'r') as f:
        params = json.load(f)

    element_lengths, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
    num_elements = len(element_lengths)
    num_k_vars = 3 * num_elements

    np.random.seed(42)
    kappas_test = np.random.randn(3, num_elements) * 0.5

    # --- Analytical Jacobian ---
    print("\nCalculating analytical position jacobian (J_p)...")
    _, J_kin_analytical = forward_kinematics_with_sensitivities(kappas_test, params)
    J_p_analytical = J_kin_analytical[0:3, :]
    print(f"  - Analytical J_p shape: {J_p_analytical.shape}")

    # --- Numerical Jacobian ---
    print("\nCalculating numerical position jacobian...")
    J_p_numerical = np.zeros((3, num_k_vars))
    epsilon = 1e-7

    for i in range(num_k_vars):
        k_plus = kappas_test.flatten().copy()
        k_plus[i] += epsilon
        T_plus, _ = forward_kinematics(k_plus.reshape(3, -1), params)
        p_plus = T_plus[:3, 3]

        k_minus = kappas_test.flatten().copy()
        k_minus[i] -= epsilon
        T_minus, _ = forward_kinematics(k_minus.reshape(3, -1), params)
        p_minus = T_minus[:3, 3]

        J_p_numerical[:, i] = (p_plus - p_minus) / (2 * epsilon)

    print(f"  - Numerical J_p shape: {J_p_numerical.shape}")

    # --- Comparison ---
    print("\nComparing analytical and numerical jacobians...")
    error_abs = np.linalg.norm(J_p_analytical - J_p_numerical)
    norm_numerical = np.linalg.norm(J_p_numerical)
    relative_error = error_abs / (norm_numerical + 1e-9)

    print(f"  - Absolute Error: {error_abs:.6e}")
    print(f"  - Norm of Numerical J_p: {norm_numerical:.6e}")
    print(f"  - Relative Error: {relative_error:.6e}")

    if relative_error < 1e-5:
        print("\n  ✅✅✅ PASS: Analytical position jacobian matches numerical approximation.")
        return True
    else:
        print("\n  ❌❌❌ FAIL: Analytical position jacobian does NOT match numerical approximation.")
        return False

if __name__ == '__main__':
    verify_position_jacobian()
