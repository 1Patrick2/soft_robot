import numpy as np
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cosserat.statics import calculate_cable_jacobian, calculate_drive_mapping
from src.cosserat.kinematics import forward_kinematics, discretize_robot

def verify_cable_jacobian():
    """
    Verifies the analytical cable jacobian J_l = d(delta_l)/d(kappas) against a numerical approximation.
    """
    print("\n" + "="*80)
    print("  VERIFICATION: CABLE JACOBIAN (J_l)")
    print("="*80)

    # --- Setup ---
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    with open(config_path, 'r') as f:
        params = json.load(f)

    element_lengths, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
    num_elements = len(element_lengths)
    num_k_vars = 3 * num_elements

    # Use a non-zero kappa configuration for testing
    np.random.seed(42)
    kappas_test = np.random.randn(3, num_elements) * 0.5

    # --- Analytical Jacobian ---
    print("\nCalculating analytical jacobian...")
    J_l_analytical = calculate_cable_jacobian(kappas_test, params)
    print(f"  - Analytical J_l shape: {J_l_analytical.shape}")

    # --- Numerical Jacobian ---
    print("\nCalculating numerical jacobian...")
    J_l_numerical = np.zeros((8, num_k_vars))
    epsilon = 1e-7

    # Central difference method
    for i in range(num_k_vars):
        # Perturb one kappa variable
        kappas_flat = kappas_test.flatten()
        
        k_plus = kappas_flat.copy()
        k_plus[i] += epsilon
        T_tip_p, _ = forward_kinematics(k_plus.reshape(3, -1), params)
        T_cms1_p, _ = forward_kinematics(k_plus.reshape(3, -1), params, upto_element='cms1')
        dl_plus = calculate_drive_mapping(k_plus.reshape(3, -1), T_tip_p, T_cms1_p, params)

        k_minus = kappas_flat.copy()
        k_minus[i] -= epsilon
        T_tip_m, _ = forward_kinematics(k_minus.reshape(3, -1), params)
        T_cms1_m, _ = forward_kinematics(k_minus.reshape(3, -1), params, upto_element='cms1')
        dl_minus = calculate_drive_mapping(k_minus.reshape(3, -1), T_tip_m, T_cms1_m, params)

        # Column of Jacobian is (f(x+h) - f(x-h)) / 2h
        J_l_numerical[:, i] = (dl_plus - dl_minus) / (2 * epsilon)

    print(f"  - Numerical J_l shape: {J_l_numerical.shape}")

    # --- Comparison ---
    print("\nComparing analytical and numerical jacobians...")
    error_abs = np.linalg.norm(J_l_analytical - J_l_numerical)
    norm_numerical = np.linalg.norm(J_l_numerical)
    relative_error = error_abs / (norm_numerical + 1e-9)

    print(f"  - Absolute Error: {error_abs:.6e}")
    print(f"  - Norm of Numerical J_l: {norm_numerical:.6e}")
    print(f"  - Relative Error: {relative_error:.6e}")

    if relative_error < 1e-5:
        print("\n  ✅✅✅ PASS: Analytical cable jacobian matches numerical approximation.")
        return True
    else:
        print("\n  ❌❌❌ FAIL: Analytical cable jacobian does NOT match numerical approximation.")
        # Optional: print specific differences
        # diff = J_l_analytical - J_l_numerical
        # print(f"Max difference at index: {np.unravel_index(np.argmax(np.abs(diff)), diff.shape)}")
        return False

if __name__ == '__main__':
    verify_cable_jacobian()