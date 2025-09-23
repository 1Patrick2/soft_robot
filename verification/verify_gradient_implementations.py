import numpy as np
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cosserat.statics import (
    calculate_actuation_gradient, 
    calculate_drive_mapping, 
    smooth_max_zero, 
    smooth_max_zero_derivative,
    calculate_cable_jacobian
)
from src.cosserat.kinematics import (
    forward_kinematics, 
    forward_kinematics_with_sensitivities, 
    discretize_robot
)

def verify_gradient_implementations():
    """
    Verifies that the J_l based gradient is equivalent to the J_p (endpoint force) based gradient.
    """
    print("\n" + "="*80)
    print("  VERIFICATION: GRADIENT IMPLEMENTATION CONSISTENCY")
    print("="*80)

    # --- Setup ---
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    with open(config_path, 'r') as f:
        params = json.load(f)

    element_lengths, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
    num_elements = len(element_lengths)

    np.random.seed(42)
    kappas_test = np.random.randn(3, num_elements) * 0.5
    delta_l_motor_test = np.random.rand(8) * 0.01

    # --- 1. Calculate J_l based gradient (current implementation) ---
    print("\n1. Calculating gradient based on J_l (cable length jacobian)...")
    grad_Jl = calculate_actuation_gradient(kappas_test, delta_l_motor_test, params)
    print(f"  - Done. Norm of grad_Jl: {np.linalg.norm(grad_Jl):.6e}")

    # --- 2. Calculate J_p based gradient (endpoint force model) ---
    print("\n2. Calculating gradient based on J_p (endpoint position jacobian)...")
    
    # a) Calculate Tensions consistent with the energy function's gradient
    T_tip, _ = forward_kinematics(kappas_test, params)
    T_cms1, _ = forward_kinematics(kappas_test, params, upto_element='cms1')
    delta_l_robot = calculate_drive_mapping(kappas_test, T_tip, T_cms1, params)
    stretch = delta_l_motor_test - delta_l_robot
    
    stretch_t = smooth_max_zero(stretch)
    stretch_t_deriv = smooth_max_zero_derivative(stretch)
    k_c = params['Drive_Properties']['cable_stiffness']
    f_pre = params['Drive_Properties'].get('pretension_force_N', 0.0)

    # Correct tension T = dU/d(elongation) = k*s(stretch)*s'(stretch) + f_pre
    Tensions = k_c * (stretch_t * stretch_t_deriv) + f_pre

    # b) Get Position Jacobians (J_p) for attachment points
    _, J_kin_tip = forward_kinematics_with_sensitivities(kappas_test, params)
    J_p_tip = J_kin_tip[0:3, :]
    _, J_kin_cms1 = forward_kinematics_with_sensitivities(kappas_test, params, upto_element='cms1')
    J_p_cms1 = J_kin_cms1[0:3, :]

    # c) Calculate cable direction vectors
    p_attach = np.zeros((8, 3))
    p_attach[0:4, :] = T_cms1[:3, 3]
    p_attach[4:8, :] = T_tip[:3, 3]

    anchors_short = np.array(params['Geometry']['anchor_points_base']['short'])
    anchors_long = np.array(params['Geometry']['anchor_points_base']['long'])
    anchor_world = np.vstack([anchors_short, anchors_long])
    
    dir_vec = anchor_world - p_attach
    norm_dir_vec = np.linalg.norm(dir_vec, axis=1, keepdims=True)
    dir_unit = dir_vec / (norm_dir_vec + 1e-9)

    # d) Assemble gradient from endpoint forces
    grad_Jp_flat = np.zeros(kappas_test.size)
    for i in range(8):
        # Force vector F = T * direction
        force_vec = Tensions[i] * dir_unit[i]
        if i < 4:
            grad_Jp_flat += J_p_cms1.T @ force_vec
        else:
            grad_Jp_flat += J_p_tip.T @ force_vec
    
    # Final gradient is the negative of the generalized forces from virtual work
    grad_Jp = -grad_Jp_flat.reshape(kappas_test.shape)

    print(f"  - Done. Norm of grad_Jp: {np.linalg.norm(grad_Jp):.6e}")

    # --- 3. Comparison ---
    print("\n3. Comparing the two gradient implementations...")
    error_abs = np.linalg.norm(grad_Jl - grad_Jp)
    norm_Jp = np.linalg.norm(grad_Jp)
    relative_error = error_abs / (norm_Jp + 1e-9)

    print(f"  - Absolute Error: {error_abs:.6e}")
    print(f"  - Relative Error: {relative_error:.6e}")

    if relative_error < 1e-5:
        print("\n  ✅✅✅ PASS: The two gradient implementations are consistent.")
        return True
    else:
        print("\n  ❌❌❌ FAIL: The two gradient implementations are NOT consistent.")
        return False

if __name__ == '__main__':
    verify_gradient_implementations()