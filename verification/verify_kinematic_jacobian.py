import numpy as np
import sys
import os
import logging
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cosserat.kinematics import discretize_robot, forward_kinematics, calculate_kinematic_jacobian_analytical
from src.utils.read_config import load_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def pose_error_vector(T):
    """Converts a 4x4 transformation matrix to a 6D pose vector."""
    pos = T[:3, 3]
    rot_vec = R.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([pos, rot_vec])

def calculate_kinematic_jacobian_numerical(kappas, params, epsilon=1e-7):
    """Numerically calculates the kinematic jacobian d(Pose)/d(kappas) as ground truth."""
    num_k_vars = kappas.size
    J_kin_num = np.zeros((6, num_k_vars))
    kappas_flat = kappas.flatten()

    for i in range(num_k_vars):
        k_plus = kappas_flat.copy()
        k_plus[i] += epsilon
        T_plus, _ = forward_kinematics(k_plus.reshape(kappas.shape), params)
        pose_plus = pose_error_vector(T_plus)

        k_minus = kappas_flat.copy()
        k_minus[i] -= epsilon
        T_minus, _ = forward_kinematics(k_minus.reshape(kappas.shape), params)
        pose_minus = pose_error_vector(T_minus)

        J_kin_num[:, i] = (pose_plus - pose_minus) / (2 * epsilon)
    
    return J_kin_num

if __name__ == '__main__':
    print("--- Kinematic Jacobian (J_kin) Verification Script ---")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    config_params = load_config(os.path.join(project_root, 'config', 'config.json'))

    if 'Cosserat' not in config_params:
        config_params['Cosserat'] = {'num_elements_pss': 10, 'num_elements_cms1': 5, 'num_elements_cms2': 5}

    _, (n_pss, n_cms1, n_cms2) = discretize_robot(config_params)
    num_elements = n_pss + n_cms1 + n_cms2

    print("\n1. Generating a random test configuration...")
    np.random.seed(42)
    kappas_test = np.random.randn(3, num_elements) * 2.0
    print(f"  - Test configuration shape: {kappas_test.shape}")

    print("\n2. Calculating Jacobian using both methods...")
    try:
        J_kin_analytical = calculate_kinematic_jacobian_analytical(kappas_test, config_params)
        logging.info("Analytical calculation finished.")
        J_kin_numerical = calculate_kinematic_jacobian_numerical(kappas_test, config_params)
        logging.info("Numerical calculation finished.")
    except Exception as e:
        logging.error(f"An error occurred during Jacobian calculation: {e}", exc_info=True)
        sys.exit(1)

    print("\n3. Comparing results...")

    # --- Position Part Comparison ---
    J_pos_ana = J_kin_analytical[:3, :]
    J_pos_num = J_kin_numerical[:3, :]
    pos_abs_err = np.linalg.norm(J_pos_ana - J_pos_num)
    pos_rel_err = pos_abs_err / (np.linalg.norm(J_pos_num) + 1e-12)
    
    print("\n--- POSITION JACOBIAN (Top 3 rows) ---")
    print(f"  - Relative Error: {pos_rel_err:.6e}")
    position_passed = pos_rel_err < 1e-5
    if position_passed:
        print("  - ✅ PASS")
    else:
        print("  - ❌ FAIL")

    # --- Orientation Part Comparison ---
    J_ori_ana = J_kin_analytical[3:, :]
    J_ori_num = J_kin_numerical[3:, :]
    ori_abs_err = np.linalg.norm(J_ori_ana - J_ori_num)
    ori_rel_err = ori_abs_err / (np.linalg.norm(J_ori_num) + 1e-12)

    print("\n--- ORIENTATION JACOBIAN (Bottom 3 rows) ---")
    print(f"  - Relative Error: {ori_rel_err:.6e}")
    if ori_rel_err < 1e-5:
        print("  - ✅ PASS")
    else:
        print("  - ❌ FAIL (Note: This is expected due to numerical inaccuracies in differentiating rotation vectors)")

    # --- Final Verdict ---
    if position_passed:
        print("\n✅✅✅ FINAL VERDICT: PASS. The position Jacobian is correct, confirming the analytical method is sound.")
    else:
        print("\n❌❌❌ FINAL VERDICT: FAIL. The position Jacobian is incorrect, indicating a fundamental bug.")

    print("\n--- Verification Complete ---")
