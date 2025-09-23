import numpy as np
import sys
import os
import logging
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cosserat.outer_solver import calculate_task_jacobian
from src.cosserat.solver import solve_static_equilibrium
from src.cosserat.kinematics import discretize_robot, forward_kinematics
from src.utils.read_config import load_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def pose_error_vector(T):
    """Converts a 4x4 transformation matrix to a 6D pose vector."""
    pos = T[:3, 3]
    rot_vec = R.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([pos, rot_vec])

def calculate_task_jacobian_numerical(kappas_guess, delta_l_motor, params, epsilon=1e-7):
    """Numerically calculates the task jacobian d(Pose)/d(delta_l) as ground truth."""
    num_dl_vars = delta_l_motor.size
    J_task_num = np.zeros((6, num_dl_vars))

    for i in range(num_dl_vars):
        logging.info(f"[Numerical] Calculating column {i+1}/{num_dl_vars}...")
        dl_plus = delta_l_motor.copy()
        dl_plus[i] += epsilon
        res_plus = solve_static_equilibrium(kappas_guess, dl_plus, params)
        if res_plus['kappas_solution'] is None:
            raise RuntimeError(f"Numerical jacobian failed: inner solver did not converge for dl_plus at index {i}")
        T_plus, _ = forward_kinematics(res_plus['kappas_solution'], params)
        pose_plus = pose_error_vector(T_plus)

        dl_minus = delta_l_motor.copy()
        dl_minus[i] -= epsilon
        res_minus = solve_static_equilibrium(kappas_guess, dl_minus, params)
        if res_minus['kappas_solution'] is None:
            raise RuntimeError(f"Numerical jacobian failed: inner solver did not converge for dl_minus at index {i}")
        T_minus, _ = forward_kinematics(res_minus['kappas_solution'], params)
        pose_minus = pose_error_vector(T_minus)

        J_task_num[:, i] = (pose_plus - pose_minus) / (2 * epsilon)
    
    return J_task_num

if __name__ == '__main__':
    print("--- Task Jacobian (J_task) End-to-End Verification Script ---")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_params = load_config(os.path.join(project_root, 'config', 'config.json'))

    if 'Cosserat' not in config_params:
        config_params['Cosserat'] = {'num_elements_pss': 10, 'num_elements_cms1': 5, 'num_elements_cms2': 5}

    _, (n_pss, n_cms1, n_cms2) = discretize_robot(config_params)
    num_elements = n_pss + n_cms1 + n_cms2

    print("\n1. Generating a non-trivial test state...")
    np.random.seed(42)
    # Start from a non-zero motor input to avoid singularity
    delta_l_test = np.random.randn(8) * 0.01 
    kappas_guess = np.zeros((3, num_elements))

    print("  - Finding initial equilibrium point...")
    res = solve_static_equilibrium(kappas_guess, delta_l_test, config_params)
    if res['kappas_solution'] is None:
        logging.error("Failed to find an initial equilibrium point for the test. Aborting.")
        sys.exit(1)
    kappas_eq_test = res['kappas_solution']
    print("  - Equilibrium point found.")

    print("\n2. Calculating Jacobian using both methods at this point...")
    try:
        J_task_analytical = calculate_task_jacobian(kappas_eq_test, delta_l_test, config_params)
        logging.info("Analytical calculation finished.")
        J_task_numerical = calculate_task_jacobian_numerical(kappas_eq_test, delta_l_test, config_params)
        logging.info("Numerical calculation finished.")
    except Exception as e:
        logging.error(f"An error occurred during Jacobian calculation: {e}", exc_info=True)
        sys.exit(1)

    print("\n3. Comparing results...")
    
    absolute_error = np.linalg.norm(J_task_analytical - J_task_numerical)
    norm_numerical = np.linalg.norm(J_task_numerical)
    relative_error = absolute_error / (norm_numerical + 1e-12)

    print(f"  - L2 Norm of Analytical Jacobian: {np.linalg.norm(J_task_analytical):.6f}")
    print(f"  - L2 Norm of Numerical Jacobian (Ground Truth): {norm_numerical:.6f}")
    print(f"  - Absolute Error (L2 Norm of difference): {absolute_error:.6e}")
    print(f"  - Relative Error: {relative_error:.6e}")

    if relative_error < 1e-5:
        print("\n✅✅✅ PASS: The analytical task jacobian matches the numerical ground truth.")
    else:
        print("\n❌❌❌ FAIL: Significant deviation found between analytical and numerical results.")

    print("\n--- Verification Complete ---")
