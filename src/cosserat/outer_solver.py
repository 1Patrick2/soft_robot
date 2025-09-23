import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import sys
import os
import logging
import multiprocessing

# --- Cosserat Model Dependencies ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cosserat.solver import solve_static_equilibrium
from src.cosserat.kinematics import forward_kinematics, discretize_robot, calculate_kinematic_jacobian_analytical
from src.cosserat.statics import calculate_hessian_approx, calculate_coupling_matrix_C
from src.utils.read_config import load_config

# =====================================================================
# === Cosserat Outer-Loop IK Solver (V-Final with Analytical J_task)
# =====================================================================

# --- Start of multiprocessing worker setup ---

# Global variables for the pool initializer to avoid data serialization issues
g_kappas_guess = None
g_params = None
g_target_pose = None

def init_worker(kappas_guess, params, target_pose):
    """Initializer for the multiprocessing pool."""
    global g_kappas_guess, g_params, g_target_pose
    g_kappas_guess = kappas_guess
    g_params = params
    g_target_pose = target_pose

def pso_worker(dl_i):
    """
    The actual work done by each process for a single particle.
    It computes the cost for a given delta_l vector.
    """
    # Access global variables initialized for the process
    solve_result = solve_static_equilibrium(g_kappas_guess, dl_i, g_params)
    kappas_eq = solve_result["kappas_solution"]
    if kappas_eq is None:
        return 1e6  # High cost for failed solves
    T_actual, _ = forward_kinematics(kappas_eq, g_params)
    error_vec = pose_error(T_actual, g_target_pose)
    # The cost function balances position error (in mm) and orientation error (in rad)
    cost = np.linalg.norm(error_vec[:3]) * 1000 + np.linalg.norm(error_vec[3:])
    return cost

# --- End of multiprocessing worker setup ---


def pose_error(T_current, T_target):
    """Calculates the 6D pose error vector (3D position, 3D orientation)."""
    pos_error = T_current[:3, 3] - T_target[:3, 3]
    R_current = T_current[:3, :3]
    R_target = T_target[:3, :3]
    R_err = R_target @ R_current.T
    rot_vec = R.from_matrix(R_err).as_rotvec()
    return np.concatenate([pos_error, rot_vec])

def calculate_kinematic_jacobian_numerical(kappas, params, epsilon=1e-7):
    """Numerically calculates the kinematic jacobian d(Pose)/d(kappas)."""
    num_vars = kappas.size
    J_kin = np.zeros((6, num_vars))
    
    T_base, _ = forward_kinematics(kappas, params)

    for i in range(num_vars):
        kappas_flat = kappas.flatten()
        kappas_flat_plus = kappas_flat.copy()
        kappas_flat_plus[i] += epsilon
        T_plus, _ = forward_kinematics(kappas_flat_plus.reshape(kappas.shape), params)
        
        kappas_flat_minus = kappas_flat.copy()
        kappas_flat_minus[i] -= epsilon
        T_minus, _ = forward_kinematics(kappas_flat_minus.reshape(kappas.shape), params)

        pose_plus = np.concatenate([T_plus[:3, 3], R.from_matrix(T_plus[:3, :3]).as_rotvec()])
        pose_minus = np.concatenate([T_minus[:3, 3], R.from_matrix(T_minus[:3, :3]).as_rotvec()])

        # This is not a perfect derivative for orientation, but a good approximation
        J_kin[:, i] = (pose_plus - pose_minus) / (2 * epsilon)

    return J_kin

def calculate_task_jacobian(kappas_eq, delta_l_motor, params, epsilon=1e-7):
    """
    Numerically calculates the task jacobian d(Pose)/d(delta_l) as the ground truth.
    This is a reliable but slow method that bypasses the implicit function theorem.
    """
    num_dl_vars = delta_l_motor.size
    J_task_num = np.zeros((6, num_dl_vars))

    for i in range(num_dl_vars):
        dl_plus = delta_l_motor.copy()
        dl_plus[i] += epsilon
        # Use current equilibrium as a warm start for the perturbed solve
        res_plus = solve_static_equilibrium(kappas_eq, dl_plus, params)
        if res_plus['kappas_solution'] is None:
            logging.warning(f"[J_task] Inner solver failed for numerical jacobian (+) at index {i}. Returning zero column.")
            J_task_num[:, i] = 0
            continue
        T_plus, _ = forward_kinematics(res_plus['kappas_solution'], params)
        
        rvp = R.from_matrix(T_plus[:3,:3]).as_rotvec()
        pose_plus = np.concatenate([T_plus[:3,3], rvp])

        dl_minus = delta_l_motor.copy()
        dl_minus[i] -= epsilon
        res_minus = solve_static_equilibrium(kappas_eq, dl_minus, params)
        if res_minus['kappas_solution'] is None:
            logging.warning(f"[J_task] Inner solver failed for numerical jacobian (-) at index {i}. Returning zero column.")
            J_task_num[:, i] = 0
            continue
        T_minus, _ = forward_kinematics(res_minus['kappas_solution'], params)
        
        rvm = R.from_matrix(T_minus[:3,:3]).as_rotvec()
        pose_minus = np.concatenate([T_minus[:3,3], rvm])

        J_task_num[:, i] = (pose_plus - pose_minus) / (2 * epsilon)
    
    return J_task_num

def residual_and_jacobian(delta_l_motor, kappas_guess, target_pose, params):
    """ 
    A single function for least_squares that returns both residual and jacobian.
    """
    # 1. Find equilibrium for the current delta_l_motor
    solve_result = solve_static_equilibrium(kappas_guess, delta_l_motor, params)
    kappas_eq = solve_result["kappas_solution"]

    if kappas_eq is None:
        error = np.full(6, 1e6)
        # Return a zero jacobian of the correct shape if solver fails
        num_vars = len(delta_l_motor)
        jacobian = np.zeros((6, num_vars))
        return error, jacobian

    # 2. Calculate pose and error (residual)
    T_actual, _ = forward_kinematics(kappas_eq, params)
    error = pose_error(T_actual, target_pose)

    # 3. Calculate analytical task jacobian
    jacobian = calculate_task_jacobian(kappas_eq, delta_l_motor, params)

    # 4. Scale the error and jacobian to balance position and orientation
    # Treat position error in 'mm' to make it comparable to orientation error in 'rad'
    pose_scale = np.array([1000, 1000, 1000, 1, 1, 1])
    error_scaled = error * pose_scale
    jacobian_scaled = jacobian * pose_scale[:, np.newaxis]

    return error_scaled, jacobian_scaled

import pyswarms as ps

def solve_ik(target_pose, params, kappas_guess=None, delta_l_guess=None, use_pso=False, n_particles=10, n_iters=15):
    """Solves the IK problem, with an optional PSO warm-start."""
    if 'Cosserat' not in params:
        params['Cosserat'] = {'num_elements_pss': 10, 'num_elements_cms1': 5, 'num_elements_cms2': 5}

    if kappas_guess is None:
        _, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
        kappas_guess = np.zeros((3, n_pss + n_cms1 + n_cms2))
    
    bounds_config = params.get('Bounds', {}).get('delta_l_bounds', [-0.05, 0.05])
    if isinstance(bounds_config[0], (list, tuple)):
        bounds = tuple(zip(*bounds_config))
    else:
        min_b, max_b = bounds_config[0], bounds_config[1]
        bounds = ([min_b] * 8, [max_b] * 8)
    
    pso_best_pos = None
    if use_pso:
        logging.info("[PSO Warm-Start] Starting global search for a better initial guess...")

        def pso_objective_func(delta_l_particles):
            # This function is now a wrapper for the parallel pool execution
            # It uses an initializer to pass large, read-only data to worker processes
            # efficiently and safely, especially on Windows.
            # Note: We must use 'if __name__ == '__main__':' in the main script
            # to prevent issues with multiprocessing on Windows.
            with multiprocessing.Pool(initializer=init_worker, initargs=(kappas_guess, params, target_pose)) as pool:
                costs = pool.map(pso_worker, delta_l_particles)
            return np.array(costs)

        pso_options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=8, options=pso_options, bounds=bounds)
        cost, pos = optimizer.optimize(pso_objective_func, iters=n_iters, verbose=True)
        logging.info(f"[PSO Warm-Start] Found best delta_l with cost {cost:.4f}. Using it as initial guess.")
        delta_l_guess = pos
        pso_best_pos = pos
    elif delta_l_guess is None:
        delta_l_guess = np.zeros(8)

    memo = {}
    def get_res_and_jac(delta_l_motor_tuple):
        if delta_l_motor_tuple not in memo:
            memo[delta_l_motor_tuple] = residual_and_jacobian(np.array(delta_l_motor_tuple), kappas_guess, target_pose, params)
        return memo[delta_l_motor_tuple]

    def fun_for_lsq(delta_l_motor):
        return get_res_and_jac(tuple(delta_l_motor))[0]

    def jac_for_lsq(delta_l_motor):
        return get_res_and_jac(tuple(delta_l_motor))[1]

    logging.info(f"[TRF Refinement] Starting local optimization from guess: {np.round(delta_l_guess, 4)}")
    result = least_squares(
        fun_for_lsq,
        delta_l_guess,
        jac=jac_for_lsq,
        bounds=bounds,
        method='trf', 
        max_nfev=50,
        ftol=1e-8, xtol=1e-8, gtol=1e-8,
        verbose=2
    )

    return_data = {
        "success": result.success,
        "final_delta_l": result.x,
        "pso_best_delta_l": pso_best_pos,
        "raw_result": result
    }

    if not result.success:
        logging.error(f"[IK Solver] Local refinement failed to converge. Message: {result.message}")

    return return_data

if __name__ == '__main__':
    # This check is crucial for multiprocessing to work correctly on Windows
    multiprocessing.freeze_support() 
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("--- Cosserat Outer-Loop IK Solver Self-Test (Analytical Jacobian) ---")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_params = load_config(os.path.join(project_root, 'config', 'config.json'))
    
    if 'Cosserat' not in config_params:
        config_params['Cosserat'] = {'num_elements_pss': 10, 'num_elements_cms1': 5, 'num_elements_cms2': 5}

    print("\n1. Generating a reachable target pose...")
    _, (n_pss, n_cms1, n_cms2) = discretize_robot(config_params)
    kappas_target = np.zeros((3, n_pss + n_cms1 + n_cms2))
    kappas_target[1, n_pss:] = 1.0
    target_pose, _ = forward_kinematics(kappas_target, config_params)
    print(f"  - Target position: {target_pose[:3, 3]}")

    print("\n2. Running IK solver to find the target...")
    ik_result = solve_ik(target_pose, config_params, use_pso=True, n_particles=10, n_iters=15)

    def verify_dl(name, delta_l_to_verify, base_kappas, target_pose, params):
        print(f"\n--- Verifying {name} Result ---")
        if delta_l_to_verify is None:
            print("  - N/A (PSO was not run or did not provide a position).")
            return
            
        print(f"  - Found delta_l: {np.round(delta_l_to_verify, 4)}")
        kappas_res = solve_static_equilibrium(base_kappas, delta_l_to_verify, params)
        kappas_sol = kappas_res['kappas_solution']
        
        if kappas_sol is not None:
            pose, _ = forward_kinematics(kappas_sol, params)
            error_mm = np.linalg.norm(pose[:3, 3] - target_pose[:3, 3]) * 1000
            print(f"  - Position Error: {error_mm:.4f} mm")
        else:
            print("  - ❌ Verification failed: Inner-loop solver could not converge.")

    # --- Stage 1: PSO Result Verification ---
    if "pso_best_delta_l" in ik_result:
        verify_dl("PSO Warm-Start", ik_result["pso_best_delta_l"], np.zeros_like(kappas_target), target_pose, config_params)

    # --- Stage 2: Final TRF Result Verification ---
    if ik_result["success"]:
        print("\n✅✅✅ TRF Refinement converged successfully!")
        verify_dl("Final (TRF)", ik_result["final_delta_l"], np.zeros_like(kappas_target), target_pose, config_params)
    else:
        print("\n❌❌❌ TRF Refinement FAILED to converge.")

    print("\n--- Self-Test Complete ---")
