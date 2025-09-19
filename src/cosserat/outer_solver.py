import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import sys
import os
import logging

# --- Cosserat Model Dependencies ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cosserat.solver import solve_static_equilibrium
from src.cosserat.kinematics import forward_kinematics, discretize_robot
from src.cosserat.statics import calculate_hessian_approx, calculate_coupling_matrix_C
from src.utils.read_config import load_config

# =====================================================================
# === Cosserat Outer-Loop IK Solver (V-Final with Analytical J_task)
# =====================================================================

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

def calculate_task_jacobian(kappas, delta_l_motor, params):
    """Calculates the analytical task jacobian J_task = d(Pose)/d(delta_l)."""
    # 1. Kinematic Jacobian (dPose/dkappas)
    J_kin = calculate_kinematic_jacobian_numerical(kappas, params)

    # 2. Hessian Approximation H = d(gradU)/d(kappas)
    H = calculate_hessian_approx(kappas, delta_l_motor, params)

    # 3. Coupling Matrix C = d(gradU)/d(delta_l)
    C = calculate_coupling_matrix_C(kappas, delta_l_motor, params)

    # 4. Solve for d(kappas)/d(delta_l) = -H^-1 * C
    try:
        # Use a robust solver for the linear system
        dk_ddl = np.linalg.solve(H, -C)
    except np.linalg.LinAlgError:
        logging.warning("[J_task] Hessian is singular. Using pseudo-inverse.")
        H_inv = np.linalg.pinv(H)
        dk_ddl = -H_inv @ C

    # 5. Final Task Jacobian J_task = J_kin @ dk_ddl
    J_task = J_kin @ dk_ddl
    return J_task

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

    return error, jacobian

def solve_ik(target_pose, params, kappas_guess=None, delta_l_guess=None):
    """Solves the IK problem using the analytical task jacobian."""
    if 'Cosserat' not in params:
        params['Cosserat'] = {'num_elements_pss': 10, 'num_elements_cms1': 5, 'num_elements_cms2': 5}

    if kappas_guess is None:
        _, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
        kappas_guess = np.zeros((3, n_pss + n_cms1 + n_cms2))
    
    if delta_l_guess is None:
        delta_l_guess = np.zeros(8)

    bounds_config = params.get('Bounds', {}).get('delta_l_bounds', [-0.05, 0.05])
    if isinstance(bounds_config[0], (list, tuple)):
        bounds = tuple(zip(*bounds_config))
    else:
        min_b, max_b = bounds_config[0], bounds_config[1]
        bounds = ([min_b] * 8, [max_b] * 8)
    
    # Memoization cache to avoid recomputing
    memo = {}
    def get_res_and_jac(delta_l_motor_tuple):
        if delta_l_motor_tuple not in memo:
            memo[delta_l_motor_tuple] = residual_and_jacobian(np.array(delta_l_motor_tuple), kappas_guess, target_pose, params)
        return memo[delta_l_motor_tuple]

    def fun_for_lsq(delta_l_motor):
        return get_res_and_jac(tuple(delta_l_motor))[0]

    def jac_for_lsq(delta_l_motor):
        return get_res_and_jac(tuple(delta_l_motor))[1]

    result = least_squares(
        fun_for_lsq,
        delta_l_guess,
        jac=jac_for_lsq,
        bounds=bounds,
        method='trf', 
        max_nfev=50,
        ftol=1e-6, xtol=1e-6, gtol=1e-6,
        verbose=2
    )

    if result.success:
        return {"success": True, "delta_l": result.x, "result": result}
    else:
        logging.error(f"[IK Solver] IK solver failed to converge. Message: {result.message}")
        return {"success": False, "delta_l": result.x, "result": result}

if __name__ == '__main__':
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
    ik_result = solve_ik(target_pose, config_params)

    if ik_result["success"]:
        print("\n✅✅✅ IK Solver converged successfully!")
        final_delta_l = ik_result['delta_l']
        print("\n3. Verifying result...")
        final_kappas_res = solve_static_equilibrium(np.zeros_like(kappas_target), final_delta_l, config_params)
        final_kappas = final_kappas_res['kappas_solution']
        if final_kappas is not None:
            final_pose, _ = forward_kinematics(final_kappas, config_params)
            final_error = np.linalg.norm(final_pose[:3, 3] - target_pose[:3, 3])
            print(f"  - Found delta_l: {np.round(final_delta_l, 4)}")
            print(f"  - Final position error: {final_error*1000:.4f} mm")
            if final_error < 1e-3:
                print("  - ✅ Accuracy test passed!")
            else:
                print("  - ❌ Accuracy test failed.")
        else:
            print("  - ❌ Verification failed: Inner-loop solver could not converge with the IK result.")
    else:
        print("\n❌❌❌ IK Solver FAILED to converge.")

    print("\n--- Self-Test Complete ---")