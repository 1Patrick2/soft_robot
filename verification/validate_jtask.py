# file: verification/validate_jtask.py
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cosserat.solver import solve_static_equilibrium
from src.cosserat.kinematics import forward_kinematics, discretize_robot
from src.cosserat.statics import calculate_hessian_approx, calculate_coupling_matrix_C
from src.cosserat.outer_solver import calculate_kinematic_jacobian_numerical
from src.utils.read_config import load_config
from scipy.spatial.transform import Rotation as R

def numeric_dk_dll(kappas_eq, delta_l, params, eps=1e-6):
    n_dl = len(delta_l)
    k_flat = kappas_eq.flatten()
    num_vars = k_flat.size
    dk_num = np.zeros((num_vars, n_dl))
    for i in range(n_dl):
        dl_p = delta_l.copy(); dl_m = delta_l.copy()
        dl_p[i] += eps; dl_m[i] -= eps
        res_p = solve_static_equilibrium(kappas_eq, dl_p, params)
        res_m = solve_static_equilibrium(kappas_eq, dl_m, params)
        kp = res_p['kappas_solution']; km = res_m['kappas_solution']
        if kp is None or km is None:
            raise RuntimeError("Inner solver failed during numeric check.")
        dk_num[:, i] = (kp.flatten() - km.flatten()) / (2*eps)
    return dk_num

def numeric_Jtask(kappas_eq, delta_l, params, eps=1e-6):
    n_dl = len(delta_l)
    Jnum = np.zeros((6, n_dl))
    for i in range(n_dl):
        dl_p = delta_l.copy(); dl_m = delta_l.copy()
        dl_p[i] += eps; dl_m[i] -= eps
        kp = solve_static_equilibrium(kappas_eq, dl_p, params)['kappas_solution']
        km = solve_static_equilibrium(kappas_eq, dl_m, params)['kappas_solution']
        Tp, _ = forward_kinematics(kp, params)
        Tm, _ = forward_kinematics(km, params)
        
        rvp = R.from_matrix(Tp[:3,:3]).as_rotvec()
        rvm = R.from_matrix(Tm[:3,:3]).as_rotvec()
        pose_p = np.concatenate([Tp[:3,3], rvp])
        pose_m = np.concatenate([Tm[:3,3], rvm])
        Jnum[:, i] = (pose_p - pose_m) / (2*eps)
    return Jnum

def validate(config_path='config/config.json'):
    # Correct the path to be relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params = load_config(os.path.join(project_root, config_path))

    if 'Cosserat' not in params:
        params['Cosserat'] = {'num_elements_pss':10,'num_elements_cms1':5,'num_elements_cms2':5}
    _, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
    num_elements = n_pss + n_cms1 + n_cms2

    # pick a non-trivial delta_l to generate a bent configuration away from singularity
    np.random.seed(42)
    delta_l = np.zeros(8)
    delta_l[0] = 0.05 # A significant actuation
    kappas_guess = np.zeros((3, num_elements))
    res = solve_static_equilibrium(kappas_guess, delta_l, params)
    k_eq = res['kappas_solution']
    if k_eq is None:
        raise RuntimeError("Inner solver failed for validation input.")

    # analytic H and C (but they are numerical in current implementation)
    H = calculate_hessian_approx(k_eq, delta_l, params)
    C = calculate_coupling_matrix_C(k_eq, delta_l, params)

    # analytic dk/dl from formula
    damping_factor = 1e-6 # A small damping factor for stability
    H_damped = H + damping_factor * np.identity(H.shape[0])
    try:
        dk_analytic = np.linalg.solve(H_damped, -C)
    except np.linalg.LinAlgError:
        dk_analytic = -np.linalg.pinv(H_damped) @ C

    # numeric dk/dl
    dk_num = numeric_dk_dll(k_eq, delta_l, params, eps=1e-6)

    print("dk analytic vs numeric: norm diff =", np.linalg.norm(dk_analytic - dk_num) / (np.linalg.norm(dk_num)+1e-12))

    # analytic J_task from formula
    Jkin = calculate_kinematic_jacobian_numerical(k_eq, params)
    Jtask_analytic = Jkin @ dk_analytic

    # numeric J_task
    Jtask_num = numeric_Jtask(k_eq, delta_l, params, eps=1e-6)
    print("Jtask analytic vs numeric: norm diff =", np.linalg.norm(Jtask_analytic - Jtask_num) / (np.linalg.norm(Jtask_num)+1e-12))

    return {'dk_rel_err': np.linalg.norm(dk_analytic - dk_num)/(np.linalg.norm(dk_num)+1e-12),
            'J_rel_err': np.linalg.norm(Jtask_analytic - Jtask_num)/(np.linalg.norm(Jtask_num)+1e-12)}

if __name__ == '__main__':
    # I need to import outer_solver to get the numerical J_kin
    # This is a bit circular, but necessary for the test as written
    try:
        from src.cosserat.outer_solver import calculate_kinematic_jacobian_numerical
    except ImportError:
        print("Could not import calculate_kinematic_jacobian_numerical from outer_solver.")
        # Define it locally if it's not there, assuming it was deleted
        def calculate_kinematic_jacobian_numerical(kappas, params, epsilon=1e-7):
            num_k_vars = kappas.size
            J_kin_num = np.zeros((6, num_k_vars))
            kappas_flat = kappas.flatten()
            for i in range(num_k_vars):
                k_plus = kappas_flat.copy(); k_plus[i] += epsilon
                T_plus, _ = forward_kinematics(k_plus.reshape(kappas.shape), params)
                rvp = R.from_matrix(T_plus[:3,:3]).as_rotvec()
                pose_p = np.concatenate([T_plus[:3,3], rvp])
                k_minus = kappas_flat.copy(); k_minus[i] -= epsilon
                T_minus, _ = forward_kinematics(k_minus.reshape(kappas.shape), params)
                rvm = R.from_matrix(T_minus[:3,:3]).as_rotvec()
                pose_m = np.concatenate([T_minus[:3,3], rvm])
                J_kin_num[:, i] = (pose_p - pose_m) / (2*eps)
            return J_kin_num

    print(validate())
