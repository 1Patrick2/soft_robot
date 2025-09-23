# verification/debug_single_path.py

import numpy as np
import sys
import os

# --- Add project root to path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Correctly import from the 'cosserat' directory ---
from src.utils.read_config import load_config
from src.cosserat.kinematics import forward_kinematics, discretize_robot
from src.cosserat.statics import calculate_total_potential_energy
from src.cosserat.solver import solve_static_equilibrium

def debug_trace_single_path(delta_l_start, params, num_steps=20):
    """Traces a single path with high-precision and detailed logging as per test.md A.1"""
    element_lengths, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
    num_elements = len(element_lengths)
    path = np.linspace(delta_l_start, np.zeros_like(delta_l_start), num_steps)
    q_guess = np.zeros((3, num_elements))

    # Use high-precision solver options for this diagnostic test
    solver_opts = {'ftol': 1e-9, 'gtol': 1e-8, 'maxiter': 2000}

    records = []
    print("\n--- Starting High-Precision Path Trace ---")
    print(f"{'Step':<5} | {'Converged':<10} | {'Tip Position (X, Y, Z)':<30} | {'Kappa Norm':<15} | {'Energy (U)':<15} | {'Grad Norm':<15} | {'Solver Iters':<10}")
    print("-" * 120)

    for i, dl in enumerate(path):
        res = solve_static_equilibrium(q_guess, dl, params, solver_options=solver_opts)
        kappas = res.get('kappas_solution')
        converged = (kappas is not None)
        grad_norm = None
        U = None
        tip = None
        kappa_norm = None
        iters = None

        if converged:
            q_guess = kappas  # Warm start for the next step
            T, _ = forward_kinematics(kappas, params)
            tip = T[:3,3].copy()
            kappa_norm = np.linalg.norm(kappas)
            U = calculate_total_potential_energy(kappas, dl, params)
            try:
                iters = res['result'].nit
                grad_norm = np.linalg.norm(res['result'].jac) if hasattr(res['result'], 'jac') else None
            except Exception:
                iters = None
        
        records.append({
            "step": i, "dl": dl.copy(), "converged": converged, "tip": tip,
            "kappa_norm": kappa_norm, "U": U, "grad_norm": grad_norm, "nit": iters
        })
        
        tip_str = f"[{tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f}]" if tip is not None else "None"
        kappa_norm_str = f"{kappa_norm:.4f}" if kappa_norm is not None else "None"
        U_str = f"{U:.4e}" if U is not None else "None"
        grad_norm_str = f"{grad_norm:.4e}" if grad_norm is not None else "None"
        iters_str = f"{iters}" if iters is not None else "None"

        print(f"{i:<5} | {str(converged):<10} | {tip_str:<30} | {kappa_norm_str:<15} | {U_str:<15} | {grad_norm_str:<15} | {iters_str:<10}")
        
        if not converged:
            print("--- Path tracing failed at this step. ---")
            break

    return records

if __name__ == '__main__':
    # Load config
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)
    np.set_printoptions(precision=6, suppress=True)

    # Define a representative boundary actuation vector
    delta_l_bounds = params.get('Bounds', {}).get('delta_l_bounds', [-0.12, 0.12])
    max_actuation = delta_l_bounds[1]
    
    delta_l_start_test = np.zeros(8)
    delta_l_start_test[0] = max_actuation # Pull cable 0 to its maximum

    # Run the diagnostic trace
    debug_records = debug_trace_single_path(delta_l_start_test, params, num_steps=20)

    print("\n--- Diagnostic Run Complete ---")
    num_converged = sum(1 for r in debug_records if r['converged'])
    print(f"Summary: {num_converged} out of {len(debug_records)} steps converged.")
