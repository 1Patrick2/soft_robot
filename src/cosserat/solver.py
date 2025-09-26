import numpy as np
from scipy.optimize import minimize
import sys
import os
import logging

# --- Cosserat Model Dependencies ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cosserat.statics import (
    calculate_total_potential_energy,
    calculate_total_gradient
)
from src.cosserat.kinematics import discretize_robot
from src.utils.read_config import load_config

# =====================================================================
# === Cosserat Inner-Loop Solver
# =====================================================================

def solve_static_equilibrium(kappas_guess, delta_l_motor, params, solver_options=None):
    """
    Solves for the static equilibrium configuration of the Cosserat rod model.
    Finds the curvatures `kappas` that minimize the total potential energy for a 
    given motor displacement `delta_l_motor`.
    Accepts custom solver options.
    """
    
    kappas_guess_flat = kappas_guess.flatten()
    shape_kappas = kappas_guess.shape

    def objective_function(kappas_flat):
        kappas = kappas_flat.reshape(shape_kappas)
        return calculate_total_potential_energy(kappas, delta_l_motor, params)

    def jacobian_function(kappas_flat):
        kappas = kappas_flat.reshape(shape_kappas)
        grad = calculate_total_gradient(kappas, delta_l_motor, params)
        return grad.flatten()

    if solver_options is None:
        lbfgsb_opts = {
            'ftol': 1e-9,
            'gtol': 1e-6,
            'maxiter': 1500
        }
    else:
        lbfgsb_opts = solver_options

    kappa_bound = params.get('Solver', {}).get('kappa_bound', 20.0)
    bounds = [(-kappa_bound, kappa_bound)] * kappas_guess.size

    result = minimize(
        objective_function, 
        kappas_guess_flat, 
        method='L-BFGS-B',
        jac=jacobian_function, 
        bounds=bounds,
        options=lbfgsb_opts
    )

    if result.success:
        kappas_solution = result.x.reshape(shape_kappas)
        return {"kappas_solution": kappas_solution, "result": result}
    else:
        # Use logging.debug for less critical failures to avoid spamming the console
        logging.debug(f"[Solver] Inner-loop solver failed to converge. Message: {result.message}")
        return {"kappas_solution": None, "result": result}


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("--- Cosserat Solver Module Self-Test ---")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_params = load_config(os.path.join(project_root, 'config', 'config.json'))
    
    if 'Cosserat' not in config_params:
        config_params['Cosserat'] = {'num_elements_pss': 10, 'num_elements_cms1': 5, 'num_elements_cms2': 5}

    _, (n_pss, n_cms1, n_cms2) = discretize_robot(config_params)
    num_elements = n_pss + n_cms1 + n_cms2

    print("\n--- Test Scenario: Find equilibrium for a simple drive input ---")
    
    kappas_guess = np.zeros((3, num_elements))
    
    delta_l_motor_test = np.zeros(8)
    delta_l_motor_test[0] = 0.01
    delta_l_motor_test[2] = -0.01

    print(f"  - Initial Guess: Straight rod (kappas=0)")
    print(f"  - Drive Input (delta_l_motor): {delta_l_motor_test}")

    # --- Run the solver ---
    solve_result = solve_static_equilibrium(kappas_guess, delta_l_motor_test, config_params)
    
    # --- Analyze result ---
    if solve_result["kappas_solution"] is not None:
        print("\n✅ Solver converged successfully!")
        final_grad_norm = np.linalg.norm(solve_result['result'].jac)
        print(f"  - Final Gradient Norm: {final_grad_norm:.4e} (should be close to zero)")
        
        final_kappas = solve_result["kappas_solution"]
        if np.linalg.norm(final_kappas) > 1e-3:
            print("  - ✅ Resulting configuration is non-trivial (robot has bent).")
        else:
            print("  - ⚠️  WARNING: Solver converged to a near-zero (straight) configuration.")
        
        if final_grad_norm < 1e-5:
            print("  - ✅ Final state is a valid equilibrium point.")
        else:
            print("  - ❌ Final state is NOT a valid equilibrium point (gradient norm too high).")

    else:
        print("\n❌ Solver FAILED to converge.")

    print("\n--- Self-Test Complete ---")
