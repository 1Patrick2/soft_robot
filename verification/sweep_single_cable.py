import numpy as np
import json
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cosserat.kinematics import forward_kinematics, discretize_robot
from src.cosserat.solver import solve_static_equilibrium

def sweep_single_cable(cable_index, params, dl_vals):
    """
    Performs a sweep of a single cable, pulling it from 0 to max and back to 0.
    V7: Definitive formatting fix.
    """
    element_lengths, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
    num_elements = len(element_lengths)
    q_guess = np.zeros((3, num_elements))
    
    solver_options = {'ftol': 1e-9, 'gtol': 1e-8, 'maxiter': 2000}

    # Get geometry for analytical Z calculation
    geo = params['Geometry']
    L_pss = geo['PSS_initial_length']
    L_cms1 = geo['CMS_proximal_length']
    L_cms2 = geo['CMS_distal_length']

    print(f"--- Starting Single Cable Sweep (Cable {cable_index}) ---")
    
    # Define header strings
    theta_p = "\u03b8_pss"
    theta_1 = "\u03b8_cms1"
    theta_2 = "\u03b8_cms2"
    # Corrected header with all 10 columns in the correct order
    header = f"{'Step':<4} | {'dl_val':<7} | {'Converged':<9} | {'Tip Z':<9} | {'z_cms1':<9} | {'Ana Z':<9} | {theta_p:<8} | {theta_1:<8} | {theta_2:<8} | {'K Norm':<10}"
    print(header)
    print("-" * len(header))

    for i, dl in enumerate(dl_vals):
        dl_vec = np.zeros(8)
        dl_vec[cable_index] = dl
        
        res = solve_static_equilibrium(q_guess, dl_vec, params, solver_options=solver_options)
        k = res.get('kappas_solution')
        
        converged = k is not None
        
        if converged:
            q_guess = k  # Warm start for the next step
            
            T_tip, _ = forward_kinematics(k, params)
            T_cms1, _ = forward_kinematics(k, params, upto_element='cms1')
            tip_z_real = T_tip[2, 3]
            z_cms1_real = T_cms1[2, 3]

            theta_pss = 0.0
            for j in range(n_pss):
                bending_kappa_norm = np.linalg.norm(k[0:2, j])
                theta_pss += bending_kappa_norm * element_lengths[j]

            theta_cms1 = 0.0
            for j in range(n_pss, n_pss + n_cms1):
                bending_kappa_norm = np.linalg.norm(k[0:2, j])
                theta_cms1 += bending_kappa_norm * element_lengths[j]

            theta_cms2 = 0.0
            for j in range(n_pss + n_cms1, n_pss + n_cms1 + n_cms2):
                bending_kappa_norm = np.linalg.norm(k[0:2, j])
                theta_cms2 += bending_kappa_norm * element_lengths[j]

            z_tip_analytical = L_pss * np.cos(theta_pss) + L_cms1 * np.cos(theta_cms1) + L_cms2 * np.cos(theta_cms2)
            
            kappa_norm_total = np.linalg.norm(k)

            # Corrected print statement with 10 values in the correct order
            print(f"{i:<4} | {dl:<7.4f} | {str(converged):<9} | {tip_z_real:<9.4f} | {z_cms1_real:<9.4f} | {z_tip_analytical:<9.4f} | {np.rad2deg(theta_pss):<8.2f} | {np.rad2deg(theta_cms1):<8.2f} | {np.rad2deg(theta_cms2):<8.2f} | {kappa_norm_total:<10.4f}")
        else:
            # Corrected print statement for failure case with 10 values
            print(f"{i:<4} | {dl:<7.4f} | {str(converged):<9} | {'N/A':<9} | {'N/A':<9} | {'N/A':<9} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<10}")

if __name__ == '__main__':
    # Load parameters from config.json
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    with open(config_path, 'r') as f:
        params = json.load(f)

    # Define the sweep values
    max_dl = 0.02 
    num_steps = 10
    forward_sweep = np.linspace(0, max_dl, num_steps)
    backward_sweep = np.linspace(max_dl, 0, num_steps)
    dl_values = np.concatenate([forward_sweep, backward_sweep[1:]])

    # --- Run the sweep for a short cable (index 0) ---
    print("\n" + "="*95)
    print("  VERIFICATION: SWEEPING SHORT CABLE (INDEX 0)")
    print("="*95)
    sweep_single_cable(cable_index=0, params=params, dl_vals=dl_values)
    
    # --- Run the sweep for a long cable (index 4) ---
    print("\n" + "="*95)
    print("  VERIFICATION: SWEEPING LONG CABLE (INDEX 4)")
    print("="*95)
    sweep_single_cable(cable_index=4, params=params, dl_vals=dl_values)
