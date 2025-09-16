# verification/verify_pss_stiffness_scan.py
import numpy as np
import sys
import os
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.solver import solve_static_equilibrium_diff4
from src.kinematics import forward_kinematics

if __name__ == '__main__':
    print("--- PSS Stiffness Scan Across Anchor Modes (Validation Step 2) ---")
    
    base_params = load_config('config/config.json')
    q0 = np.zeros(6)
    
    # Define stiffness values and test input
    pss_stiffness_values = [0.5, 1.0, 2.0, 4.0, 8.0]
    test_diff4 = np.array([0.05, 0, 0, 0]) # X-positive drive

    modes_to_test = ["base", "pss_end"]

    np.set_printoptions(precision=6, suppress=True)
    print(f"Scanning PSS stiffness values: {pss_stiffness_values}")
    print(f"Test drive diff4: {test_diff4}\n")

    for mode in modes_to_test:
        print(f"\n{'='*20} TESTING MODE: {mode.upper()} {'='*20}")
        
        # Set the anchor mode for this entire loop
        params = copy.deepcopy(base_params)
        params['ModelOptions']['cable_anchor_mode'] = mode

        for pss_stiff in pss_stiffness_values:
            print(f"--- Testing PSS Stiffness = {pss_stiff:.2f} ---")
            
            # Update stiffness
            params['Stiffness']['pss_total_equivalent_bending_stiffness'] = pss_stiff
            
            # Run solver
            res = solve_static_equilibrium_diff4(q0, test_diff4, params)
            q_eq = res['q_solution']
            
            if q_eq is not None:
                T, _ = forward_kinematics(q_eq, params)
                pos = T[:3,3]
                xy_displacement = np.linalg.norm(pos[:2])
                print(f"  -> End Position: [X={pos[0]:.5f}, Y={pos[1]:.5f}, Z={pos[2]:.5f}]")
                print(f"     - XY Displacement: {xy_displacement:.5f} m")
            else:
                print("  -> Solver failed.")
