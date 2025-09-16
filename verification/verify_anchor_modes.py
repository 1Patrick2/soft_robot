# verification/verify_anchor_modes.py
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.solver import solve_static_equilibrium_diff4
from src.kinematics import forward_kinematics
from src.statics import calculate_drive_mapping, calculate_actuation_jacobian_analytical, expand_diff4_to_motor8

# Suppress scientific notation for clearer output
np.set_printoptions(suppress=True, precision=6)

print("--- Running verify_anchor_modes.py (Validation Step 1 from test.md) ---")

# 1. Load base config and define test case
params = load_config('config/config.json')
q0 = np.zeros(6)
# Use a non-zero diff4 input to see meaningful changes
diff4_test = np.array([0.03, 0.03, 0.0, 0.0]) 

# 2. Define test modes
modes_to_test = {
    "base": {"cable_anchor_mode": "base", "cable_anchor_blend": 0.0},
    "pss_end": {"cable_anchor_mode": "pss_end", "cable_anchor_blend": 1.0}, # blend is irrelevant but set for clarity
    "blend_0.5": {"cable_anchor_mode": "blend", "cable_anchor_blend": 0.5}
}

# 3. Iterate through modes and run tests
for name, mode_opts in modes_to_test.items():
    print(f"\n{'='*20} TESTING MODE: {name.upper()} {'='*20}")
    
    # Update params for the current mode
    params['ModelOptions'] = mode_opts
    print(f"  - Params updated with: {mode_opts}")

    # Call the solver
    res = solve_static_equilibrium_diff4(q0, diff4_test, params)
    
    if res['q_solution'] is not None:
        q_eq = res['q_solution']
        T_final, _ = forward_kinematics(q_eq, params)
        delta_l_robot = calculate_drive_mapping(q_eq, params)
        J_act = calculate_actuation_jacobian_analytical(q_eq, params)

        print(f"  - Solver SUCCESSFUL.")
        print(f"  - Equilibrium q: {q_eq}")
        print(f"  - Tip Position (T[:3,3]): {T_final[:3,3]}")
        print(f"  - delta_l_robot: {delta_l_robot}")
        print(f"  - Actuation Jacobian (J_act, first 3 columns for PSS):")
        print(J_act[:, 0:3])
    else:
        print(f"  - Solver FAILED to find a solution for this mode.")

print(f"\n{'='*20} VERIFICATION COMPLETE {'='*20}")
