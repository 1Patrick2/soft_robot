# verification/component_test.py
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.solver import solve_static_equilibrium_diff4
from src.kinematics import forward_kinematics
from src.statics import expand_diff4_to_motor8, calculate_drive_mapping

# Suppress scientific notation for clearer output
np.set_printoptions(suppress=True, precision=6)

print("--- Running component_test.py (Step 2 from test.md) ---")

params = load_config('config/config.json')
q0 = np.zeros(6)

tests = {
  'diff4_xpos': np.array([0.05, 0, 0, 0]),
  'diff4_xneg': np.array([-0.05, 0, 0, 0]),
  'diff4_ypos': np.array([0, 0.05, 0, 0]),
  'diff4_yneg': np.array([0, -0.05, 0, 0]),
  'diff4_diag': np.array([0.03,0.03,0,0])
}

for name,d in tests.items():
    # Note: The test.md code snippet had a call to expand_diff4_to_motor8 that wasn't used.
    # The solve_static_equilibrium_diff4 function handles the expansion internally.
    res = solve_static_equilibrium_diff4(q0, d, params)
    
    if res['q_solution'] is not None:
        q_eq = res['q_solution']
        T,_ = forward_kinematics(q_eq, params)
        # We need to get the motor delta_l that the solver used.
        # The solver uses the *new* mapping, so we call it here for verification.
        delta_l_motor = expand_diff4_to_motor8(d, params)
        delta_l_robot = calculate_drive_mapping(q_eq, params)
        
        print(f"\n--- Test Case: {name} ---")
        print(f"  diff4 input: {d}")
        print(f"  Tip Position (T[:3,3]): {T[:3,3]}")
        print(f"  Equilibrium q: {q_eq}")
        print(f"  Motor delta_l: {delta_l_motor}")
        print(f"  Robot delta_l: {delta_l_robot}")
    else:
        print(f"\n--- Test Case: {name} ---")
        print(f"  diff4 input: {d}")
        print("  Solver failed to find a solution.")

print("\n--- Component test finished. ---")
