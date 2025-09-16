import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.solver import solve_static_equilibrium_disp_ctrl
from src.kinematics import forward_kinematics

def run_single_pull_test():
    """
    As per test.md, this script performs a "single-pull" test to verify
    if the robot model is physically responsive to actuation.
    It pulls each of the 8 cables individually and records the resulting end-effector position.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)
    q0_7d = np.array(params['Initial_State']['q0'])
    q0_6d = q0_7d[1:]

    pull_magnitude = 0.06  # 60mm pull, a significant actuation

    print("--- Starting Single-Pull Test ---")
    print(f"Pulling each cable individually with a magnitude of {pull_magnitude * 1000} mm.")
    print("---------------------------------------------------")

    results = {}

    for i in range(8):
        delta_l_motor = np.zeros(8)
        delta_l_motor[i] = pull_magnitude

        # Use a diverse random guess for the solver
        q_guess_rand = np.zeros(6)
        q_guess_rand[::2] = np.random.uniform(-5, 5, 3)
        q_guess_rand[1::2] = np.random.uniform(-np.pi, np.pi, 3)

        print(f"Testing Cable #{i+1}...")
        
        q_eq = solve_static_equilibrium_disp_ctrl(q_guess_rand, delta_l_motor, params)

        if q_eq is not None:
            T_final, _ = forward_kinematics(q_eq, params)
            position = T_final[:3, 3]
            results[f"Cable_{i+1}"] = position
            print(f"  -> SUCCESS: End-effector moved to {np.round(position, 4)}")
        else:
            results[f"Cable_{i+1}"] = "Solver Failed"
            print("  -> FAILED: Solver could not find a solution.")

    print("\n--- Test Summary ---")
    base_position = np.array([0, 0, params['Geometry']['PSS_initial_length'] + params['Geometry']['CMS_proximal_length'] + params['Geometry']['CMS_distal_length']])
    print(f"Reference Straight Position: {np.round(base_position, 4)}")

    all_failed = True
    max_displacement = 0

    for cable, pos in results.items():
        if isinstance(pos, np.ndarray):
            all_failed = False
            displacement = np.linalg.norm(pos - base_position)
            if displacement > max_displacement:
                max_displacement = displacement
            print(f"  - {cable}: Moved to {np.round(pos, 4)} (Displacement: {displacement*1000:.2f} mm)")
        else:
            print(f"  - {cable}: Solver Failed")

    print("\n--- Final Diagnosis ---")
    if all_failed:
        print("RESULT: [CRITICAL] The model is unresponsive. All single-pull tests failed.")
        print("        This strongly suggests the model is too stiff or there is a fundamental issue in the physics/solver.")
    elif max_displacement < 0.01: # Less than 1cm displacement
        print("RESULT: [WARNING] The model is highly unresponsive.")
        print(f"        Maximum displacement was only {max_displacement*1000:.2f} mm.")
        print("        This suggests the model is physically too stiff for the given actuation.")
    else:
        print("RESULT: [SUCCESS] The model is physically responsive.")
        print(f"        Maximum displacement was {max_displacement*1000:.2f} mm.")
        print("        The 'hollow workspace' issue is likely caused by the SAMPLING STRATEGY, not physical stiffness.")
    print("---------------------------------------------------")

if __name__ == '__main__':
    run_single_pull_test()
