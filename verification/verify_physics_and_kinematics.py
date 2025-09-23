# verification/verify_physics_and_kinematics.py

import numpy as np
import sys
import os

# --- Add project root to path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Correctly import from the 'cosserat' directory ---
from src.utils.read_config import load_config
from src.cosserat.kinematics import forward_kinematics, discretize_robot
from src.cosserat.solver import solve_static_equilibrium

def run_tests():
    """Runs a series of sanity checks as defined in test.md."""
    
    print("--- Loading Configuration ---")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)
    np.set_printoptions(precision=6, suppress=True)

    element_lengths, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
    num_elements = len(element_lengths)
    L_total = np.sum(element_lengths)
    print(f"Robot discretized into {num_elements} elements. Total length: {L_total:.4f} m")

    # ====================================================================
    # 1. 直杆 sanity check
    # ====================================================================
    print("\n--- Test 1: Straight Rod Sanity Check ---")
    print("Input: kappa = all zeros")
    kappas_zero = np.zeros((3, num_elements))
    T_final_zero, _ = forward_kinematics(kappas_zero, params)
    
    print("Resulting Transformation Matrix T_final:")
    print(T_final_zero)
    
    pos_final = T_final_zero[:3, 3]
    pos_expected = np.array([0, 0, L_total])
    error = np.linalg.norm(pos_final - pos_expected)
    
    print(f"\nExpected end position: {pos_expected}")
    print(f"Actual end position:   {pos_final}")
    print(f"Position error: {error:.6e}")
    
    if error < 1e-9:
        print("✅ PASS: forward_kinematics correctly produces a straight rod.")
    else:
        print("❌ FAIL: forward_kinematics does NOT produce a straight rod for zero curvature.")

    # ====================================================================
    # 2. 单弯曲 sanity check
    # ====================================================================
    print("\n--- Test 2: Single Bend Sanity Check ---")
    BEND_KAPPA = 5.0
    print(f"Input: kappa = [{BEND_KAPPA}, 0, 0] for all elements")
    kappas_bend = np.zeros((3, num_elements))
    kappas_bend[1, :] = BEND_KAPPA # Apply a constant curvature of ky=5.0
    T_final_bend, _ = forward_kinematics(kappas_bend, params)
    pos_final_bend = T_final_bend[:3, 3]
    
    # Theoretical position for constant curvature in x-z plane
    theta = BEND_KAPPA * L_total
    r = 1 / BEND_KAPPA
    pos_expected_bend = np.array([r * (1 - np.cos(theta)), 0, r * np.sin(theta)])

    print("Resulting end position:")
    print(pos_final_bend)
    print(f"(Theoretical end position for constant curvature: {pos_expected_bend})")

    if pos_final_bend[1] < 1e-9 and pos_final_bend[0] > 0 and pos_final_bend[2] > 0:
        print("✅ PASS: Robot bends in the correct plane (Y=0) and general direction.")
    else:
        print("❌ FAIL: Robot does not bend in the expected XZ-plane.")

    # ====================================================================
    # 3. 单 cable 拉动 sanity check
    # ====================================================================
    print("\n--- Test 3: Single Cable Pull Sanity Check ---")
    delta_l_pull = np.zeros(8)
    PULL_AMOUNT = 0.01 # 1 cm pull
    CABLE_IDX = 0
    delta_l_pull[CABLE_IDX] = PULL_AMOUNT
    print(f"Input: delta_l = {delta_l_pull}")

    kappas_guess = np.zeros((3, num_elements))
    solve_result = solve_static_equilibrium(kappas_guess, delta_l_pull, params)
    kappas_eq = solve_result.get("kappas_solution")

    if kappas_eq is not None:
        print("✅ Solver converged.")
        T_final_pull, _ = forward_kinematics(kappas_eq, params)
        pos_final_pull = T_final_pull[:3, 3]
        print("Resulting end position:")
        print(pos_final_pull)
        # For cable 0, we expect movement primarily in the +X direction
        if pos_final_pull[0] > 0 and abs(pos_final_pull[1]) < pos_final_pull[0]:
            print("✅ PASS: Robot appears to bend in the correct direction (+X).")
        else:
            print("❌ FAIL: Robot did not bend in the expected direction.")
    else:
        print("❌ FAIL: Solver did not converge.")

if __name__ == '__main__':
    run_tests()
