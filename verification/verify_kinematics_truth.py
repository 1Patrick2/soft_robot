import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.kinematics import forward_kinematics

def run_diagnostic_checks():
    """ 
    Runs a series of diagnostic checks on the kinematics model as per test.md.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    params = load_config(config_path)
    Lp = params['Geometry']['PSS_initial_length']
    Lc1 = params['Geometry']['CMS_proximal_length']
    Lc2 = params['Geometry']['CMS_distal_length']
    L_total = Lp + Lc1 + Lc2

    print(f"--- Kinematics Diagnostic Report ---")
    print(f"Theoretical Straight Length (Lp+Lc1+Lc2): {L_total:.4f} m")
    print("-" * 35)

    # 1. Straight Pose Verification
    print("\nStep 1: Straight Pose Verification")
    q_straight = np.zeros(6)
    T_straight, _ = forward_kinematics(q_straight, params)
    z_straight_model = T_straight[2, 3]
    print(f"  - Input q: {q_straight}")
    print(f"  - Model Output Z: {z_straight_model:.6f}")
    print(f"  - Theoretical Z:  {L_total:.6f}")
    if np.isclose(z_straight_model, L_total):
        print("  - ✅ PASS: Model calculates straight height correctly.")
    else:
        print(f"  - ❌ FAIL: Discrepancy of {abs(z_straight_model - L_total):.6f} m.")

    print("-" * 35)

    # 2. Single Segment Bend Verification
    print("\nStep 2: Single Segment Bend Verification")
    q_bend_pss = np.array([0.2, 0, 0, 0, 0, 0]) # Only PSS bends
    T_bend, _ = forward_kinematics(q_bend_pss, params)
    z_bend_model = T_bend[2, 3]
    print(f"  - Input q: {q_bend_pss}")
    print(f"  - Model Output Z: {z_bend_model:.6f}")
    print(f"  - Straight Height:  {L_total:.6f}")
    if z_bend_model < L_total:
        print(f"  - ✅ PASS: Bent height ({z_bend_model:.4f}) is correctly less than straight height ({L_total:.4f}).")
    else:
        print(f"  - ❌ FAIL: Bent height ({z_bend_model:.4f}) is NOT less than straight height ({L_total:.4f}). Z-axis calculation is non-physical.")

    print("-" * 35)

    # 3. Segment-by-Segment Height Trace
    print("\nStep 3: Segment-by-Segment Height Trace")
    q_trace = np.array([0.2, 0.1, 0.3, 0.4, 0.5, 0.6]) # A general bending case
    T_final, _, T_pss, T_base_cms2 = forward_kinematics(q_trace, params, return_all_transforms=True)
    z_pss_end = T_pss[2, 3]
    z_cms1_end = T_base_cms2[2, 3]
    z_cms2_end = T_final[2, 3]
    print(f"  - Input q: {q_trace}")
    print(f"  - PSS End Z:      {z_pss_end:.6f} (Segment Length: {Lp})")
    print(f"  - CMS1 End Z:     {z_cms1_end:.6f} (Cumulative Length: {Lp+Lc1})")
    print(f"  - CMS2 End Z (Tip): {z_cms2_end:.6f} (Cumulative Length: {Lp+Lc1+Lc2})")

    print("\n--- End of Report ---")

if __name__ == '__main__':
    run_diagnostic_checks()
