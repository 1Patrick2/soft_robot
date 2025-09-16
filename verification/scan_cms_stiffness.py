# verification/scan_cms_stiffness.py
import numpy as np
import sys
import os
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.solver import solve_static_equilibrium_diff4
from src.kinematics import forward_kinematics

if __name__ == '__main__':
    print("--- CMS Stiffness Parameter Scan ---")
    
    base_params = load_config('config/config.json')
    q0 = np.zeros(6)
    
    # 固定 PSS 刚度，扫描 CMS 刚度
    pss_stiffness_fixed = 2.0
    cms_stiffness_values = [0.2, 0.1, 0.05, 0.02]
    
    base_params['Stiffness']['pss_total_equivalent_bending_stiffness'] = pss_stiffness_fixed

    # 定义一个固定的测试驱动 (X正向)
    test_diff4 = np.array([0.05, 0, 0, 0])

    np.set_printoptions(precision=6, suppress=True)
    print(f"Fixed PSS stiffness = {pss_stiffness_fixed}")
    print(f"Scanning CMS stiffness values: {cms_stiffness_values}")
    print(f"Test drive diff4: {test_diff4}\n")

    for cms_stiff in cms_stiffness_values:
        print(f"--- Testing CMS Stiffness = {cms_stiff:.2f} (Ratio PSS/CMS = {pss_stiffness_fixed/cms_stiff:.1f}) ---")
        
        params = copy.deepcopy(base_params)
        params['Stiffness']['cms_bending_stiffness'] = cms_stiff
        
        res = solve_static_equilibrium_diff4(q0, test_diff4, params)
        q_eq = res['q_solution']
        
        if q_eq is not None:
            T, _ = forward_kinematics(q_eq, params)
            pos = T[:3,3]
            print(f"  -> End Position: [X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}]")
            
            xy_displacement = np.linalg.norm(pos[:2])
            z_height = pos[2]
            print(f"     - XY Displacement: {xy_displacement:.4f} m")
            print(f"     - Z Height:        {z_height:.4f} m")
        else:
            print("  -> Solver failed.")
