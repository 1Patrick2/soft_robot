import numpy as np
import sys
import os

# =====================================================================
# === Cosserat Rod Kinematics Engine
# =====================================================================

EPSILON = 1e-9

def skew(v):
    """Converts a 3-element vector to a 3x3 skew-symmetric matrix."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def discretize_robot(params):
    """
    Discretizes the robot into a series of elements based on config parameters.
    
    Returns:
        element_lengths (np.ndarray): Array of each element's length ds.
        segment_counts (tuple): A tuple containing the number of elements 
                                in each segment (n_pss, n_cms1, n_cms2).
    """
    cosserat_params = params.get('Cosserat', {})
    n_pss = cosserat_params.get('num_elements_pss', 10)
    n_cms1 = cosserat_params.get('num_elements_cms1', 5)
    n_cms2 = cosserat_params.get('num_elements_cms2', 5)
    
    Lp = params['Geometry']['PSS_initial_length']
    Lc1 = params['Geometry']['CMS_proximal_length']
    Lc2 = params['Geometry']['CMS_distal_length']

    ds_pss = Lp / n_pss if n_pss > 0 else 0
    ds_cms1 = Lc1 / n_cms1 if n_cms1 > 0 else 0
    ds_cms2 = Lc2 / n_cms2 if n_cms2 > 0 else 0

    element_lengths = np.concatenate([
        np.full(n_pss, ds_pss),
        np.full(n_cms1, ds_cms1),
        np.full(n_cms2, ds_cms2)
    ])

    segment_counts = (n_pss, n_cms1, n_cms2)

    return element_lengths, segment_counts

def forward_kinematics(kappas, params, return_all_transforms=False):
    """
    [V-Cosserat.CoM] Calculates the forward kinematics of the Cosserat rod model.
    Also computes the Center of Mass (CoM) for each discrete element.
    """
    element_lengths, (n_pss, n_cms1, n_cms2) = discretize_robot(params)
    num_elements = len(element_lengths)

    if kappas.shape[1] != num_elements:
        raise ValueError(f"Shape of kappas ({kappas.shape[1]}) does not match number of elements ({num_elements})")

    e3 = np.array([0, 0, 1])
    T = np.identity(4)
    
    all_transforms = [T]
    com_positions = []

    for i in range(num_elements):
        kappa_i = kappas[:, i]
        ds = element_lengths[i]
        
        T_base_element = T

        # --- Calculate CoM for the current element i ---
        # Approximate local CoM at the center of the element
        p_com_local = np.array([0, 0, ds / 2.0])
        # Transform to world frame using the transform at the element's base
        p_com_world = T_base_element[:3, :3] @ p_com_local + T_base_element[:3, 3]
        com_positions.append(p_com_world)

        # --- RK4 for a single element ---
        R_base = T_base_element[:3, :3]
        p_base = T_base_element[:3, 3]

        def p_dot(R_local): return R_local @ e3
        def R_dot(R_local): return R_local @ skew(kappa_i)

        k1_p = p_dot(R_base); k1_R = R_dot(R_base)
        k2_p = p_dot(R_base + 0.5 * ds * k1_R); k2_R = R_dot(R_base + 0.5 * ds * k1_R)
        k3_p = p_dot(R_base + 0.5 * ds * k2_R); k3_R = R_dot(R_base + 0.5 * ds * k2_R)
        k4_p = p_dot(R_base + ds * k3_R); k4_R = R_dot(R_base + ds * k3_R)

        p_new = p_base + (ds / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        R_new = R_base + (ds / 6.0) * (k1_R + 2*k2_R + 2*k3_R + k4_R)
        
        U, _, Vt = np.linalg.svd(R_new)
        R_new = U @ Vt

        T = np.identity(4)
        T[:3, :3] = R_new
        T[:3, 3] = p_new
        all_transforms.append(T)

    T_final = T
    com_positions_arr = np.array(com_positions)

    if return_all_transforms:
        return T_final, com_positions_arr, all_transforms
    else:
        return T_final, com_positions_arr

if __name__ == '__main__':
    print("--- Cosserat Kinematics Module Self-Test (with CoM) ---")
    from src.utils.read_config import load_config
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_params = load_config(os.path.join(project_root, 'config', 'config.json'))
    
    if 'Cosserat' not in config_params:
        config_params['Cosserat'] = {
            'num_elements_pss': 10,
            'num_elements_cms1': 5,
            'num_elements_cms2': 5
        }

    element_lengths, (n_pss, n_cms1, n_cms2) = discretize_robot(config_params)
    num_elements = n_pss + n_cms1 + n_cms2
    L_total = np.sum(element_lengths)
    print(f"Robot discretized into {num_elements} elements (PSS:{n_pss}, CMS1:{n_cms1}, CMS2:{n_cms2}). Total length: {L_total:.4f} m")

    print("\n--- Test 1: Straight Pose (Tip and CoM) ---")
    kappas_straight = np.zeros((3, num_elements))
    T_straight, coms_straight = forward_kinematics(kappas_straight, config_params)
    p_straight = T_straight[:3, 3]
    print(f"  - Model Tip Position: {p_straight}")
    print(f"  - Expected Tip Position: [0. 0. {L_total:.4f}]")
    pass1_tip = np.allclose(p_straight, [0, 0, L_total])
    print(f"  - Tip Position Test: {'✅ PASS' if pass1_tip else '❌ FAIL'}")

    # Sanity check the CoM of the last element
    last_com_z = coms_straight[-1, 2]
    last_ds = element_lengths[-1]
    expected_last_com_z = L_total - last_ds / 2.0
    print(f"\n  - Last Element CoM Z: {last_com_z:.6f}")
    print(f"  - Expected Last CoM Z: {expected_last_com_z:.6f}")
    pass1_com = np.isclose(last_com_z, expected_last_com_z)
    print(f"  - CoM Position Test: {'✅ PASS' if pass1_com else '❌ FAIL'}")

    print("\n--- Test 2: Simple Bend ---")
    kappas_bend = np.zeros((3, num_elements))
    k_test = 5.0
    kappas_bend[1, :] = k_test # Apply a constant curvature of ky=5.0
    T_bend, _ = forward_kinematics(kappas_bend, config_params)
    p_bend = T_bend[:3, 3]
    
    if np.isclose(k_test, 0):
        p_analytical = np.array([0, 0, L_total])
    else:
        p_analytical = np.array([
            (1/k_test) * (1 - np.cos(k_test * L_total)),
            0,
            (1/k_test) * np.sin(k_test * L_total)
        ])

    print(f"  - Model Tip Position: {p_bend}")
    print(f"  - Analytical Tip Position: {p_analytical}")
    error = np.linalg.norm(p_bend - p_analytical)
    print(f"  - Error: {error:.6f}")
    pass2 = error < 1e-3
    print(f"  - Bending Test: {'✅ PASS' if pass2 else '❌ FAIL'}")

    print("\n--- Self-Test Complete ---")
    if pass1_tip and pass1_com and pass2:
        print("\n✅✅✅ All Kinematics tests passed!")
    else:
        print("\n❌ Some Kinematics tests failed.")