# src/statics.py

import numpy as np
from scipy.optimize import approx_fprime
import sys
import os

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kinematics import (
    forward_kinematics,
    calculate_com_jacobians_analytical,
    
)
from src.utils.read_config import load_config

def calculate_drive_mapping(q_6d, params):
    """[V-Final.GeometricExact & MODULAR ANCHOR] Calculates drive mapping based on the exact geometric
    distance between cable anchor points in 3D space. 
    Supports multiple cable anchor modes as per test.md.
    """
    # --- Get parameters ---
    geo = params['Geometry']
    Lp = geo['PSS_initial_length']
    Lc1 = geo['CMS_proximal_length']
    Lc2 = geo['CMS_distal_length']
    r_short = geo['short_lines']['diameter_m'] / 2
    r_long  = geo['long_lines']['diameter_m'] / 2
    alphas_short = np.deg2rad(geo['short_lines']['angles_deg'])
    alphas_long  = np.deg2rad(geo['long_lines']['angles_deg'])

    # --- [NEW] Selectable Anchor Mode & Model Consistency --- 
    model_opts = params.get('ModelOptions', {})
    mode = model_opts.get('cable_anchor_mode', 'base')
    blend = model_opts.get('cable_anchor_blend', 0.0)

    q_for_fk = q_6d.copy()

    if mode == 'pss_end':
        z_anchor = Lp
        q_for_fk[0:2] = 0.0 # Enforce PSS decoupling for FK calculation
    elif mode == 'base':
        z_anchor = 0.0
    else: # 'blend' mode
        z_anchor = (1.0 - blend) * 0.0 + blend * Lp
        # Note: Blending q is complex, for now we only blend jacobian and anchor Z

    # --- Define Start and End Anchor Points ---
    # Start points (at base or PSS end)
    p_start_s = np.vstack([r_short * np.cos(alphas_short), r_short * np.sin(alphas_short), np.full_like(alphas_short, z_anchor)])
    p_start_l = np.vstack([r_long * np.cos(alphas_long),   r_long * np.sin(alphas_long),   np.full_like(alphas_long, z_anchor)])

    # [BUGFIX] End points are defined locally (z=0) in their respective frames
    p_end_s_local = np.vstack([r_short * np.cos(alphas_short), r_short * np.sin(alphas_short), np.zeros_like(alphas_short)])
    p_end_l_local = np.vstack([r_long * np.cos(alphas_long),   r_long * np.sin(alphas_long),   np.zeros_like(alphas_long)])

    # --- FK to get segment end transforms ---
    T_final, _, _, T_base_cms2 = forward_kinematics(q_for_fk, params, return_all_transforms=True)
    T_cms1_end = T_base_cms2
    T_cms2_end = T_final

    # --- Transform end points to world frame ---
    p_end_s_world = (T_cms1_end @ np.vstack([p_end_s_local, np.ones_like(alphas_short)]))[:3, :]
    p_end_l_world = (T_cms2_end @ np.vstack([p_end_l_local, np.ones_like(alphas_long)]))[:3, :]

    # --- Calculate cable lengths ---
    # The straight length (l0) depends on where the cable starts
    l0_s = (Lp - z_anchor) + Lc1
    l0_l = (Lp - z_anchor) + Lc1 + Lc2

    len_s_bent = np.linalg.norm(p_end_s_world - p_start_s, axis=0)
    len_l_bent = np.linalg.norm(p_end_l_world - p_start_l, axis=0)

    delta_s = l0_s - len_s_bent
    delta_l = l0_l - len_l_bent

    return np.concatenate([delta_s, delta_l])
    # --- Get parameters ---
    geo = params['Geometry']
    Lp = geo['PSS_initial_length']
    Lc1 = geo['CMS_proximal_length']
    Lc2 = geo['CMS_distal_length']
    r_short = geo['short_lines']['diameter_m'] / 2
    r_long  = geo['long_lines']['diameter_m'] / 2
    alphas_short = np.deg2rad(geo['short_lines']['angles_deg'])
    alphas_long  = np.deg2rad(geo['long_lines']['angles_deg'])

    # --- [NEW] Selectable Anchor Mode & Model Consistency --- 
    model_opts = params.get('ModelOptions', {})
    mode = model_opts.get('cable_anchor_mode', 'base')
    blend = model_opts.get('cable_anchor_blend', 0.0)

    q_for_fk = q_6d.copy()

    if mode == 'pss_end':
        z_anchor = Lp
        q_for_fk[0:2] = 0.0 # Enforce PSS decoupling for FK calculation
    elif mode == 'base':
        z_anchor = 0.0
    else: # 'blend' mode
        z_anchor = (1.0 - blend) * 0.0 + blend * Lp
        # Note: Blending q is complex, for now we only blend jacobian and anchor Z

    # --- Define Start and End Anchor Points ---
    # Start points (at base or PSS end)
    p_start_s = np.vstack([r_short * np.cos(alphas_short), r_short * np.sin(alphas_short), np.full_like(alphas_short, z_anchor)])
    p_start_l = np.vstack([r_long * np.cos(alphas_long),   r_long * np.sin(alphas_long),   np.full_like(alphas_long, z_anchor)])

    # [BUGFIX] End points are defined locally (z=0) in their respective frames
    p_end_s_local = np.vstack([r_short * np.cos(alphas_short), r_short * np.sin(alphas_short), np.zeros_like(alphas_short)])
    p_end_l_local = np.vstack([r_long * np.cos(alphas_long),   r_long * np.sin(alphas_long),   np.zeros_like(alphas_long)])

    # --- FK to get segment end transforms ---
    T_final, _, _, T_base_cms2 = forward_kinematics(q_for_fk, params, return_all_transforms=True)
    T_cms1_end = T_base_cms2
    T_cms2_end = T_final

    # --- Transform end points to world frame ---
    p_end_s_world = (T_cms1_end @ np.vstack([p_end_s_local, np.ones_like(alphas_short)]))[:3, :]
    p_end_l_world = (T_cms2_end @ np.vstack([p_end_l_local, np.ones_like(alphas_long)]))[:3, :]

    # --- Calculate cable lengths ---
    # The straight length (l0) depends on where the cable starts
    l0_s = (Lp - z_anchor) + Lc1
    l0_l = (Lp - z_anchor) + Lc1 + Lc2

    len_s_bent = np.linalg.norm(p_end_s_world - p_start_s, axis=0)
    len_l_bent = np.linalg.norm(p_end_l_world - p_start_l, axis=0)

    delta_s = l0_s - len_s_bent
    delta_l = l0_l - len_l_bent

    return np.concatenate([delta_s, delta_l])

def calculate_elastic_potential_energy(q_6d, params):
    kp, _, kc1, _, kc2, _ = q_6d
    Lp = params['Geometry']['PSS_initial_length']
    Lc1 = params['Geometry']['CMS_proximal_length']
    Lc2 = params['Geometry']['CMS_distal_length']
    K_bending_pss = params['Stiffness']['pss_total_equivalent_bending_stiffness']
    K_bending_cms = params['Stiffness']['cms_bending_stiffness']
    U_pss_bending = 0.5 * K_bending_pss * Lp * kp**2
    U_cms1_bending = 0.5 * K_bending_cms * Lc1 * kc1**2
    U_cms2_bending = 0.5 * K_bending_cms * Lc2 * kc2**2
    return U_pss_bending + U_cms1_bending + U_cms2_bending

def calculate_gravity_potential_energy(q_6d, params):
    g = params['Mass']['g']
    g_vec = np.array([0, 0, g])
    masses = [params['Mass']['pss_kg'], params['Mass']['cms_proximal_kg'], params['Mass']['cms_distal_kg']]
    _, com_positions = forward_kinematics(q_6d, params)
    com_vectors = [com_positions['pss'], com_positions['cms1'], com_positions['cms2']]
    U_G = sum(m * np.dot(g_vec, p_com) for m, p_com in zip(masses, com_vectors))
    return U_G

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def smooth_max_zero(x, beta=50.0):
    """ [FIXED] A smooth approximation of max(0, x) that is zero at x=0. """
    return x * _sigmoid(beta * x)

def smooth_max_zero_derivative(x, beta=50.0):
    """ [FIXED] The derivative of the new smooth_max_zero function. """
    s = _sigmoid(beta * x)
    return s + beta * x * s * (1 - s)

def calculate_total_potential_energy_disp_ctrl(q_6d, delta_l_motor, params):
    U_robot = calculate_elastic_potential_energy(q_6d, params) + calculate_gravity_potential_energy(q_6d, params)
    k_cable = params['Drive_Properties']['cable_stiffness']
    delta_l_robot = calculate_drive_mapping(q_6d, params)
    stretch = delta_l_motor - delta_l_robot
    stretch_tensioned = smooth_max_zero(stretch)
    U_cable = 0.5 * k_cable * np.sum(stretch_tensioned**2)
    f_pre = params['Drive_Properties'].get('pretension_force_N', 0.0)
    U_pretension = -f_pre * np.sum(delta_l_robot)
    regularization_strength = 1e-5
    U_regularization = 0.5 * regularization_strength * np.sum(q_6d**2)
    return U_robot + U_cable + U_pretension + U_regularization

def calculate_elastic_gradient_analytical(q_6d, params):
    kp, _, kc1, _, kc2, _ = q_6d
    Lp = params['Geometry']['PSS_initial_length']
    Lc1 = params['Geometry']['CMS_proximal_length']
    Lc2 = params['Geometry']['CMS_distal_length']
    K_bending_pss = params['Stiffness']['pss_total_equivalent_bending_stiffness']
    K_bending_cms = params['Stiffness']['cms_bending_stiffness']
    grad = np.zeros(6)
    grad[0] = K_bending_pss * Lp * kp
    grad[2] = K_bending_cms * Lc1 * kc1
    grad[4] = K_bending_cms * Lc2 * kc2
    return grad

def calculate_gravity_gradient_analytical(q_6d, params):
    g = params['Mass']['g']
    mass_p = params['Mass']['pss_kg']
    mass_c1 = params['Mass']['cms_proximal_kg']
    mass_c2 = params['Mass']['cms_distal_kg']
    J_com_dict = calculate_com_jacobians_analytical(q_6d, params)
    grad = g * (mass_p * J_com_dict['pss'][2, :] + mass_c1 * J_com_dict['cms1'][2, :] + mass_c2 * J_com_dict['cms2'][2, :])
    return grad

# def calculate_actuation_jacobian_numerical(q, params, epsilon=1e-6):
#     """ [DEPRECATED] Now replaced by analytical version. """
#     J_act = np.zeros((8, 6))
#     for i in range(6):
#         h = epsilon * max(1.0, abs(q[i]))
#         q_plus = q.copy(); q_plus[i] += h
#         q_minus = q.copy(); q_minus[i] -= h
#         plus = calculate_drive_mapping(q_plus, params)
#         minus = calculate_drive_mapping(q_minus, params)
#         J_act[:, i] = (plus - minus) / (2.0 * h)
#     return J_act

def _skew(v):
    """Converts a 3-element vector to a 3x3 skew-symmetric matrix."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def calculate_actuation_jacobian_analytical(q_6d, params):
    """
    [V6.0 REFACTORED & MODULAR ANCHOR] Calculates the analytical actuation jacobian d(delta_l_robot)/dq.
    This version directly uses the verified jacobians from the kinematics module and supports multiple anchor modes.
    """
    # --- 核心依赖 ---
    from src.kinematics import calculate_kinematic_jacobian_analytical

    # --- Get geometry and config ---
    geo = params['Geometry']
    Lp = geo['PSS_initial_length']
    r_s = geo['short_lines']['diameter_m'] / 2
    r_l = geo['long_lines']['diameter_m'] / 2
    alphas_s = np.deg2rad(geo['short_lines']['angles_deg'])
    alphas_l = np.deg2rad(geo['long_lines']['angles_deg'])

    # --- [NEW] Selectable Anchor Mode & Model Consistency --- 
    model_opts = params.get('ModelOptions', {})
    mode = model_opts.get('cable_anchor_mode', 'base')
    blend = model_opts.get('cable_anchor_blend', 0.0)

    q_for_fk = q_6d.copy()

    if mode == 'pss_end':
        z_anchor = Lp
        q_for_fk[0:2] = 0.0 # Enforce PSS decoupling for FK calculation
    elif mode == 'base':
        z_anchor = 0.0
    else: # 'blend' mode
        z_anchor = (1.0 - blend) * 0.0 + blend * Lp

    # --- Get full kinematic Jacobians from the kinematics module ---
    J_total, J_cms1, _ = calculate_kinematic_jacobian_analytical(q_for_fk, params)

    # --- Get intermediate transforms from kinematics ---
    T_final, _, _, T_base_cms2 = forward_kinematics(q_for_fk, params, return_all_transforms=True)
    T_cms1_end = T_base_cms2
    T_cms2_end = T_final

    # --- Define Anchor Points (with bugfix and new feature) ---
    # Start points (at base or PSS end)
    p_start_s = np.vstack([r_s*np.cos(alphas_s), r_s*np.sin(alphas_s), np.full_like(alphas_s, z_anchor)])
    p_start_l = np.vstack([r_l*np.cos(alphas_l), r_l*np.sin(alphas_l), np.full_like(alphas_l, z_anchor)])

    # [BUGFIX] End points are defined locally (z=0) in their respective frames
    p_end_s_local = np.vstack([r_s*np.cos(alphas_s), r_s*np.sin(alphas_s), np.zeros_like(alphas_s)])
    p_end_l_local = np.vstack([r_l*np.cos(alphas_l), r_l*np.sin(alphas_l), np.zeros_like(alphas_l)])

    # --- Transform end points to world frame ---
    p_end_s_world = (T_cms1_end @ np.vstack([p_end_s_local, np.ones(4)]))[:3,:]
    p_end_l_world = (T_cms2_end @ np.vstack([p_end_l_local, np.ones(4)]))[:3,:]

    # --- Calculate world-frame point jacobians (J_p = J_v - skew(r) @ J_w) ---
    J_p_s_all = np.zeros((4, 3, 6))
    J_p_l_all = np.zeros((4, 3, 6))

    r_s_world_cols = (T_cms1_end[:3, :3] @ p_end_s_local).T
    J_v_s = J_cms1[:3, :]; J_w_s = J_cms1[3:, :]
    for i in range(4):
        J_p_s_all[i, :, :] = J_v_s - _skew(r_s_world_cols[i, :]) @ J_w_s

    r_l_world_cols = (T_cms2_end[:3, :3] @ p_end_l_local).T
    J_v_l = J_total[:3,:]; J_w_l = J_total[3:,:]
    for i in range(4):
        J_p_l_all[i, :, :] = J_v_l - _skew(r_l_world_cols[i, :]) @ J_w_l

    # --- 核心依赖 ---
    from src.kinematics import calculate_kinematic_jacobian_analytical

    # --- Get geometry and config ---
    geo = params['Geometry']
    Lp = geo['PSS_initial_length']
    r_s = geo['short_lines']['diameter_m'] / 2
    r_l = geo['long_lines']['diameter_m'] / 2
    alphas_s = np.deg2rad(geo['short_lines']['angles_deg'])
    alphas_l = np.deg2rad(geo['long_lines']['angles_deg'])

    # --- [NEW] Selectable Anchor Mode & Model Consistency --- 
    model_opts = params.get('ModelOptions', {})
    mode = model_opts.get('cable_anchor_mode', 'base')
    blend = model_opts.get('cable_anchor_blend', 0.0)

    q_for_fk = q_6d.copy()

    if mode == 'pss_end':
        z_anchor = Lp
        q_for_fk[0:2] = 0.0 # Enforce PSS decoupling for FK calculation
    elif mode == 'base':
        z_anchor = 0.0
    else: # 'blend' mode
        z_anchor = (1.0 - blend) * 0.0 + blend * Lp

    # --- Get full kinematic Jacobians from the kinematics module ---
    J_total, J_cms1, _ = calculate_kinematic_jacobian_analytical(q_for_fk, params)

    # --- Get intermediate transforms from kinematics ---
    T_final, _, _, T_base_cms2 = forward_kinematics(q_for_fk, params, return_all_transforms=True)
    T_cms1_end = T_base_cms2
    T_cms2_end = T_final

    # --- Define Anchor Points (with bugfix and new feature) ---
    # Start points (at base or PSS end)
    p_start_s = np.vstack([r_s*np.cos(alphas_s), r_s*np.sin(alphas_s), np.full_like(alphas_s, z_anchor)])
    p_start_l = np.vstack([r_l*np.cos(alphas_l), r_l*np.sin(alphas_l), np.full_like(alphas_l, z_anchor)])

    # [BUGFIX] End points are defined locally (z=0) in their respective frames
    p_end_s_local = np.vstack([r_s*np.cos(alphas_s), r_s*np.sin(alphas_s), np.zeros_like(alphas_s)])
    p_end_l_local = np.vstack([r_l*np.cos(alphas_l), r_l*np.sin(alphas_l), np.zeros_like(alphas_l)])

    # --- Transform end points to world frame ---
    p_end_s_world = (T_cms1_end @ np.vstack([p_end_s_local, np.ones(4)]))[:3,:]
    p_end_l_world = (T_cms2_end @ np.vstack([p_end_l_local, np.ones(4)]))[:3,:]

    # --- Calculate world-frame point jacobians (J_p = J_v - skew(r) @ J_w) ---
    J_p_s_all = np.zeros((4, 3, 6))
    J_p_l_all = np.zeros((4, 3, 6))

    r_s_world_cols = (T_cms1_end[:3, :3] @ p_end_s_local).T
    J_v_s = J_cms1[:3, :]; J_w_s = J_cms1[3:, :]
    for i in range(4):
        J_p_s_all[i, :, :] = J_v_s - _skew(r_s_world_cols[i, :]) @ J_w_s

    r_l_world_cols = (T_cms2_end[:3, :3] @ p_end_l_local).T
    J_v_l = J_total[:3,:]; J_w_l = J_total[3:,:]
    for i in range(4):
        J_p_l_all[i, :, :] = J_v_l - _skew(r_l_world_cols[i, :]) @ J_w_l

    # --- Calculate final actuation jacobian d(delta_l)/dq ---
    d_s = p_end_s_world - p_start_s
    d_l = p_end_l_world - p_start_l
    len_s = np.linalg.norm(d_s, axis=0)
    len_l = np.linalg.norm(d_l, axis=0)

    J_act = np.zeros((8, 6))
    for i in range(4):
        if len_s[i] > 1e-9:
            J_act[i, :] = - (d_s[:, i].T / len_s[i]) @ J_p_s_all[i, :, :]
    for i in range(4):
        if len_l[i] > 1e-9:
            J_act[i+4, :] = - (d_l[:, i].T / len_l[i]) @ J_p_l_all[i, :, :]

    # --- [NEW] Blend/Zero PSS columns based on anchor mode ---
    # The logic of zeroing out q_for_fk[0:2] for pss_end mode should
    # naturally result in the correct (zero) jacobian columns.
    # Manually zeroing them here was redundant and caused inconsistencies.
    # The same applies to the blend mode scaling.
    # if mode == 'pss_end':
    #     J_act[:, 0:2] = 0.0
    # elif mode == 'blend':
    #     J_act[:, 0:2] *= (1.0 - blend)

    return J_act


def calculate_actuation_gradient_analytical(q_6d, delta_l_motor, params):
    k_cable = params['Drive_Properties']['cable_stiffness']
    delta_l_robot = calculate_drive_mapping(q_6d, params)
    stretch = delta_l_motor - delta_l_robot
    stretch_tensioned = smooth_max_zero(stretch)
    deriv_smooth_max = smooth_max_zero_derivative(stretch)
    dU_cable_d_delta_l = -k_cable * stretch_tensioned * deriv_smooth_max
    # --- USE ANALYTICAL JACOBIAN ---
    J_act = calculate_actuation_jacobian_analytical(q_6d, params)
    grad_cable = J_act.T @ dU_cable_d_delta_l
    f_pre = params['Drive_Properties'].get('pretension_force_N', 0.0)
    grad_pretension = -f_pre * (J_act.T @ np.ones(8))
    return grad_cable + grad_pretension

def calculate_regularization_gradient(q_6d, params):
    regularization_strength = 1e-5
    return regularization_strength * q_6d

def calculate_gradient_disp_ctrl(q_6d, delta_l_motor, params):
    grad_e = calculate_elastic_gradient_analytical(q_6d, params)
    grad_g = calculate_gravity_gradient_analytical(q_6d, params)
    grad_a = calculate_actuation_gradient_analytical(q_6d, delta_l_motor, params)
    grad_r = calculate_regularization_gradient(q_6d, params)
    total_grad = grad_e + grad_g + grad_a + grad_r
    grad_norm = np.linalg.norm(total_grad)
    clipping_threshold = 1e3
    if grad_norm > clipping_threshold:
        total_grad = total_grad / grad_norm * clipping_threshold
    return total_grad

def expand_diff4_to_motor8(diff4, params):
    """
    [V3 CORRECTED FOR DIAGONAL DRIVE] Expands a 4D differential drive vector 
    into an 8D motor displacement vector. This version correctly handles the
    45-degree offset of the long lines using vector projection.
    """
    delta_l_motor = np.zeros(8)
    
    # --- Short lines (proximal bending) ---
    # Orthogonal layout (0, 90, 180, 270 deg) - This part is correct.
    delta_l_motor[0] =  diff4[0]  # +X direction
    delta_l_motor[1] =  diff4[1]  # +Y direction
    delta_l_motor[2] = -diff4[0]  # -X direction
    delta_l_motor[3] = -diff4[1]  # -Y direction
    
    # --- Long lines (distal bending) ---
    # Correctly decompose desired (dx, dy) motion onto diagonal cables.
    dx = diff4[2]  # Desired X-component motion
    dy = diff4[3]  # Desired Y-component motion
    
    # Get the physical angles from the config file to be robust
    long_line_angles = np.deg2rad(params['Geometry']['long_lines']['angles_deg'])

    # Project the (dx, dy) vector onto each cable's direction vector
    for i in range(4):
        angle = long_line_angles[i]
        # The amount to pull/release is the projection of (dx, dy) onto the cable's direction vector.
        # Note: Cable's direction vector is (cos(angle), sin(angle)).
        # Pull is positive, so we use this projection directly.
        delta_l_motor[i + 4] = dx * np.cos(angle) + dy * np.sin(angle)
    
    return delta_l_motor

if __name__ == '__main__':
    print("---\u5206\u5206\u5782\u76f4\u9a8c\u8bc1 (V-Final.Post-Cleanup) ---")
    params = load_config('config/config.json')
    np.set_printoptions(precision=8, suppress=True, linewidth=120)

    q_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    delta_l_motor_test = np.random.rand(8) * 0.001

    print(f"\n---\u6d4b\u8bd5\u6761\u4ef6---")
    print(f"  - \u6d4b\u8bd5\u6784\u578b q: {q_test}")
    print(f"  - \u7535\u673a\u4f4d\u79fb \u0394l_motor: {delta_l_motor_test}")

    print("\n---\u5206\u9879\u5782\u76f4\u9a8c\u8bc1---")
    eps = 1e-8
    
    def compare_gradients(name, analytical_grad, numerical_grad):
        diff_norm = np.linalg.norm(analytical_grad - numerical_grad)
        ref_norm = np.linalg.norm(numerical_grad)
        rel_error = diff_norm / (ref_norm + 1e-12)
        print(f"\n- {name} \u5782\u76f4:")
        print(f"  - \u89e3\u6790\u6cd5: {analytical_grad}")
        print(f"  - \u6570\u503c\u6cd5 (\u4e2d\u5fc3\u5dee\u5206): {numerical_grad}")
        print(f"  - \u76f8\u5bf9\u8bef\u5dee: {rel_error:.6e}")
        if rel_error < 5e-5: # Looser tolerance for actuation gradient
            print("  - ✅ \u901a\u8fc7")
            
            return True
        else:
            print(f"  - ❌ \u5931\u8d25 (\u5dee\u503c: {analytical_grad - numerical_grad})")
            return False

    # 1. \u5f39\u6027\u80fd\u91cf
    grad_e_ana = calculate_elastic_gradient_analytical(q_test, params)
    grad_e_num = approx_fprime(q_test, lambda q: calculate_elastic_potential_energy(q, params), eps)
    test1 = compare_gradients("\u5f39\u6027\u80fd\u91cf (Elastic)", grad_e_ana, grad_e_num)

    # 2. \u91cd\u529b\u52bf\u80fd
    grad_g_ana = calculate_gravity_gradient_analytical(q_test, params)
    grad_g_num = approx_fprime(q_test, lambda q: calculate_gravity_potential_energy(q, params), eps)
    test2 = compare_gradients("\u91cd\u529b\u52bf\u80fd (Gravity)", grad_g_ana, grad_g_num)

    # 3. \u9a71\u52a8\u76f8\u5173\u80fd\u91cf (U_cable + U_pretension)
    grad_a_ana = calculate_actuation_gradient_analytical(q_test, delta_l_motor_test, params)
    def actuation_energy_func(q):
        k_cable = params['Drive_Properties']['cable_stiffness']
        f_pre = params['Drive_Properties'].get('pretension_force_N', 0.0)
        delta_l_robot = calculate_drive_mapping(q, params)
        stretch = delta_l_motor_test - delta_l_robot
        U_cable = 0.5 * k_cable * np.sum(smooth_max_zero(stretch)**2)
        U_pretension = -f_pre * np.sum(delta_l_robot)
        return U_cable + U_pretension
    grad_a_num = approx_fprime(q_test, actuation_energy_func, eps)
    test3 = compare_gradients("\u9a71\u52a8\u80fd\u91cf (Actuation)", grad_a_ana, grad_a_num)

    # 4. \u603b\u5782\u76f4
    print("\n---\u603b\u5782\u76f4\u9a8c\u8bc1---")
    total_grad_ana = calculate_gradient_disp_ctrl(q_test, delta_l_motor_test, params)
    total_grad_num = approx_fprime(q_test, lambda q: calculate_total_potential_energy_disp_ctrl(q, delta_l_motor_test, params), eps)
    test4 = compare_gradients("\u603b\u548c (Total)", total_grad_ana, total_grad_num)

    print("\n---\u81ea\u68c0\u5b8c\u6210---")
    if all([test1, test2, test3, test4]):
        print("✅✅✅ \u6574\u4f53\u5782\u76f4\u9a8c\u8bc1\u901a\u8fc7\uff01\u9759\u529b\u5b66\u6a21\u5757\u5730\u57fa\u7a33\u56fa\uff01")
    else:
        print("❌ \u6574\u4f53\u5782\u76f4\u9a8c\u8bc1\u5931\u8d25\uff0c\u8bf7\u68c0\u67e5\u5206\u9879\u9519\u8bef\u3002")

# =====================================================================
# === \u4e8c\u9636\u533f\u540d\uff1a\u6d77\u6851\u7279\u77e9\u90f4 (Hessian) - [V-Final.HighPerformance]
# =====================================================================

def calculate_elastic_hessian_analytical(q_6d, params):
    """
    [NEW] Calculates the analytical Hessian of the elastic potential energy for the 6D model.
    """
    stiff = params['Stiffness']
    geo = params['Geometry']
    
    Lp = geo['PSS_initial_length']
    Lc1 = geo['CMS_proximal_length']
    Lc2 = geo['CMS_distal_length']
    
    K_bending_pss = stiff['pss_total_equivalent_bending_stiffness']
    K_bending_cms = stiff['cms_bending_stiffness']
    
    H_E = np.zeros((6, 6))
    
    # The Hessian is a diagonal matrix because the energy terms are decoupled squares of kappa.
    H_E[0, 0] = K_bending_pss * Lp
    H_E[2, 2] = K_bending_cms * Lc1
    H_E[4, 4] = K_bending_cms * Lc2
    
    return H_E

def calculate_regularization_hessian_analytical(q_6d, params):
    """
    [NEW] Calculates the analytical Hessian of the regularization term.
    """
    regularization_strength = 1e-5
    return np.eye(6) * regularization_strength

def calculate_hessian_disp_ctrl_high_performance(q_6d, delta_l_motor, params):
    """
    [IMPROVED] High-performance analytical approximation for the total Hessian.
    Implements the J.T @ M @ J approximation for the gravity Hessian.
    """
    # 1. Analytical Elastic Hessian
    H_E = calculate_elastic_hessian_analytical(q_6d, params)
    
    # 2. Gauss-Newton approximation for Cable Hessian
    k_cable = params['Drive_Properties']['cable_stiffness']
    J_act = calculate_actuation_jacobian_numerical(q_6d, params)
    H_C_approx = k_cable * (J_act.T @ J_act)
    
    # 3. Analytical Regularization Hessian
    H_reg = calculate_regularization_hessian_analytical(q_6d, params)

    # 4. [IMPROVED] Approximate Gravity Hessian H_G ≈ sum(m_i * J_com_i.T @ J_com_i)
    masses = [params['Mass']['pss_kg'], params['Mass']['cms_proximal_kg'], params['Mass']['cms_distal_kg']]
    J_com_dict = calculate_com_jacobians_analytical(q_6d, params)
    J_coms = [J_com_dict['pss'], J_com_dict['cms1'], J_com_dict['cms2']]
    
    H_G = np.zeros((6, 6))
    for mass, J_com in zip(masses, J_coms):
        H_G += mass * (J_com.T @ J_com) # Note: This is a heuristic approximation
    
    # 5. Sum the components
    H_total_approx = H_E + H_C_approx + H_reg + H_G
    
    return H_total_approx




