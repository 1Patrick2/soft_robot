import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.debug_config import DEBUG_SINGLE_SEGMENT

EPSILON = 1e-9


def adjoint_transformation_matrix(T):
    """
    计算齐次变换矩阵 T (4x4) 的伴随矩阵 (6x6)
    Ad_T = [[R, skew(p)R],
            [0,      R   ]]
    """
    R = T[:3, :3]
    p = T[:3, 3]

    def skew(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[:3, 3:] = skew(p) @ R
    return Ad


def pcc_transformation(kappa, phi, L):
    """
    [V-Final.Genesis & CS-Friendly V2] 使用100%正确的、经过外部验证的PCC变换矩阵公式。
    修复了isclose对复数处理不当的问题,确保虚部能被正确传递。
    """
    # 检查输入是否为复数，以确定输出矩阵的类型
    output_dtype = np.complex128 if np.iscomplexobj(kappa) or np.iscomplexobj(phi) else np.float64

    # [FIX] np.isclose(complex, 0) is problematic. A better way to handle the singularity.
    if abs(kappa) < EPSILON:
        # 如果没有弯曲，则是纯粹的Z轴平移
        T = np.identity(4, dtype=output_dtype)
        T[2, 3] = L
    else:
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_kL, s_kL = np.cos(kappa * L), np.sin(kappa * L)

        T = np.array([
            [c_phi**2 * (c_kL - 1) + 1, s_phi * c_phi * (c_kL - 1),  c_phi * s_kL, c_phi * (1 - c_kL) / kappa],
            [s_phi * c_phi * (c_kL - 1), s_phi**2 * (c_kL - 1) + 1,  s_phi * s_kL, s_phi * (1 - c_kL) / kappa],
            [-c_phi * s_kL,             -s_phi * s_kL,              c_kL,          s_kL / kappa],
            [0,                         0,                          0,             1]
        ], dtype=output_dtype)

    return T

def get_segment_com_local(kappa, L):
    if np.isclose(L, 0):
        return np.array([0.0, 0.0, 0.0])

    theta = kappa * L
    if np.isclose(kappa, 0):
        # 直线极限
        return np.array([0.0, 0.0, L / 2.0])
    else:
        x_com = np.sin(theta) / kappa
        z_com = (1 - np.cos(theta)) / kappa
        return np.array([x_com, 0.0, z_com])

def forward_kinematics(q, robot_params, return_all_transforms=False):
    kappa_p, phi_p, kappa_c1, phi_c1, kappa_c2, phi_c2 = q
    L_p = robot_params['Geometry']['PSS_initial_length']
    L_c1 = robot_params['Geometry']['CMS_proximal_length']
    L_c2 = robot_params['Geometry']['CMS_distal_length']

    T_pss = pcc_transformation(kappa_p, phi_p, L_p)
    T_cms1 = pcc_transformation(kappa_c1, phi_c1, L_c1)
    T_cms2 = pcc_transformation(kappa_c2, phi_c2, L_c2)

    T_base_cms1 = T_pss
    T_base_cms2 = T_base_cms1 @ T_cms1
    T_final = T_base_cms2 @ T_cms2

    com_pss_local = get_segment_com_local(kappa_p, L_p)
    com_cms1_local = get_segment_com_local(kappa_c1, L_c1)
    com_cms2_local = get_segment_com_local(kappa_c2, L_c2)

    # [BUGFIX] Correct CoM world position calculation
    # The CoM of a segment is its local CoM vector transformed by the segment's BASE transformation matrix.
    com_pss_world = com_pss_local # PSS base is Identity
    com_cms1_world = T_pss[:3, :3] @ com_cms1_local + T_pss[:3, 3] # CMS1 base is T_pss
    com_cms2_world = T_base_cms2[:3, :3] @ com_cms2_local + T_base_cms2[:3, 3] # CMS2 base is T_base_cms2

    com_positions = {
        'pss': com_pss_world,
        'cms1': com_cms1_world,
        'cms2': com_cms2_world
    }

    if return_all_transforms:
        return T_final, com_positions, T_pss, T_base_cms2
    else:
        return T_final, com_positions

def pcc_derivatives_analytical(kappa, phi, L):
    dT_dL = np.zeros((4, 4))
    dT_dkappa = np.zeros((4, 4))
    dT_dphi = np.zeros((4, 4))
    c_phi, s_phi = np.cos(phi), np.sin(phi)

    if np.isclose(kappa, 0):
        dT_dL[2, 3] = 1
        dT_dkappa[0, 2] = L * c_phi
        dT_dkappa[1, 2] = L * s_phi
        dT_dkappa[2, 0] = -L * c_phi
        dT_dkappa[2, 1] = -L * s_phi
        dT_dphi[0, 1] = -L
        dT_dphi[1, 0] = L
    else:
        kappa_safe = kappa + EPSILON
        c_kL, s_kL = np.cos(kappa * L), np.sin(kappa * L)
        dT_dL[0, 0] = c_phi**2 * (-kappa * s_kL); dT_dL[0, 1] = s_phi * c_phi * (-kappa * s_kL); dT_dL[0, 2] = c_phi * kappa * c_kL; dT_dL[0, 3] = c_phi * s_kL
        dT_dL[1, 0] = dT_dL[0, 1]; dT_dL[1, 1] = s_phi**2 * (-kappa * s_kL); dT_dL[1, 2] = s_phi * kappa * c_kL; dT_dL[1, 3] = s_phi * s_kL
        dT_dL[2, 0] = -c_phi * kappa * c_kL; dT_dL[2, 1] = -s_phi * kappa * c_kL; dT_dL[2, 2] = -kappa * s_kL; dT_dL[2, 3] = c_kL
        term1 = (L * s_kL * kappa - (1 - c_kL)) / (kappa_safe**2)
        dT_dkappa[0, 0] = c_phi**2 * (-L * s_kL); dT_dkappa[0, 1] = s_phi * c_phi * (-L * s_kL); dT_dkappa[0, 2] = c_phi * L * c_kL; dT_dkappa[0, 3] = c_phi * term1
        dT_dkappa[1, 0] = dT_dkappa[0, 1]; dT_dkappa[1, 1] = s_phi**2 * (-L * s_kL); dT_dkappa[1, 2] = s_phi * L * c_kL; dT_dkappa[1, 3] = s_phi * term1
        dT_dkappa[2, 0] = -c_phi * L * c_kL; dT_dkappa[2, 1] = -s_phi * L * c_kL; dT_dkappa[2, 2] = -L * s_kL; dT_dkappa[2, 3] = (c_kL * kappa * L - s_kL) / (kappa_safe**2)
        term2 = c_kL - 1
        dT_dphi[0, 0] = 2 * c_phi * -s_phi * term2; dT_dphi[0, 1] = (c_phi**2 - s_phi**2) * term2; dT_dphi[0, 2] = -s_phi * s_kL; dT_dphi[0, 3] = -s_phi * (1 - c_kL) / kappa_safe
        dT_dphi[1, 0] = dT_dphi[0, 1]; dT_dphi[1, 1] = 2 * s_phi * c_phi * term2; dT_dphi[1, 2] = c_phi * s_kL; dT_dphi[1, 3] = c_phi * (1 - c_kL) / kappa_safe
        dT_dphi[2, 0] = s_phi * s_kL; dT_dphi[2, 1] = -c_phi * s_kL
    return dT_dL, dT_dkappa, dT_dphi

def calculate_kinematic_jacobian_analytical(q, params):
    """ [V6.0 UPGRADED] Returns total, cms1, and pss end-frame jacobians. """
    J_total = np.zeros((6, 6))
    J_cms1 = np.zeros((6, 6))
    J_pss = np.zeros((6, 6))

    kappa_p, phi_p, kappa_c1, phi_c1, kappa_c2, phi_c2 = q
    L_p = params['Geometry']['PSS_initial_length']; L_c1 = params['Geometry']['CMS_proximal_length']; L_c2 = params['Geometry']['CMS_distal_length']
    
    # --- Transformations ---
    T_pss = pcc_transformation(kappa_p, phi_p, L_p)
    T_cms1 = pcc_transformation(kappa_c1, phi_c1, L_c1)
    T_cms2 = pcc_transformation(kappa_c2, phi_c2, L_c2)
    
    T_cms1_end = T_pss @ T_cms1
    T_total_end = T_cms1_end @ T_cms2

    # --- Derivatives of T ---
    _, dT_pss_dkp, dT_pss_dphip = pcc_derivatives_analytical(kappa_p, phi_p, L_p)
    _, dT_cms1_dkc1, dT_cms1_dphic1 = pcc_derivatives_analytical(kappa_c1, phi_c1, L_c1)
    _, dT_cms2_dkc2, dT_cms2_dphic2 = pcc_derivatives_analytical(kappa_c2, phi_c2, L_c2)

    # --- Helper to get twist vector from dT ---
    def get_twist(dT, T_frame):
        R_frame = T_frame[:3, :3]
        dp = dT[:3, 3]
        dR = dT[:3, :3]
        S = dR @ R_frame.T
        omega = np.array([S[2, 1], S[0, 2], S[1, 0]])
        return np.concatenate([dp, omega])

    # --- Populate J_pss ---
    J_pss[:, 0] = get_twist(dT_pss_dkp, T_pss)
    J_pss[:, 1] = get_twist(dT_pss_dphip, T_pss)

    # --- Populate J_cms1 ---
    J_cms1[:, 0] = get_twist(dT_pss_dkp @ T_cms1, T_cms1_end)
    J_cms1[:, 1] = get_twist(dT_pss_dphip @ T_cms1, T_cms1_end)
    J_cms1[:, 2] = get_twist(T_pss @ dT_cms1_dkc1, T_cms1_end)
    J_cms1[:, 3] = get_twist(T_pss @ dT_cms1_dphic1, T_cms1_end)

    # --- Populate J_total ---
    J_total[:, 0] = get_twist(dT_pss_dkp @ T_cms1 @ T_cms2, T_total_end)
    J_total[:, 1] = get_twist(dT_pss_dphip @ T_cms1 @ T_cms2, T_total_end)
    J_total[:, 2] = get_twist(T_pss @ dT_cms1_dkc1 @ T_cms2, T_total_end)
    J_total[:, 3] = get_twist(T_pss @ dT_cms1_dphic1 @ T_cms2, T_total_end)
    J_total[:, 4] = get_twist(T_pss @ T_cms1 @ dT_cms2_dkc2, T_total_end)
    J_total[:, 5] = get_twist(T_pss @ T_cms1 @ dT_cms2_dphic2, T_total_end)
    
    return J_total, J_cms1, J_pss

def pcc_com_derivatives(q_segment, var_index):
    L, kappa, phi = q_segment[0], q_segment[1], q_segment[2]
    if np.isclose(L, 0): return np.array([0, 0, 0])
    if np.isclose(kappa, 0):
        if var_index == 0: return np.array([0, 0, 0.5])
        elif var_index == 1: return np.array([0, 0, 0]) # [Bug] Was L^2/6, 0, 0 which is CoM not dCoM/dKappa
        else: return np.array([0, 0, 0])
    kL = kappa * L; sin_kL = np.sin(kL); cos_kL = np.cos(kL)
    if var_index == 0:
        d_com_x_local_dL = (-L * kappa * cos_kL + sin_kL) / (L**2 * kappa**2 + EPSILON)
        d_com_z_local_dL = (L * kappa * sin_kL + cos_kL - 1) / (L**2 * kappa**2 + EPSILON)
        return np.array([d_com_x_local_dL, 0, d_com_z_local_dL])
    elif var_index == 1:
        d_com_x_local_dk = (-L * kappa * cos_kL - L * kappa + 2 * sin_kL) / (L * kappa**3 + EPSILON)
        d_com_z_local_dk = (L * kappa * sin_kL + 2 * cos_kL - 2) / (L * kappa**3 + EPSILON)
        return np.array([d_com_x_local_dk, 0, d_com_z_local_dk])
    else: return np.array([0,0,0])

def calculate_com_jacobians_analytical(q, params, epsilon=1e-7):
    """
    [V-Final.GoldStandard] Calculates the Center of Mass Jacobian using a high-precision
    central difference numerical method, as per test.md instructions.
    The name `_analytical` is kept for compatibility with calling modules.
    """
    J_com_pss = np.zeros((3, 6))
    J_com_cms1 = np.zeros((3, 6))
    J_com_cms2 = np.zeros((3, 6))

    for i in range(6):
        h = epsilon * max(1.0, abs(q[i]))
        q_plus = q.copy(); q_plus[i] += h
        q_minus = q.copy(); q_minus[i] -= h

        _, com_plus = forward_kinematics(q_plus, params)
        _, com_minus = forward_kinematics(q_minus, params)

        J_com_pss[:, i] = (com_plus['pss'] - com_minus['pss']) / (2 * h)
        J_com_cms1[:, i] = (com_plus['cms1'] - com_minus['cms1']) / (2 * h)
        J_com_cms2[:, i] = (com_plus['cms2'] - com_minus['cms2']) / (2 * h)

    return {'pss': J_com_pss, 'cms1': J_com_cms1, 'cms2': J_com_cms2}


def calculate_kinematic_jacobian_numerical(q, params, epsilon=1e-7):
    """
    [新增] 使用有限差分法计算运动学雅可比矩阵 (6x6 Twist)
    [Optimized] Now uses relative step size for robustness.
    """
    J_num = np.zeros((6, 6))
    T_base, _ = forward_kinematics(q, params)

    for i in range(6):
        h = 1e-6 * max(1.0, abs(q[i]))
        q_plus = q.copy()
        q_plus[i] += h
        T_plus, _ = forward_kinematics(q_plus, params)

        q_minus = q.copy()
        q_minus[i] -= h
        T_minus, _ = forward_kinematics(q_minus, params)

        # 使用中心差分计算dT/dq_i
        dT_dqi = (T_plus - T_minus) / (2 * h)
        
        dR_dqi = dT_dqi[:3, :3]
        dp_dqi = dT_dqi[:3, 3]

        # 从 dR/dq_i @ R.T 提取角速度向量
        S_omega = dR_dqi @ T_base[:3, :3].T
        omega = np.array([S_omega[2, 1], S_omega[0, 2], S_omega[1, 0]])
        
        J_num[:, i] = np.concatenate([dp_dqi, omega])

    return J_num


if __name__ == '__main__':
    print("--- 运动学模块功能自检 (6D模型) ---")
    from src.utils.read_config import load_config
    config_params = load_config('config/config.json')
    print("成功加载 'config.json' 中的参数.\n")
    np.set_printoptions(precision=6, suppress=True)
    q_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) # Test away from zero
    print(f"--- 测试构型 q (6D) ---\n{q_test}")
    T_final, _ = forward_kinematics(q_test, config_params)
    print(f"\n--- 正运动学计算结果 (末端位姿 T) ---")
    print(T_final)
    print("\n--- 运动学雅可比对比测试 ---")
    J_ana, _, _ = calculate_kinematic_jacobian_analytical(q_test, config_params)
    J_num = calculate_kinematic_jacobian_numerical(q_test, config_params)
    diff_norm = np.linalg.norm(J_ana - J_num)
    ref_norm = np.linalg.norm(J_num)
    rel_error = diff_norm / (ref_norm + 1e-12)
    print(f"解析法雅可比 J_analytical (6x6):\n{J_ana}")
    print(f"数值法雅可比 J_numerical (6x6):\n{J_num}")
    print(f"差值范数 (绝对): {diff_norm:.8f}")
    print(f"相对误差: {rel_error:.8f}")
    if rel_error < 1e-4:
        print("【测试通过】: 解析法与数值法计算的运动学雅可比高度一致 (相对误差 < 1e-4)。\n")
    else:
        print("【测试失败】: 雅可比计算存在显著差异。\n")
    print("--- 自检完成 ---")
