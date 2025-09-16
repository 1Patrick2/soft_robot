# verification/verify_forward_kinematics.py

import numpy as np
import sys
import os

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kinematics import forward_kinematics, pcc_transformation
from src.utils.read_config import load_config

def verify_fk():
    """Verifies the forward kinematics against a simple, known case."""
    print("--- 正运动学模块真值验证 ---")
    config = load_config('config/config.json')
    np.set_printoptions(precision=6, suppress=True)

    # --- 测试案例：第一段 (PSS) 弯曲90度 ---
    Lp = config['Geometry']['PSS_initial_length']
    kappa_p = (np.pi / 2) / Lp
    # 确认使用6D模型
    q_test = np.array([kappa_p, 0, 0, 0, 0, 0])

    print(f"测试构型 q: {q_test}")
    print(f"(PSS段长度 L_p = {Lp}m, 期望弯曲90度)")

    # --- 理论真值计算 ---
    c_phi, s_phi = np.cos(0), np.sin(0)
    c_kL, s_kL = np.cos(np.pi/2), np.sin(np.pi/2)
    T_pss_true = np.array([
        [c_phi**2 * (c_kL - 1) + 1, s_phi * c_phi * (c_kL - 1),  c_phi * s_kL, c_phi * (1 - c_kL) / kappa_p],
        [s_phi * c_phi * (c_kL - 1), s_phi**2 * (c_kL - 1) + 1,  s_phi * s_kL, s_phi * (1 - c_kL) / kappa_p],
        [-c_phi * s_kL,             -s_phi * s_kL,              c_kL,          s_kL / kappa_p],
        [0,                         0,                          0,             1]
    ])
    Lc1 = config['Geometry']['CMS_proximal_length']
    Lc2 = config['Geometry']['CMS_distal_length']
    T_cms1_true = np.identity(4); T_cms1_true[2, 3] = Lc1
    T_cms2_true = np.identity(4); T_cms2_true[2, 3] = Lc2
    T_final_true = T_pss_true @ T_cms1_true @ T_cms2_true
    pos_true = T_final_true[:3, 3]

    print(f"\n理论末端位置 (x, y, z): ({pos_true[0]:.6f}, {pos_true[1]:.6f}, {pos_true[2]:.6f}) m")

    # --- 代码计算值 (带调试打印) ---
    T_pss_code = pcc_transformation(kappa_p, 0.0, Lp)
    print("\n--- 调试信息 ---")
    print(f"kappa_p: {kappa_p}, Lp: {Lp}, kappa_p*Lp: {kappa_p*Lp}")
    print("T_pss (from code):\n", np.round(T_pss_code, 8))

    T_code, _ = forward_kinematics(q_test, config)
    pos_code = T_code[:3, 3]
    print(f"代码计算出的末端位置: {pos_code}")

    # --- 对比 ---
    error = np.linalg.norm(pos_code - pos_true)
    print(f"\n位置误差: {error * 1000:.4f} mm")

    if error < 1e-6:
        print("✅✅✅ 【验证通过】正运动学核心函数计算正确！")
    else:
        print("❌❌❌ 【验证失败】正运动学核心函数存在Bug！")

if __name__ == '__main__':
    verify_fk()