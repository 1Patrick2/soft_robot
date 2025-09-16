import numpy as np
import sys
import os

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kinematics import forward_kinematics
from src.utils.read_config import load_config

def main():
    """
    对正运动学函数 forward_kinematics 进行基础性的单元测试。
    使用一个简单的、phi=0的构型，对比程序输出和手动计算的理论真值。
    """
    print("--- 正运动学 (forward_kinematics) 基础单元测试 ---")

    # 1. 加载单段机器人配置
    config = load_config('config/config.json')
    if not (config['Geometry']['CMS_proximal_length'] == 0.0 and config['Geometry']['CMS_distal_length'] == 0.0):
        print("❌ [错误] 此验证脚本应在单段机器人配置下运行。\n   请先将 config.json 修改为单段配置。")
        return
    print("成功加载单段机器人配置。")

    # 2. 定义一个简单的测试构型 q (在X-Z平面弯曲)
    L = 0.1
    kappa = 1.0
    phi = 0.0
    q_test = np.array([L, kappa, phi, 0.0, 0.0, 0.0, 0.0])
    print(f"\n测试构型 q: L={L}, kappa={kappa}, phi={phi}")

    # 3. 手动计算理论真值矩阵 (Expected Matrix)
    print("\n正在手动计算理论真值矩阵 T_expected...")
    c_phi = 1  # cos(0)
    s_phi = 0  # sin(0)
    kL = kappa * L
    c_kL = np.cos(kL)
    s_kL = np.sin(kL)

    T_expected = np.array([
        [c_phi**2 * (c_kL - 1) + 1,  s_phi * c_phi * (c_kL - 1), c_phi * s_kL, c_phi * (1 - c_kL) / kappa],
        [s_phi * c_phi * (c_kL - 1),  s_phi**2 * (c_kL - 1) + 1, s_phi * s_kL, s_phi * (1 - c_kL) / kappa],
        [-c_phi * s_kL,             -s_phi * s_kL,             c_kL,        s_kL / kappa],
        [0,                         0,                         0,           1]
    ])

    # 4. 调用函数计算实际值
    print("正在调用 forward_kinematics 函数计算实际矩阵 T_actual...")
    T_actual, _ = forward_kinematics(q_test, config)

    # 5. 对比结果
    print("\n--- 结果对比 ---")
    np.set_printoptions(precision=8, suppress=True)

    print(f"\n理论真值矩阵 T_expected:")
    print(T_expected)
    print(f"\n函数计算矩阵 T_actual:")
    print(T_actual)

    diff_norm = np.linalg.norm(T_actual - T_expected)
    print(f"\n差值矩阵范数 (Norm of Difference): {diff_norm:.10f}")

    # 最终结论
    print("\n--- 最终结论 ---")
    if np.allclose(T_expected, T_actual, atol=1e-8):
        print("✅ [通过] 函数 `forward_kinematics` 与理论真值完全一致。")
    else:
        print("❌ [失败] 正运动学函数 `forward_kinematics` 与理论真值存在差异！已找到BUG！")

if __name__ == "__main__":
    main()