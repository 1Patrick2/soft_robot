"""
本模块负责物理问题的“无量纲化”。

无量纲化是一种标准的工程技术，用于解决因参数尺度差异巨大而导致的数值不稳定问题。
通过将所有物理量（SI单位）转换为围绕1.0的、无单位的相对值，我们为求解器
创造了一个数值上更健康、更易于计算的优化问题。

本模块是“物理世界”和“数学世界”之间的“翻译官”：
1. 物理世界：使用标准的SI单位（米、牛顿等）。所有物理模型模块（kinematics.py, statics.py）
   都只在这个世界中运行。
2. 数学世界：使用无量纲的数值。Scipy优化器只在这个世界中运行。
"""
import numpy as np

def get_characteristic_scales(params):
    """
    根据给定的物理参数（SI单位），计算并返回一组特征尺度。
    """
    geo = params['Geometry']
    stiff = params['Stiffness']

    L_char = (
              geo['PSS_initial_length'] + 
              geo['CMS_proximal_length'] + 
              geo['CMS_distal_length']
    )
    K_char = stiff['pss_total_equivalent_bending_stiffness']
    F_char = K_char / (L_char**2)
    U_char = F_char * L_char

    scales = {
        'L_char': L_char,
        'K_char': K_char,
        'F_char': F_char,
        'U_char': U_char
    }
    return scales

def q_to_nondimensional(q, scales):
    """将6D物理构型向量q (SI单位) 转换为无量纲构型向量hat_q."""
    L_char = scales['L_char']
    hat_q = np.copy(q).astype(float)
    # q = [kp, phip, kc1, phic1, kc2, phic2]
    # 物理曲率 kappa (1/m) -> 无量纲 hat_kappa
    hat_q[0::2] *= L_char   
    # 角度 phi (q[1::2]) 本身就是无量纲的，无需转换
    return hat_q

def q_from_nondimensional(hat_q, scales):
    """将6D无量纲构型向量hat_q转换回物理构型向量q (SI单位)."""
    L_char = scales['L_char']
    q = np.copy(hat_q).astype(float)
    # 无量纲 hat_kappa -> 物理曲率 kappa (1/m)
    q[0::2] /= L_char   
    return q

def tau_to_nondimensional(tensions, scales):
    """将物理驱动力字典 (SI单位) 转换为无量纲字典."""
    F_char = scales['F_char']
    hat_tensions = {
        'tensions_short': np.array(tensions['tensions_short']) / F_char,
        'tensions_long': np.array(tensions['tensions_long']) / F_char,
    }
    return hat_tensions

def tau_from_nondimensional(hat_tensions, scales):
    """将无量纲驱动力字典转换回物理字典 (SI单位)."""
    F_char = scales['F_char']
    tensions = {
        'tensions_short': np.array(hat_tensions['tensions_short']) * F_char,
        'tensions_long': np.array(hat_tensions['tensions_long']) * F_char,
    }
    return tensions

def gradient_to_nondimensional(grad_phys, scales):
    """
    将6D物理梯度(SI单位)转换为无量纲梯度。
    """
    L_char = scales['L_char']
    U_char = scales['U_char']
    
    hat_grad = np.copy(grad_phys).astype(float)
    
    # dU/d(kappa) (phys) -> d(hat_U)/d(hat_kappa)
    hat_grad[0::2] = grad_phys[0::2] / L_char / U_char

    # dU/d(phi) (phys) -> d(hat_U)/d(hat_phi)
    hat_grad[1::2] = grad_phys[1::2] / U_char
    
    return hat_grad

if __name__ == '__main__':
    print("--- 无量纲化模块功能自检 (6D模型) ---")

    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    from src.utils.read_config import load_config

    config_params = load_config(os.path.join(project_root, 'config', 'config.json'))

    char_scales = get_characteristic_scales(config_params)
    print("\n成功加载参数并计算特征尺度:")
    for key, value in char_scales.items():
        print(f"  - {key}: {value:.6f}")

    print("\n验证转换函数:")
    q_physical = np.array([10.0, 1.57, 5.0, 3.14, 2.0, 0.0])
    hat_q_test = q_to_nondimensional(q_physical, char_scales)
    q_reverted = q_from_nondimensional(hat_q_test, char_scales)
    assert np.allclose(q_physical, q_reverted), "q转换可逆性测试失败"
    print("  - q 转换可逆性测试: 通过")

    tau_physical = {'tensions_short': np.array([1.0, 2.0, 3.0, 4.0]), 'tensions_long': np.array([5.0, 6.0, 7.0, 8.0])}
    hat_tau_test = tau_to_nondimensional(tau_physical, char_scales)
    tau_reverted = tau_from_nondimensional(hat_tau_test, char_scales)
    assert np.allclose(tau_physical['tensions_short'], tau_reverted['tensions_short']), "tau转换可逆性测试失败"
    assert np.allclose(tau_physical['tensions_long'], tau_reverted['tensions_long']), "tau转换可逆性测试失败"
    print("  - tau 转换可逆性测试: 通过")

    print("\n验证梯度转换函数:")
    grad_physical = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    hat_grad_test = gradient_to_nondimensional(grad_physical, char_scales)
    
    expected_hat_grad = np.copy(grad_physical)
    L_char = char_scales['L_char']
    U_char = char_scales['U_char']
    expected_hat_grad[0::2] = grad_physical[0::2] / L_char / U_char
    expected_hat_grad[1::2] = grad_physical[1::2] / U_char
    
    assert np.allclose(hat_grad_test, expected_hat_grad), "gradient_to_nondimensional 计算错误"
    print("  - 梯度转换函数测试: 通过")

    print("\n--- 全部自检通过 ---")