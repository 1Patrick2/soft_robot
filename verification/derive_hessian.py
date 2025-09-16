import sympy as sp

def derive_hessian():
    """使用Sympy推导重力势能的海森矩阵 H_G"""
    # 1. 定义符号变量
    g = sp.Symbol('g')
    m_pss, m_c1, m_c2 = sp.symbols('m_pss m_c1 m_c2')
    
    # 定义7个构型变量q
    q = [sp.Symbol(f'q{i}') for i in range(7)]
    Lp, kp, phip, kc1, phic1, kc2, phic2 = q

    # 定义各段质心的世界坐标 (z分量)
    # 注意：这里我们只需要z分量，因为重力只在z方向
    # 并且，这里的函数是简化的，只为了推导，实际的函数在kinematics.py中
    com_z_pss = sp.Function('com_z_pss')(Lp, kp, phip)
    com_z_c1 = sp.Function('com_z_c1')(Lp, kp, phip, kc1, phic1)
    com_z_c2 = sp.Function('com_z_c2')(Lp, kp, phip, kc1, phic1, kc2, phic2)

    # 2. 定义重力势能 U_g
    U_g = g * (m_pss * com_z_pss + m_c1 * com_z_c1 + m_c2 * com_z_c2)

    # 3. 计算梯度 grad_g (一阶偏导数)
    grad_g = [sp.diff(U_g, var) for var in q]

    # 4. 计算海森矩阵 H_g (二阶偏导数)
    H_g = sp.Matrix([[sp.diff(g_i, var) for var in q] for g_i in grad_g])

    # 5. 打印结果
    print("--- Symbolic Gravity Hessian H_g ---")
    # 打印一个元素作为示例，因为完整矩阵太大
    print("H_g[0,0] = d^2(U_g)/dLp^2 = ", H_g[0,0])
    print("\n")
    print("H_g[0,1] = d^2(U_g)/dLp d(kp) = ", H_g[0,1])
    print("\n")
    print("由于表达式过于复杂，且依赖于未定义的函数，")
    print("此脚本的主要目的是展示推导的逻辑，实际的数值计算需要代入具体的质心雅可比函数。")
    print("真正的修复需要发生在 statics.py 中，确保 H_G 的计算正确调用了 com_jacobians 的导数。")

if __name__ == "__main__":
    derive_hessian()
