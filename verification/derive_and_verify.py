# verification/derive_com_derivatives.py

import sympy as sp
import numpy as np

def generate_com_derivatives():
    """
    使用 SymPy 符号计算库来推导单段PCC机器人质心的偏导数。
    这个脚本的目标是生成可以被直接复制到 kinematics.py 中的、100%正确的Python代码。
    """
    print("--- 使用 SymPy 开始推导质心偏导数 ---")

    # 1. 定义符号变量
    L, k = sp.symbols('L kappa', real=True, positive=True)

    # 2. 定义基于 kappa**2 的、正确的质心x和z分量的公式
    print("\n[Step 1] 定义正确的质心公式 (分母为 kappa**2):")
    com_x_expr = (k*L - sp.sin(k*L)) / (L * k**2)
    com_z_expr = (1 - sp.cos(k*L)) / (L * k**2)
    
    print(f"com_x = {com_x_expr}")
    print(f"com_z = {com_z_expr}")

    # 3. 计算对 L 的偏导数
    print("\n[Step 2] 计算对长度 L 的偏导数 (d/dL):")
    d_com_x_dL = sp.diff(com_x_expr, L)
    d_com_z_dL = sp.diff(com_z_expr, L)

    # 4. 计算对 kappa 的偏导数
    print("\n[Step 3] 计算对曲率 kappa 的偏导数 (d/d_kappa):")
    d_com_x_dk = sp.diff(com_x_expr, k)
    d_com_z_dk = sp.diff(com_z_expr, k)

    # 5. 打印结果，并格式化为可以直接使用的Python代码
    print("\n" + "="*50)
    print("  最终结果：请将以下代码块复制并替换到")
    print("  src/kinematics.py -> pcc_com_derivatives 函数中")
    print("="*50 + "\n")

    # 使用 sympy.printing.pycode 模块来生成更简洁的Python代码
    # 并进行一些手动替换以匹配我们的变量名
    
    # ------------------- d/dL 分支 -------------------
    print("    if var_index == 0: # d/dL")
    # 替换 sympy 的变量名和函数名为 numpy 的
    dx_dl_code = sp.pycode(sp.simplify(d_com_x_dL)).replace('k', 'kappa').replace('sin', 'np.sin').replace('cos', 'np.cos')
    dz_dl_code = sp.pycode(sp.simplify(d_com_z_dL)).replace('k', 'kappa').replace('sin', 'np.sin').replace('cos', 'np.cos')
    print(f"        d_com_x_local_dL = {dx_dl_code}")
    print(f"        d_com_z_local_dL = {dz_dl_code}")
    print("        return np.array([d_com_x_local_dL, 0, d_com_z_local_dL])")
    
    print("")

    # ------------------- d/d_kappa 分支 -------------------
    print("    elif var_index == 1: # d/d_kappa")
    dx_dk_code = sp.pycode(sp.simplify(d_com_x_dk)).replace('k', 'kappa').replace('sin', 'np.sin').replace('cos', 'np.cos')
    dz_dk_code = sp.pycode(sp.simplify(d_com_z_dk)).replace('k', 'kappa').replace('sin', 'np.sin').replace('cos', 'np.cos')
    print(f"        d_com_x_local_dk = {dx_dk_code}")
    print(f"        d_com_z_local_dk = {dz_dk_code}")
    print("        return np.array([d_com_x_local_dk, 0, d_com_z_local_dk])")
    
    print("\n" + "="*50)

if __name__ == '__main__':
    # 确保sympy已安装
    try:
        import sympy
    except ImportError:
        print("\n错误: SymPy 库未安装。")
        print("请在您的终端中运行: pip install sympy\n")
        exit()
        
    generate_com_derivatives()