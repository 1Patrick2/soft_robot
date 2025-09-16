# verification/test_single_axis.py
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.solver import solve_static_equilibrium_diff4
from src.kinematics import forward_kinematics
from src.statics import expand_diff4_to_motor8, calculate_drive_mapping

print("--- Single-Axis/Component Behavior Test ---")

# 1. 关闭 statics 模块的调试打印，以免干扰输出
# (我们已经在之前的步骤中手动注释掉了，这里作为提醒)

# 2. 加载配置和初始构型
params = load_config('config/config.json')
q0 = np.zeros(6)

# 3. 定义测试用例
tests = {
  'diff4_xpos': np.array([0.05, 0, 0, 0]),
  'diff4_xneg': np.array([-0.05, 0, 0, 0]),
  'diff4_ypos': np.array([0, 0.05, 0, 0]),
  'diff4_yneg': np.array([0, -0.05, 0, 0]),
  'diff4_diag': np.array([0.03,0.03,0,0])
}

# 4. 运行测试并打印结果
np.set_printoptions(precision=6, suppress=True)
for name, d in tests.items():
    print(f"--- Testing: {name} ---")
    print(f"Input diff4: {d}")
    
    # 调用 diff4 求解器
    res = solve_static_equilibrium_diff4(q0, d, params)
    q_eq = res['q_solution']
    
    if q_eq is not None:
        T, _ = forward_kinematics(q_eq, params)
        delta_l_motor = expand_diff4_to_motor8(d, params)
        # delta_l_robot = calculate_drive_mapping(q_eq, params) # 暂时关闭，减少输出
        
        print(f"  -> Resulting End-Effector Position: {T[:3,3]}")
        # print(f"     delta_l_motor: {delta_l_motor}")
        # print(f"     delta_l_robot: {np.round(delta_l_robot, 6)}")
    else:
        print("  -> Solver failed to find a solution.")
