# verification/verify_multisegment_drive_mapping.py

import numpy as np
import sys
import os
import logging

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import both functions for comparison
from src.statics import calculate_drive_mapping, calculate_drive_mapping_geometric
from src.utils.read_config import load_config

logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == '__main__':
    print("--- 多段驱动映射函数 (Multi-Segment Drive Mapping) 对比验证 ---")
    config = load_config('config/config.json')
    np.set_printoptions(precision=8, suppress=True, linewidth=150)

    # --- 1. 恢复三段机器人配置 (如果之前被修改过) ---
    # This assumes the default config is for the 3-segment robot.
    # If not, we would modify it here.
    print("\n--- 测试条件: 完整三段机器人模型 ---")

    # --- 2. 执行对比测试 ---
    # A non-trivial configuration that involves all segments
    q_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) 
    print(f"  - 测试构型 q: {q_test}")

    # 2.1 获取新的解析法结果
    analytical_result = calculate_drive_mapping(q_test, config)

    # 2.2 获取旧的几何法结果
    geometric_result = calculate_drive_mapping_geometric(q_test, config)

    # --- 3. 对比和结论 ---
    diff_norm = np.linalg.norm(analytical_result - geometric_result)
    total_length = config['Geometry']['PSS_initial_length'] + config['Geometry']['CMS_proximal_length'] + config['Geometry']['CMS_distal_length']
    
    # As per test.md, tolerance should be < 1e-3 * L
    tolerance = 1e-3 * total_length

    print("\n--- 结果对比 ---")
    print(f"  - 新解析法结果: {analytical_result}")
    print(f"  - 旧几何法结果: {geometric_result}")
    print(f"  - 差值范数:     {diff_norm:.8e}")
    print(f"  - 容忍误差:     < {tolerance:.8e} (1e-3 * L_total)")

    if diff_norm < tolerance:
        print("\n[结论] ✅ 验证通过！ 新的解析法与旧的几何法在多段模型下结果接近。")
        print("这说明解析法作为近似是合理的，可以用它来修复梯度。")
    else:
        print("\n[结论] ❌ 验证失败！ 新旧方法差异巨大，解析法的多段累加逻辑可能存在问题。")
