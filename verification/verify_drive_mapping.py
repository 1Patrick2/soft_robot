# verification/verify_drive_mapping.py

import numpy as np
import sys
import os

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.statics import calculate_drive_mapping
from src.utils.read_config import load_config

if __name__ == '__main__':
    print("--- 驱动映射 (Drive Mapping) 单点检查 ---")
    
    # 1. 加载配置
    params = load_config('config/config.json')
    Lp = params['Geometry']['PSS_initial_length']
    Lc1 = params['Geometry']['CMS_proximal_length']
    
    # 2. 定义测试构型 q (PSS 伸直, CMS_Proximal 弯曲90度)
    # 这是一个更符合物理现实的测试场景
    Lc1 = params['Geometry']['CMS_proximal_length']
    kappa_c1_90_deg = (np.pi / 2) / Lc1
    q_test = np.array([0.0, 0.0, kappa_c1_90_deg, 0.0, 0.0, 0.0])
    
    print(f"\n测试构型 q (PSS straight, CMS_Proximal 90-deg bend):")
    print(f"  {q_test}")
    
    # 3. 调用驱动映射函数
    # 注意：calculate_drive_mapping 内部有详细的打印输出
    delta_l = calculate_drive_mapping(q_test, params)
    
    # 4. 提取并打印关键结果
    # calculate_drive_mapping 的实现是从 l0 - len_bent 得到 delta_l
    # 我们需要从它的打印输出中观察，或者重新计算以清晰展示
    
    # 为了清晰，我们在这里重新计算并打印 delta_s 的分量
    delta_s = delta_l[:4]
    l0_s = Lp + Lc1
    
    # len_s_bent 可以从 delta_s 和 l0_s 反推出来
    len_s_bent = l0_s - delta_s
    
    print("\n--- 检查结果 (短线组) ---")
    np.set_printoptions(precision=6, suppress=True)
    print(f"  - 初始长度 (l0_s): {l0_s:.6f} m")
    print(f"  - 弯曲后各缆绳长度 (len_s_bent): {len_s_bent}")
    print(f"  - 长度变化量 (delta_s): {delta_s}")
    
    # 5. 分析 delta_s 的量级
    avg_delta_s = np.mean(np.abs(delta_s))
    print(f"\n  - 平均长度变化量 |delta_s|: {avg_delta_s:.6f} m ({avg_delta_s*1000:.3f} mm)")
    
    if avg_delta_s > 1e-4: # 阈值：0.1mm
        print("\n✅ 结论: delta_s 的量级看起来是合理的 (非零且可观)。")
    else:
        print("\n❌ 结论: delta_s 的量级过小，可能存在几何定义或模型问题。")