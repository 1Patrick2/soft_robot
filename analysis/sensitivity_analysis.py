# analysis/sensitivity_analysis.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import logging
from src.utils.read_config import load_config
from src.outer_solver import solve_ik_globally

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_sensitivity_analysis():
    """
    对机器人的关键刚度参数进行敏感性分析，以找到最优配置。
    """
    # --- 1. 定义扫描参数 ---
    # 我们将扫描PSS和CMS的抗弯刚度
    pss_stiffness_range = np.logspace(np.log10(0.1), np.log10(10), 5) # 0.1 to 10
    cms_stiffness_range = np.logspace(np.log10(0.1), np.log10(10), 5) # 0.1 to 10

    results = []

    # --- 2. 定义固定的测试目标 ---
    base_config = load_config('config/config.json')
    L_total = base_config['Geometry']['PSS_initial_length'] + base_config['Geometry']['CMS_proximal_length'] + base_config['Geometry']['CMS_distal_length']
    target_pose = np.array([
        [1, 0, 0, 0.05],
        [0, 1, 0, 0.05],
        [0, 0, 1, L_total * 0.7],
        [0, 0, 0, 1]
    ])

    logging.info("Starting stiffness sensitivity analysis...")
    logging.info(f"PSS Stiffness Range: {pss_stiffness_range}")
    logging.info(f"CMS Stiffness Range: {cms_stiffness_range}")

    # --- 3. 循环遍历所有参数组合 ---
    total_runs = len(pss_stiffness_range) * len(cms_stiffness_range)
    current_run = 0
    for pss_stiffness in pss_stiffness_range:
        for cms_stiffness in cms_stiffness_range:
            current_run += 1
            start_time = time.time()
            logging.info(f"--- Running ({current_run}/{total_runs}): PSS={pss_stiffness:.4f}, CMS={cms_stiffness:.4f} ---")

            # a. 修改并加载配置
            temp_config = base_config.copy()
            temp_config['Stiffness']['pss_total_equivalent_bending_stiffness'] = pss_stiffness
            temp_config['Stiffness']['cms_bending_stiffness'] = cms_stiffness

            # b. 运行IK求解器
            # 为了结果的稳定性，我们可以考虑运行多次取平均值，但为了速度，先运行一次
            result = solve_ik_globally(target_pose, temp_config)
            final_error_mm = result['error_mm']

            # c. 记录结果
            results.append({
                'pss_stiffness': pss_stiffness,
                'cms_stiffness': cms_stiffness,
                'final_error_mm': final_error_mm
            })
            end_time = time.time()
            logging.info(f"Result: error = {final_error_mm:.4f} mm. (Took {end_time - start_time:.2f}s)")

    # --- 4. 数据处理与可视化 ---
    df = pd.DataFrame(results)
    
    # 创建一个数据透视表用于热力图
    pivot_table = df.pivot(index='cms_stiffness', columns='pss_stiffness', values='final_error_mm')

    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis_r", 
                linewidths=.5, cbar_kws={'label': 'Final Position Error (mm)'})
    plt.title('Stiffness Sensitivity Analysis for IK Solver Accuracy')
    plt.xlabel('PSS Bending Stiffness (Nm^2)')
    plt.ylabel('CMS Bending Stiffness (Nm^2)')
    
    # 保存图像
    output_path = 'plots/sensitivity_analysis_stiffness_heatmap.png'
    plt.savefig(output_path)
    logging.info(f"Sensitivity analysis complete. Heatmap saved to {output_path}")
    plt.show()

if __name__ == '__main__':
    run_sensitivity_analysis()