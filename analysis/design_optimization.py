import numpy as np
import sys
import os
import time
import multiprocessing
import itertools
from copy import deepcopy

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from analysis.workspace_analysis import run_monte_carlo_workspace_analysis, calculate_workspace_volume

def run_single_design_iteration(args):
    """并行计算的工作单元，负责测试单一参数组合。"""
    config_path, param_combination, base_q0_6d = args
    
    # 1. 加载基础配置
    params = load_config(config_path)
    
    # 2. 应用当前的参数组合进行修改
    params['Stiffness']['pss_total_equivalent_bending_stiffness'] = param_combination['pss_stiffness']
    params['Stiffness']['cms_bending_stiffness'] = param_combination['cms_stiffness']
    params['Drive_Properties']['cable_stiffness'] = param_combination['cable_stiffness']
    
    # 3. 运行蒙特卡洛工作空间分析 (使用较少的样本以加速)
    #    注意：这里关闭了tqdm，避免在并行输出中造成混乱
    workspace_points = run_monte_carlo_workspace_analysis(
        params,
        num_samples=2000,  # 在寻优中使用较少的样本量以提高速度
        max_displacement=0.05,
        q0_6d=base_q0_6d,
        use_tqdm=False, # 关闭内层tqdm
        parallel=False # 工作空间分析本身串行运行，因为我们已经在设计层面并行
    )
    
    # 4. 计算并返回工作空间体积
    volume = calculate_workspace_volume(workspace_points)
    
    return param_combination, volume

if __name__ == '__main__':
    start_time = time.time()
    print("--- [V-Final] 启动参数化设计寻优... --- ")

    # --- 1. 定义参数的搜索空间 ---
    # 根据 test.md 的建议
    search_space = {
        'pss_stiffness': [0.05, 0.1, 0.2, 0.5, 1.0],
        'cms_stiffness': [0.002, 0.005, 0.01],
        'cable_stiffness': [2000, 6000, 10000]
    }
    print("\n定义的参数搜索空间:")
    for key, value in search_space.items():
        print(f"  - {key}: {value}")

    # 生成所有参数组合
    keys, values = zip(*search_space.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    num_combinations = len(param_combinations)
    print(f"\n总计需要测试 {num_combinations} 种参数组合。")

    # --- 2. 准备基础配置和并行任务 ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    base_config = load_config(config_path)
    q0_7d = np.array(base_config['Initial_State']['q0'])
    q0_6d = q0_7d[1:]

    tasks = [(config_path, combo, q0_6d) for combo in param_combinations]

    # --- 3. 并行执行设计寻优 ---
    best_volume = -1
    best_combination = None
    
    num_cpus = multiprocessing.cpu_count()
    print(f"\n使用 {max(1, num_cpus - 1)} 个CPU核心并行执行寻优...")
    
    from tqdm import tqdm
    with multiprocessing.Pool(processes=max(1, num_cpus - 1)) as pool:
        # 使用tqdm显示总进度
        results = list(tqdm(pool.imap(run_single_design_iteration, tasks), total=num_combinations, desc="Design Optimization"))

    # --- 4. 分析并报告结果 ---
    print("\n--- 设计寻优完成 ---")
    for combination, volume in results:
        if volume > best_volume:
            best_volume = volume
            best_combination = combination

    end_time = time.time() 
    print(f"总耗时: {end_time - start_time:.2f} 秒")

    if best_combination:
        print("\n🏆 发现冠军配置! 🏆")
        print(f"  - 最大工作空间体积: {best_volume * 1e6:.2f} cm^3")
        print("  - 最佳参数组合:")
        for key, value in best_combination.items():
            print(f"    - {key}: {value}")
    else:
        print("\n❌ 未能找到任何有效的配置。")
