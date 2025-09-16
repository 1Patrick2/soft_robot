import numpy as np
import sys
import os
import time
import multiprocessing
import itertools
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from analysis.workspace_analysis import run_monte_carlo_workspace_analysis, calculate_workspace_volume


def run_stiffness_test_worker(args):
    """并行计算的工作单元，负责测试单一刚度组合。"""
    config_path, pss_stiffness, cms_stiffness, base_q0_6d = args
    
    # 1. 加载基础配置
    params = load_config(config_path)
    
    # 2. 应用当前的刚度组合进行修改
    params['Stiffness']['pss_total_equivalent_bending_stiffness'] = pss_stiffness
    params['Stiffness']['cms_bending_stiffness'] = cms_stiffness
    
    # 3. 运行蒙特卡洛工作空间分析 (使用较少的样本以加速)
    workspace_points = run_monte_carlo_workspace_analysis(
        params,
        num_samples=1000,  # [V-Final 优化] 减少样本量以在合理时间内完成分析
        max_displacement=0.05,
        q0_6d=base_q0_6d,
        use_tqdm=False, # 在子进程中关闭tqdm，避免打印混乱
        parallel=False  # <--- [核心修复] 彻底关闭内层并行！
    )
    
    # 4. 计算并返回工作空间体积
    volume = calculate_workspace_volume(workspace_points)
    
    return pss_stiffness, cms_stiffness, volume

if __name__ == '__main__':
    start_time = time.time()
    print("--- [V-Final] 启动刚度参数敏感性分析... --- ")

    # --- 1. 定义刚度的对数搜索空间 ---
    # --- 1. [核心] 定义第三轮的“冲顶”搜索空间 ---
    # X轴 (PSS): 在0.01到0.05之间进行精细线性搜索
    pss_stiffness_range = np.linspace(0.01, 0.05, 5)

    # Y轴 (CMS): 在0.5到15.0之间进行拓展性对数搜索
    cms_stiffness_range = np.logspace(np.log10(0.5), np.log10(15.0), 5)
    
    num_combinations = len(pss_stiffness_range) * len(cms_stiffness_range)

    print("\n定义的参数搜索空间:")
    print(f"  - pss_stiffness: {len(pss_stiffness_range)} points from {pss_stiffness_range[0]:.3f} to {pss_stiffness_range[-1]:.3f}")
    print(f"  - cms_stiffness: {len(cms_stiffness_range)} points from {cms_stiffness_range[0]:.3f} to {cms_stiffness_range[-1]:.3f}")
    print(f"\n总计需要测试 {num_combinations} 种刚度组合。")

    # --- 2. 准备基础配置和并行任务 ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'config.json')
    base_config = load_config(config_path)
    q0_7d = np.array(base_config['Initial_State']['q0'])
    q0_6d = q0_7d[1:]

    tasks = [(config_path, pss, cms, q0_6d) for pss in pss_stiffness_range for cms in cms_stiffness_range]

    # --- 3. 并行执行设计寻优 ---
    num_cpus = multiprocessing.cpu_count()
    print(f"\n使用 {max(1, num_cpus - 1)} 个CPU核心并行执行寻优...")
    
    from tqdm import tqdm
    with multiprocessing.Pool(processes=max(1, num_cpus - 1)) as pool:
        results = list(tqdm(pool.imap(run_stiffness_test_worker, tasks), total=num_combinations, desc="Stiffness Sensitivity"))

    # --- 4. 分析并报告结果 ---
    print("\n--- 敏感性分析完成 ---")
    
    best_volume = -1
    best_pss = None
    best_cms = None

    # 将结果重塑为网格以便绘图
    volumes = np.zeros((len(pss_stiffness_range), len(cms_stiffness_range)))
    for pss, cms, vol in results:
        if vol > best_volume:
            best_volume = vol
            best_pss = pss
            best_cms = cms
        # 找到pss和cms在原始范围中的索引
        pss_idx = np.where(np.isclose(pss_stiffness_range, pss))[0][0]
        cms_idx = np.where(np.isclose(cms_stiffness_range, cms))[0][0]
        volumes[pss_idx, cms_idx] = vol

    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")

    if best_pss is not None:
        print("\n🏆 发现最优刚度组合! 🏆")
        print(f"  - 最大工作空间体积: {best_volume * 1e6:.2f} cm^3")
        print(f"  - 最佳PSS刚度: {best_pss:.4f}")
        print(f"  - 最佳CMS刚度: {best_cms:.4f}")
        print(f"  - 最佳刚度比 (PSS/CMS): {best_pss/best_cms:.2f}")
    else:
        print("\n❌ 未能找到任何有效的配置。")

    # --- 5. 绘制热力图 ---
    print("\n正在绘制热力图...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(volumes.T, xticklabels=np.round(pss_stiffness_range, 3), yticklabels=np.round(cms_stiffness_range, 4), annot=True, fmt=".2e", cmap="viridis")
    plt.xlabel("PSS Bending Stiffness (pss_total_equivalent_bending_stiffness)")
    plt.ylabel("CMS Bending Stiffness (cms_bending_stiffness)")
    plt.title("Stiffness Sensitivity vs. Workspace Volume (Convex Hull)")
    plt.gca().invert_yaxis() # Y轴通常是升序的
    
    heatmap_path = os.path.join(project_root, 'plots', 'sensitivity_analysis_stiffness_heatmap_round3.png')
    plt.savefig(heatmap_path, dpi=300)
    print(f"热力图已保存至: {os.path.abspath(heatmap_path)}")
    plt.close()
