"""
本脚本对求解器的 `smoothing_weight` (平滑权重) 参数进行敏感性分析。

它旨在探索两个互斥目标之间的权衡关系：
1. 路径跟踪精度：机器人实际路径与目标路径的贴合程度。
2. 驱动力平滑度：驱动力在相邻路径点之间的变化幅度。

脚本会遍历一系列不同的平滑权重值，对每个值都运行一次完整的路径跟踪仿真，
并记录相应的平均路径误差和平均驱动力变化。最后，它将两个指标绘制在同一张
双Y轴图上，以可视化地展示它们之间的消长关系，从而帮助为具体应用选择一个
最均衡的权重值。
"""
import numpy as np
import matplotlib.pyplot as plt
import time

# 导入项目模块
from src.outer_solver import solve_inverse_kinematics
from src.kinematics import forward_kinematics
from src.solver import solve_static_equilibrium
from src.utils.read_config import load_config

def run_path_tracking_with_smoothing(config, smoothing_weight):
    """
    一个辅助函数，为给定的配置和指定的平滑权重，运行一次路径跟踪仿真。

    Args:
        config (dict): 机器人的完整配置字典。
        smoothing_weight (float): 求解器中用于惩罚驱动力变化的权重。

    Returns:
        tuple: 一个包含两个性能指标的元组：
               - mean_path_error (float): 平均路径（位置）误差。
               - mean_force_change (float): 相邻路径点间驱动力的平均变化范数。
    """
    # 1. 定义本实验的目标路径 (使用新的、在工作空间内的圆形路径)
    target_poses = []
    angles = np.linspace(0, 2 * np.pi, 40) # 一个完整的圆形路径
    for angle in angles:
        x = 0.0 + 0.04 * np.cos(angle)
        y = 0.0 + 0.04 * np.sin(angle)
        z = 0.08
        target_poses.append(np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ]))

    # 2. 初始化仿真循环变量
    q_guess = np.array(config['Initial_State']['q0'])
    # 将初始猜测的驱动力 `tau_guess_dict` 和“上一步”的驱动力 `previous_tau_dict` 设为相同，
    # 以避免在第一个路径点就产生不必要的巨大跳变。
    # 使用随机化的初始猜测来打破对称性僵局
    initial_tau_flat = np.random.rand(8) * 10
    tau_guess_dict = {
        'tensions_short': initial_tau_flat[:4],
        'tensions_long': initial_tau_flat[4:]
    }
    previous_tau_dict = tau_guess_dict.copy()

    # 3. 循环求解路径上的每个点，并记录性能指标
    path_errors = []
    force_changes = []
    for target_pose in target_poses:
        ik_result = solve_inverse_kinematics(
            target_pose, tau_guess_dict, q_guess, config, 
            previous_tau=previous_tau_dict, 
            smoothing_weight=smoothing_weight
        )
        
        if ik_result.success and ik_result.status > 0:
            force_flat = ik_result.x
            tensions_final = {'tensions_short': force_flat[:4], 'tensions_long': force_flat[4:]}
            q_solution = solve_static_equilibrium(q_guess, tensions_final, config)
            actual_pose, _ = forward_kinematics(q_solution, config)
            
            # --- 记录性能指标 ---
            # 指标1: 路径跟踪误差 (仅位置)
            pos_error = np.linalg.norm(actual_pose[:3, 3] - target_pose[:3, 3])
            path_errors.append(pos_error)
            
            # 指标2: 驱动力变化 (平滑度)
            prev_force_flat = np.concatenate(list(previous_tau_dict.values()))
            force_change = np.linalg.norm(force_flat - prev_force_flat)
            force_changes.append(force_change)

            # --- 为下一次迭代更新“暖启动”变量 ---
            tau_guess_dict = tensions_final
            previous_tau_dict = tensions_final
            q_guess = q_solution
    
    # 计算两个指标的平均值
    mean_path_error = np.mean(path_errors) if path_errors else float('inf')
    mean_force_change = np.mean(force_changes) if force_changes else float('inf')
    
    return mean_path_error, mean_force_change

def main():
    """
    组织和执行平滑权重敏感性分析的主函数。
    """
    print("--- 开始平滑权重敏感性分析 ---")
    start_time = time.time()

    # 1. 定义要测试的平滑权重范围
    smoothing_weights = np.logspace(-8, -2, 10) # 在1e-8到0.01的对数尺度上取10个点
    path_error_results = []
    force_change_results = []

    weights_str = np.array2string(smoothing_weights, formatter={'float_kind':lambda x: "%.4f" % x})
    print(f"将要测试的平滑权重值: \n{weights_str}")

    # 2. 循环遍历每个权重，运行仿真，并存储返回的两个性能指标
    config = load_config('config/config.json')
    for i, weight in enumerate(smoothing_weights):
        print(f"\n--- 正在运行实验 {i+1}/{len(smoothing_weights)}: 平滑权重 = {weight:.4e} ---")
        mean_error, mean_force_change = run_path_tracking_with_smoothing(config, weight)
        path_error_results.append(mean_error)
        force_change_results.append(mean_force_change)
        print(f"--> 完成！平均路径误差: {mean_error:.4f} m, 平均驱动力变化: {mean_force_change:.4f} N")

    # 3. 在一张图上绘制两个性能指标，以可视化它们之间的权衡关系
    print("\n--- 所有实验完成，正在绘制结果图 ---")
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 坐标轴1 (左侧): 路径误差
    color = 'tab:red'
    ax1.set_xlabel('平滑权重 (对数坐标)')
    ax1.set_ylabel('平均路径误差 (m)', color=color)
    ax1.plot(smoothing_weights, path_error_results, 'o-', color=color, label='路径误差')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.grid(True, which="both", ls="--")

    # 坐标轴2 (右侧): 驱动力变化，共享X轴
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('平均驱动力变化 (N)', color=color)
    ax2.plot(smoothing_weights, force_change_results, 's--', color=color, label='驱动力变化')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('敏感性分析: 平滑权重 vs. 性能权衡', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图像
    output_path = "plots/smoothing_analysis_tradeoff.png"
    plt.savefig(output_path)
    print(f"结果图已保存至: {output_path}")
    plt.close()

    end_time = time.time()
    print(f"\n分析完成，总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
