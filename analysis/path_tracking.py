# analysis/path_tracking.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
import time

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.kinematics import forward_kinematics
# [V-Final.X] 导入所有需要的“武器”
from src.outer_solver import (
    solve_ik_statistically_robust,
    solve_inverse_kinematics,
    calculate_task_jacobian,
    get_characteristic_scales,
    calculate_pose_error
)

def generate_circular_path(radius, num_points, center_x, center_y, center_z):
    """
    在 xy 平面上生成一个圆形路径，z 高度固定。
    """
    path_points = []
    angles = np.linspace(0, 2 * np.pi, num_points)

    for angle in angles:
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        z = center_z
        
        target_pose = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])
        path_points.append(target_pose)
    
    return path_points

def plot_results(target_path, actual_path, forces, title_suffix=""):
    """
    可视化仿真结果，包括三视图、力曲线和带颜色渐变的路径。
    """
    print("\n正在绘制结果图...")
    
    plt.figure(figsize=(15, 7))
    forces_array = np.array(forces)
    for i in range(forces_array.shape[1]):
        plt.plot(forces_array[:, i], label=f'τ_{{i+1}}')
    plt.title(f'Driving Forces vs. Path Points {title_suffix}')
    plt.xlabel('Path Point Index')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/forces_plot{title_suffix}.png")
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(f'Path Tracking Analysis {title_suffix}', fontsize=16)

    target_x = [p[0, 3] for p in target_path]
    target_y = [p[1, 3] for p in target_path]
    target_z = [p[2, 3] for p in target_path]
    actual_x = [p[0, 3] for p in actual_path]
    actual_y = [p[1, 3] for p in actual_path]
    actual_z = [p[2, 3] for p in actual_path]

    num_points = len(actual_x)
    colors = plt.cm.viridis(np.linspace(0, 1, num_points))

    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.plot(target_x, target_y, target_z, 'r--', label='Target Path', alpha=0.7, linewidth=2)
    ax3d.plot(actual_x, actual_y, actual_z, 'o-', color='cyan', markersize=3, linewidth=1.5, label='Actual Path', zorder=5)
    # ax3d.scatter(actual_x, actual_y, actual_z, c=colors, s=25, label='Actual Path', zorder=5)
    ax3d.scatter(actual_x[0], actual_y[0], actual_z[0], c='lime', s=100, ec='black', zorder=10, label='Start')
    ax3d.set_title('3D View')
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.legend(); ax3d.axis('equal')

    views = [
        (fig.add_subplot(2, 2, 2), (target_x, target_y), (actual_x, actual_y), 'XY Plane', 'X (m)', 'Y (m)'),
        (fig.add_subplot(2, 2, 3), (target_x, target_z), (actual_x, actual_z), 'XZ Plane', 'X (m)', 'Z (m)'),
        (fig.add_subplot(2, 2, 4), (target_y, target_z), (actual_y, actual_z), 'YZ Plane', 'Y (m)', 'Z (m)')
    ]

    for ax, target_coords, actual_coords, title, xlabel, ylabel in views:
        ax.plot(target_coords[0], target_coords[1], 'r--', alpha=0.7, linewidth=2, label='Target')
        ax.plot(actual_coords[0], actual_coords[1], 'o-', color='cyan', markersize=3, linewidth=1.5, label='Actual Path', zorder=5)
        ax.scatter(actual_coords[0][0], actual_coords[1][0], c='lime', s=100, ec='black', zorder=10, label='Start')
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True); ax.axis('equal'); ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/path_plot_3_view{title_suffix}.png")
    plt.close()

def track_path_final_calibrated(target_poses, config, reset_interval=10):
    """
    [V-Final.X] 终极版：使用“统计择优”进行定期校准的预测-校正跟踪器
    """
    print(f"\n--- 开始最终校准路径跟踪 (每 {reset_interval} 点进行统计择优) ---")
    start_time = time.time()
    actual_poses, required_forces = [], []
    q_prev, tau_prev_dict, tau_prev_flat = None, None, None
    scales = get_characteristic_scales(config)

    for i, target_pose in enumerate(tqdm(target_poses, desc="Calibrated Tracking")):
        # --- 模式一：“GPS信号塔”校准 (在第一点和每个重置点) ---
        if i == 0 or i % reset_interval == 0:
            if i == 0:
                print(f"Point {i}: Performing initial robust statistical solve...")
            else:
                print(f"\nPoint {i}: Performing robust statistical recalibration...")
            
            ik_result = solve_ik_statistically_robust(target_pose, config, num_runs=5)
            
            if not (ik_result and ik_result.success and ik_result.q_solution is not None):
                print(f"❌ 致命错误: 路径点 {i} 全局校准失败。跟踪终止。\n")
                break
        
        # --- 模式二：“巡航引擎” (对其他所有点) ---
        else:
            try:
                T_prev = actual_poses[-1]
                J_task = calculate_task_jacobian(tau_prev_dict, q_prev, config, scales)
                pose_error_vec = calculate_pose_error(T_prev, target_pose)
                
                lambda_sq = 0.01**2
                J_J_T = J_task @ J_task.T + lambda_sq * np.eye(6)
                delta_tau = J_task.T @ np.linalg.inv(J_J_T) @ pose_error_vec
                
                tau_guess_flat = np.clip(tau_prev_flat - delta_tau, 0, 100)
                tau_guess_dict = {'tensions_short': tau_guess_flat[:4], 'tensions_long': tau_guess_flat[4:]}
                q_guess_for_corrector = q_prev

            except Exception as e:
                print(f"\n警告: 路径点 {i} 预测步失败: {e}。将回退到简单的热启动。\n")
                tau_guess_dict = tau_prev_dict
                q_guess_for_corrector = q_prev

            ik_result = solve_inverse_kinematics(
                T_target=target_pose,
                initial_tau_guess=tau_guess_dict,
                q_guess=q_guess_for_corrector,
                params=config,
                previous_tau=tau_prev_dict,
                position_error_weight=50000.0,
                orientation_error_weight=100.0,
                grad_energy_weight=0.1,
                smoothing_weight=1e-6
            )

        # --- 通用结果处理 ---
        if ik_result and ik_result.success and ik_result.q_solution is not None:
            q_prev = ik_result.q_solution
            tau_prev_flat = ik_result.x
            tau_prev_dict = {'tensions_short': tau_prev_flat[:4], 'tensions_long': tau_prev_flat[4:]}
        else:
            print(f"\n警告: 路径点 {i} 求解失败。将沿用上一点的状态，这可能导致误差累积。\n")
            # If solve fails, we keep the previous state (q_prev, tau_prev_dict, tau_prev_flat)
        
        T_actual, _ = forward_kinematics(q_prev, config)
        actual_poses.append(T_actual)
        required_forces.append(tau_prev_flat)

    end_time = time.time()
    print(f"路径跟踪完成，耗时: {end_time - start_time:.2f} 秒")
    return actual_poses, required_forces, "_final_calibrated_tracking"

def main():
    """
    路径跟踪仿真主函数。
    """
    config = load_config('config/config.json')
    workspace_data_path = 'plots/workspace_points.npy'
    if not os.path.exists(workspace_data_path):
        print(f"错误: 工作空间数据文件未找到: {workspace_data_path}\n")
        return

    workspace_points = np.load(workspace_data_path)
    
    center_x = np.mean(workspace_points[:, 0])
    center_y = np.mean(workspace_points[:, 1])
    center_z = np.mean(workspace_points[:, 2])
    radius = 0.4 * min(np.std(workspace_points[:, 0]), np.std(workspace_points[:, 1]))

    if radius < 0.001:
        print(f"错误: 计算出的路径半径 ({radius:.4f} m) 过小。\n")
        return

    target_poses = generate_circular_path(
        radius=radius, 
        num_points=50,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z
    )

    # [核心] 调用最终的、带校准的跟踪函数
    actual_poses, required_forces, suffix = track_path_final_calibrated(target_poses, config)

    if actual_poses:
        plot_results(target_poses, actual_poses, required_forces, title_suffix=suffix)
    else:
        print("没有成功求解的路径点，无法生成结果图。\n")

if __name__ == "__main__":
    # 确保在Windows上使用multiprocessing时，代码在`if __name__ == "__main__":`内
    # 这是防止子进程无限递归地重新导入和执行代码的关键保护措施。
    main()
