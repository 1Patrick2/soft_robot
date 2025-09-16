import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
import time
from joblib import Parallel, delayed

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 假设项目结构允许这样导入
from src.outer_solver import solve_inverse_kinematics, calculate_pose_error, calculate_task_jacobian
from src.kinematics import forward_kinematics
from src.solver import solve_static_equilibrium
from src.utils.read_config import load_config
from src.nondimensionalizer import get_characteristic_scales

def generate_circular_path(radius, num_points, center_x, center_y, center_z):
    """
    在 xy 平面上生成一个圆形路径，z 高度固定。
    返回一个姿态矩阵的列表。
    """
    print("正在生成圆形路径...")
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
    
    print(f"成功生成包含 {len(path_points)} 个点的路径。")
    return path_points

def plot_results(target_path, actual_path, forces, title_suffix=""):
    """
    可视化仿真结果，包括三视图、力曲线和带颜色渐变的路径。
    """
    print("\n正在绘制结果图...")
    
    # 1. 绘制驱动力曲线
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
    print(f"Forces plot saved to plots/forces_plot{title_suffix}.png")
    plt.close()

    # 2. 创建三视图和3D路径对比图
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

    # --- 3D View ---
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.plot(target_x, target_y, target_z, 'r--', label='Target Path', alpha=0.7, linewidth=2)
    ax3d.plot(actual_x, actual_y, actual_z, '-', color='deepskyblue', alpha=0.8, label='Actual Path')
    ax3d.scatter(actual_x, actual_y, actual_z, c=colors, s=25, zorder=5)
    ax3d.scatter(actual_x[0], actual_y[0], actual_z[0], c='lime', s=100, ec='black', zorder=10, label='Start')
    ax3d.set_title('3D View')
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.legend()
    ax3d.axis('equal')

    # --- 2D Views ---
    views = [
        (fig.add_subplot(2, 2, 2), (target_x, target_y), (actual_x, actual_y), 'XY Plane', 'X (m)', 'Y (m)'),
        (fig.add_subplot(2, 2, 3), (target_x, target_z), (actual_x, actual_z), 'XZ Plane', 'X (m)', 'Z (m)'),
        (fig.add_subplot(2, 2, 4), (target_y, target_z), (actual_y, actual_z), 'YZ Plane', 'Y (m)', 'Z (m)')
    ]

    for ax, target_coords, actual_coords, title, xlabel, ylabel in views:
        ax.plot(target_coords[0], target_coords[1], 'r--', alpha=0.7, linewidth=2, label='Target')
        ax.plot(actual_coords[0], actual_coords[1], '-', color='deepskyblue', alpha=0.8, label='Actual Path')
        ax.scatter(actual_coords[0], actual_coords[1], c=colors, s=25, zorder=5)
        ax.scatter(actual_coords[0][0], actual_coords[1][0], c='lime', s=100, ec='black', zorder=10, label='Start')
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True); ax.axis('equal'); ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/path_plot_3_view{title_suffix}.png")
    print(f"3-view path plot saved to plots/path_plot_3_view{title_suffix}.png")
    plt.close()

def track_path_task_space(target_poses, config):
    """
    [V-Final] 最终版：对路径上的每个点都执行多起点IK求解，以保证全局最优。
    """
    print("\n--- 开始任务空间 (Task-space) 跟踪 [多起点全局优化] ---")
    start_time = time.time()

    actual_poses = []
    required_forces = []
    
    # 初始化状态变量
    q_prev = np.array(config['Initial_State']['q0'])
    Lp0 = config['Geometry']['PSS_initial_length']

    # --- [核心修改] 将多起点搜索逻辑封装成一个函数，用于路径的每一点 ---
    def solve_point_globally(target_pose, q_guess_for_inner_loop):
        # 定义单次尝试的辅助函数
        def solve_single_attempt(attempt_idx, params_dict, target_pose):
            from src.statics import calculate_actuation_jacobian_analytical, calculate_elastic_gradient_analytical, calculate_gravity_gradient_analytical
            # 智能初始猜测：随机生成有效构型 -> 反算匹配的驱动力
            q_rand = np.random.rand(7)
            q_rand[0] = Lp0 * (0.8 + 0.4 * np.random.rand())
            q_rand[1::2] = (np.random.rand(3) - 0.5) * 80
            q_rand[2::2] = np.random.rand(3) * 2 * np.pi

            J_act_T = calculate_actuation_jacobian_analytical(q_rand, params_dict).T
            grad_E = calculate_elastic_gradient_analytical(q_rand, params_dict)
            grad_G = calculate_gravity_gradient_analytical(q_rand, params_dict)
            b = grad_E + grad_G
            
            tau_vec, _, _, _ = np.linalg.lstsq(J_act_T, b, rcond=None)
            tau_vec = np.clip(tau_vec, 0, 100)
            
            tau_guess = {'tensions_short': tau_vec[:4], 'tensions_long': tau_vec[4:]}

            # 使用这对(q_rand, tau_guess)作为初始点进行IK求解
            return solve_inverse_kinematics(
                T_target=target_pose, 
                initial_tau_guess=tau_guess, 
                q_guess=q_rand, # 使用随机的q作为内循环起点
                params=params_dict,
                attempt_index=attempt_idx,
                pose_error_weight=10000.0,
                grad_energy_weight=1.0,
                smoothing_weight=1e-5
            )

        # 为当前目标点执行并行化的多起点搜索
        num_attempts = 8 # 路径跟踪中，可适当减少尝试次数以提高速度
        results_list = Parallel(n_jobs=-1, backend='loky')( 
            delayed(solve_single_attempt)(i, config, target_pose) for i in range(num_attempts)
        )

        # 从结果中筛选最佳解
        successful_results = [res for res in results_list if res.success and res.q_solution is not None]
        if not successful_results:
            return None # 如果所有尝试都失败，则返回None
        
        best_result = min(successful_results, key=lambda res: res.cost)
        return best_result

    # --- 路径跟踪主循环 ---
    for i, target_pose in enumerate(tqdm(target_poses, desc="Tracking Path Globally")):
        # 对每个点都进行全局最优求解
        ik_result = solve_point_globally(target_pose, q_prev)

        # --- 处理IK结果 ---
        if ik_result:
            tau_final_flat = ik_result.x
            q_solution = ik_result.q_solution

            # 更新状态以备下一步使用 (尽管在当前策略下，q_prev不是必须的)
            q_prev = q_solution
            
            required_forces.append(tau_final_flat)
            actual_pose, _ = forward_kinematics(q_solution, config)
            actual_poses.append(actual_pose)
        else:
            print(f"\n警告: 路径点 {i} 全局求解失败。将终止跟踪。")
            break

    end_time = time.time()
    print(f"全局路径跟踪完成，耗时: {end_time - start_time:.2f} 秒")

    return actual_poses, required_forces, "_task_space_global_final"





def main():
    """
    路径跟踪仿真主函数。
    """
    print("开始路径跟踪仿真...")

    # 1. 加载配置和工作空间数据
    config = load_config('config/config.json')
    workspace_data_path = 'plots/workspace_points.npy'
    if not os.path.exists(workspace_data_path):
        print(f"错误: 工作空间数据文件未找到: {workspace_data_path}")
        print("请先运行 analysis/workspace_analysis.py 来生成数据。")
        return

    workspace_points = np.load(workspace_data_path)
    
    # 2. 基于工作空间数据，计算一个保守的目标路径
    center_x = np.mean(workspace_points[:, 0])
    center_y = np.mean(workspace_points[:, 1])
    center_z = np.mean(workspace_points[:, 2])
    
    radius = 0.4 * min(np.std(workspace_points[:, 0]), np.std(workspace_points[:, 1]))

    print("\n--- 基于工作空间数据的路径参数 ---")
    print(f"中心点 (X, Y, Z): ({center_x:.4f}, {center_y:.4f}, {center_z:.4f}) m")
    print(f"半径: {radius:.4f} m")
    print("------------------------------------")

    if radius < 0.001:
        print(f"错误: 计算出的路径半径 ({radius:.4f} m) 过小。")
        return

    target_poses = generate_circular_path(
        radius=radius, 
        num_points=50,
        center_x=center_x, 
        center_y=center_y, 
        center_z=center_z
    )

    actual_poses, required_forces, suffix = track_path_task_space(target_poses, config)

    print("\n--- 仿真循环结束 ---")

    if actual_poses:
        plot_results(target_poses[:len(actual_poses)], actual_poses, required_forces, title_suffix=suffix)
    else:
        print("没有成功求解的路径点，无法生成结果图。")

    print("路径跟踪仿真结束。")


if __name__ == "__main__":
    main()