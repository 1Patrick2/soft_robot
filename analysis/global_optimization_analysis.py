import numpy as np
import matplotlib.pyplot as plt
import time

# 本地模块
from src.utils.read_config import load_config
from src.kinematics import forward_kinematics
from src.solver import solve_static_equilibrium
from src.outer_solver import solve_inverse_kinematics

def run_path_tracking(params, use_global_search=False, num_starts=10):
    """
    执行路径跟踪的核心函数。

    Args:
        params: 机器人配置参数。
        use_global_search: 是否启用多起点全局搜索策略。
        num_starts: 在全局搜索中使用的随机起点的数量。

    Returns:
        一个包含误差和时间的字典。
    """
    # 1. 生成目标路径 (一个在x-y平面上的圆环)
    num_points = 50
    radius = 0.05 # 5 cm
    center_z = 0.15 # 15 cm
    angles = np.linspace(0, 2 * np.pi, num_points)
    target_positions = np.array([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.full(num_points, center_z)
    ]).T

    # 2. 初始化
    q_guess = np.array(params['Initial_State']['q0'])
    tau_guess = {'tensions_short': np.full(4, 5.0), 'tensions_long': np.full(4, 5.0)}
    last_tau = tau_guess

    actual_path = []
    errors = []
    total_time = 0

    # 3. 遍历路径上的每个目标点
    for i, p_target in enumerate(target_positions):
        print(f"正在处理路径点 {i+1}/{num_points}...")
        T_target = np.identity(4)
        T_target[:3, 3] = p_target

        start_time = time.time()

        if use_global_search:
            # --- 多起点全局搜索策略 ---
            best_result = None
            min_cost = float('inf')

            # 生成N个随机初始猜测
            random_starts = [np.random.rand(8) * 20 for _ in range(num_starts)]
            # 将用户提供的初始猜测也加入搜索列表
            initial_guesses = [tau_guess] + [
                {'tensions_short': s[:4], 'tensions_long': s[4:]} for s in random_starts
            ]

            for start_guess in initial_guesses:
                result = solve_inverse_kinematics(
                    T_target, start_guess, q_guess, params, 
                    previous_tau=last_tau, use_jacobian=True
                )
                if result.success and result.cost < min_cost:
                    min_cost = result.cost
                    best_result = result
            
            ik_result = best_result
        else:
            # --- 标准单次求解策略 ---
            ik_result = solve_inverse_kinematics(
                T_target, tau_guess, q_guess, params, 
                previous_tau=last_tau, use_jacobian=True
            )

        total_time += time.time() - start_time

        if ik_result and ik_result.success:
            optimal_tau_flat = ik_result.x
            last_tau = {'tensions_short': optimal_tau_flat[:4], 'tensions_long': optimal_tau_flat[4:]}
            
            q_final = solve_static_equilibrium(q_guess, last_tau, params)
            T_final, _ = forward_kinematics(q_final, params)
            p_actual = T_final[:3, 3]
            
            actual_path.append(p_actual)
            error = np.linalg.norm(p_actual - p_target)
            errors.append(error)
        else:
            print(f"  - 警告: 路径点 {i+1} 求解失败。  ")
            actual_path.append(actual_path[-1] if actual_path else p_target) # 保持路径长度
            errors.append(np.linalg.norm(actual_path[-1] - p_target))

    # 4. 返回结果
    return {
        'errors': errors,
        'mean_error': np.mean(errors),
        'total_time': total_time,
        'target_path': target_positions,
        'actual_path': np.array(actual_path)
    }

if __name__ == '__main__':
    config = load_config('config/config.json')

    # --- 运行标准求解 ---
    print("--- 开始标准路径跟踪 (单次求解) ---")
    results_standard = run_path_tracking(config, use_global_search=False)
    print(f"标准求解完成。平均误差: {results_standard['mean_error']*1000:.2f} mm, 总耗时: {results_standard['total_time']:.2f} s")

    # --- 运行多起点全局搜索 ---
    print("\n--- 开始多起点全局搜索路径跟踪 ---")
    results_global = run_path_tracking(config, use_global_search=True, num_starts=5) # 使用5个随机起点以节省测试时间
    print(f"全局搜索完成。平均误差: {results_global['mean_error']*1000:.2f} mm, 总耗时: {results_global['total_time']:.2f} s")

    # --- 绘图对比 ---
    fig = plt.figure(figsize=(12, 10))
    
    # 3D 视图
    ax_3d = fig.add_subplot(2, 2, (1, 2), projection='3d')
    ax_3d.plot(results_standard['target_path'][:, 0], results_standard['target_path'][:, 1], results_standard['target_path'][:, 2], 'g--', label='Target Path')
    ax_3d.plot(results_standard['actual_path'][:, 0], results_standard['actual_path'][:, 1], results_standard['actual_path'][:, 2], 'b-o', markersize=3, label=f"Standard (Err: {results_standard['mean_error']*1000:.2f} mm)")
    ax_3d.plot(results_global['actual_path'][:, 0], results_global['actual_path'][:, 1], results_global['actual_path'][:, 2], 'r-o', markersize=3, label=f"Global Search (Err: {results_global['mean_error']*1000:.2f} mm)")
    ax_3d.set_xlabel('X (m)'); ax_3d.set_ylabel('Y (m)'); ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Path Tracking Comparison')
    ax_3d.legend()
    ax_3d.view_init(elev=30, azim=-60)

    # XY 视图
    ax_xy = fig.add_subplot(2, 2, 3)
    ax_xy.plot(results_standard['target_path'][:, 0], results_standard['target_path'][:, 1], 'g--')
    ax_xy.plot(results_standard['actual_path'][:, 0], results_standard['actual_path'][:, 1], 'b-o', markersize=3)
    ax_xy.plot(results_global['actual_path'][:, 0], results_global['actual_path'][:, 1], 'r-o', markersize=3)
    ax_xy.set_xlabel('X (m)'); ax_xy.set_ylabel('Y (m)'); ax_xy.set_title('XY Plane View'); ax_xy.grid(True); ax_xy.axis('equal')

    # XZ 视图
    ax_xz = fig.add_subplot(2, 2, 4)
    ax_xz.plot(results_standard['target_path'][:, 0], results_standard['target_path'][:, 2], 'g--')
    ax_xz.plot(results_standard['actual_path'][:, 0], results_standard['actual_path'][:, 2], 'b-o', markersize=3)
    ax_xz.plot(results_global['actual_path'][:, 0], results_global['actual_path'][:, 2], 'r-o', markersize=3)
    ax_xz.set_xlabel('X (m)'); ax_xz.set_ylabel('Z (m)'); ax_xz.set_title('XZ Plane View'); ax_xz.grid(True); ax_xz.axis('equal')

    plt.tight_layout()
    plt.savefig('plots/global_search_path_comparison.png')
    print("\n对比图已保存至 'plots/global_search_path_comparison.png'")
    plt.show()
