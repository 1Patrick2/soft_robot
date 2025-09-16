import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 将项目根目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.outer_solver import solve_inverse_kinematics, calculate_pose_error
from src.kinematics import forward_kinematics
from src.solver import solve_static_equilibrium
from src.statics import calculate_elastic_potential_energy
from src.utils.read_config import load_config

def run_tuning_analysis():
    """
    固定pose_error_weight，扫描不同的energy_weight值。
    """
    print("--- 开始权重调优分析 (固定高位姿权重) ---")
    config = load_config('config/config.json')

    target_pose = np.array([
        [1, 0, 0, 0.04],
        [0, 1, 0, 0.0],
        [0, 0, 1, -0.05],
        [0, 0, 0, 1]
    ])
    print(f"目标位姿设定完成。")

    # 定义待测试的 energy_weight 值
    energy_weights = np.logspace(-6, 2, num=20)
    # 固定一个较高的位姿误差权重
    pose_weight = 100.0
    print(f"将测试 {len(energy_weights)} 个energy_weight, 固定pose_error_weight={pose_weight}")

    results = []
    q_guess = np.array(config['Initial_State']['q0'])
    initial_tau_guess = {
        'tensions_short': np.full(4, 1.0),
        'tensions_long': np.full(4, 1.0)
    }

    for i, e_weight in enumerate(energy_weights):
        print(f"\n[测试 {i+1}/{len(energy_weights)}] energy_weight = {e_weight:.2e}")
        
        ik_result = solve_inverse_kinematics(
            target_pose,
            initial_tau_guess=initial_tau_guess,
            q_guess=q_guess,
            params=config,
            use_jacobian=True,
            smoothing_weight=0.0,
            energy_weight=e_weight,
            pose_error_weight=pose_weight
        )

        if ik_result.success and ik_result.status > 0:
            final_tau_flat = ik_result.x
            tensions_final = {'tensions_short': final_tau_flat[:4], 'tensions_long': final_tau_flat[4:]}
            q_solution = solve_static_equilibrium(q_guess, tensions_final, config)
            actual_pose, _ = forward_kinematics(q_solution, config)
            
            final_pose_error = np.linalg.norm(calculate_pose_error(actual_pose, target_pose))
            final_elastic_energy = calculate_elastic_potential_energy(q_solution, config)
            
            print(f"  - 求解成功。")
            print(f"  - 最终位姿误差: {final_pose_error:.6f}")
            print(f"  - 最终弹性势能: {final_elastic_energy:.6f}")
            
            results.append({
                'weight': e_weight,
                'error': final_pose_error,
                'energy': final_elastic_energy
            })
        else:
            print(f"  - 求解失败: {ik_result.message}")

    return results

def plot_tuning_results(results):
    """
    将调优结果可视化。
    """
    if not results:
        print("没有有效的分析结果可供绘图。")
        return

    print("\n--- 正在绘制调优结果图 ---")
    weights = [r['weight'] for r in results]
    errors = [r['error'] for r in results]
    energies = [r['energy'] for r in results]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    color = 'tab:red'
    ax1.set_xlabel('Energy Weight (log scale)')
    ax1.set_ylabel('Pose Error (norm)', color=color)
    ax1.loglog(weights, errors, 'o-', color=color, label='Pose Error')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which="both", ls="--")

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Elastic Energy (J)', color=color)
    ax2.loglog(weights, energies, 's-', color=color, label='Elastic Energy')
    ax2.tick_params(axis='y', labelcolor=color)

    try:
        errors_norm = np.array(errors) / np.max(errors)
        energies_norm = np.array(energies) / np.max(energies)
        tradeoff_metric = errors_norm * energies_norm
        best_idx = np.argmin(tradeoff_metric)
        best_weight = weights[best_idx]
        best_error = errors[best_idx]
        best_energy = energies[best_idx]

        ax1.axvline(best_weight, color='green', linestyle='--', label=f'Best Trade-off (w={best_weight:.2e})')
        ax1.plot(best_weight, best_error, 'g*', markersize=15)
        ax2.plot(best_weight, best_energy, 'g*', markersize=15)
    except (ValueError, IndexError) as e:
        print(f"无法计算最佳平衡点: {e}")

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    fig.suptitle('Pose Error vs. Elastic Energy Trade-off (Pose Weight = 100)', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = "plots/weight_tuning_analysis.png"
    plt.savefig(output_path)
    print(f"分析图已保存到: {output_path}")
    plt.close()

if __name__ == "__main__":
    analysis_results = run_tuning_analysis()
    plot_tuning_results(analysis_results)
    print("--- 调优分析完成 ---")