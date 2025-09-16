# run_ik_stability_test.py (V-Final.19 - multiprocessing with initializer)

import numpy as np
import time
import multiprocessing as mp
import os

# --- 核心依赖 ---
from src.utils.read_config import load_config
from src.outer_solver import solve_ik_pure_displacement

# --- 全局变量，用于在工作进程中存储数据 ---
CONFIG_PARAMS = None
T_TARGET = None

def init_worker(config, target):
    """每个工作进程启动时运行的初始化函数。"""
    global CONFIG_PARAMS, T_TARGET
    CONFIG_PARAMS = config
    T_TARGET = target

def worker_function(run_idx):
    """
    这是每个子进程将要执行的、独立的、无参数的工作函数。
    它从自己的全局空间读取配置和目标。
    """
    print(f"  [PID:{os.getpid()}] 开始运行第 {run_idx+1} 号测试...")
    # [V-Final 修复] 调用正确的求解器函数
    result = solve_ik_pure_displacement(T_TARGET, CONFIG_PARAMS)
    print(f"  [PID:{os.getpid()}] 完成第 {run_idx+1} 号测试。")
    
    # [V-Final 修复] 适配返回字典的格式，以兼容旧的统计逻辑
    # 将物理误差从 'error' 键，复制到 'physical_position_error' 键
    if result and result.get('success'):
        result['physical_position_error'] = result.get('error')
        result['x'] = result.get('delta_l')
        result['q_solution'] = result.get('q')
    else: # 确保失败时也有这个键，值为无穷大
        result['physical_position_error'] = np.inf

    return result

def main():
    # --- 1. 在主进程中加载配置 ---
    config_params = load_config('config/config.json')
    robot_initial_length = (
        config_params['Geometry']['PSS_initial_length'] +
        config_params['Geometry']['CMS_proximal_length'] +
        config_params['Geometry']['CMS_distal_length']
    )
    np.set_printoptions(precision=6, suppress=True)
    target_x_offset = 0.03
    t_target = np.array([
        [1, 0, 0, target_x_offset],
        [0, 1, 0, 0.0],
        [0, 0, 1, robot_initial_length - 0.01],
        [0, 0, 0, 1]
    ])
    print(f"\n目标位姿 T_target:\n{t_target}")

    # --- 2. [核心升级] 使用带初始化器的进程池 ---
    num_runs = 5
    print(f"\n--- 开始最终高精度IK求解 (使用 multiprocessing.Pool, {num_runs} 个进程) ---")
    start_time = time.time()
    
    # a. 创建一个进程池，并使用`initializer`和`initargs`一次性传递大数据
    with mp.Pool(processes=num_runs, initializer=init_worker, initargs=(config_params, t_target)) as pool:
        # b. pool.map现在只传递简单的任务索引，没有大数据序列化的开销
        all_results = pool.map(worker_function, range(num_runs))

    end_time = time.time()
    print(f"\n并行统计测试完成，总耗时: {end_time - start_time:.2f} 秒")
    
    # --- 3. 结果分析 (逻辑不变) ---
    successful_results = [res for res in all_results if res and hasattr(res, 'physical_position_error') and res.physical_position_error is not None]

    if not successful_results:
        print("\n❌ [诊断失败] 所有独立运行均未能找到有效解。\n")
        exit()
    
    best_overall_result = min(successful_results, key=lambda res: res.physical_position_error)
    
    print("\n" + "="*50)
    print("--- 稳定性测试最终总结 ---")
    print(f"成功运行次数: {len(successful_results)}/{num_runs}")
    
    all_errors = [res.physical_position_error * 1000 for res in successful_results]
    print(f"所有成功解的误差 (mm): {[f'{err:.2f}' for err in all_errors]}")
    
    print("\n--- 最佳结果详情 ---")
    print(f"成功标志 (可能不代表收敛): {best_overall_result.success}")
    print(f"信息: {best_overall_result.message}")
    print(f"最终物理位置误差: {best_overall_result.physical_position_error * 1000:.3f} mm")
    print(f"求解得到的构型 q_solution:\n{best_overall_result.q_solution}")
    print(f"求解得到的驱动力 tau_solution:\n{best_overall_result.x}")
    
if __name__ == '__main__':
    main()
