import numpy as np
import sys
import os
import logging

# --- 核心依赖 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.read_config import load_config
from src.solver import solve_static_equilibrium_disp_ctrl

# =====================================================================
# === "真实性检验" 脚本
# =====================================================================

def run_truth_test(num_samples_to_test=100):
    """
    从已有的工作空间数据中随机抽取样本，并用更严格的求解器设置重新求解，
    以验证我们之前快速得到的解的"真实性"。
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    np.set_printoptions(precision=8, suppress=True)

    # --- 1. 加载数据 ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'plots', 'workspace_points_full.npy')
    
    logging.info(f"Loading workspace data from {data_path}...")
    if not os.path.exists(data_path):
        logging.error("Workspace data file not found. Please run workspace_analysis.py first.")
        return

    try:
        workspace_data = np.load(data_path, allow_pickle=True)
        if len(workspace_data) == 0:
            logging.error("Workspace data file is empty.")
            return
        logging.info(f"Successfully loaded {len(workspace_data)} data points.")
    except Exception as e:
        logging.error(f"Failed to load or parse workspace data: {e}")
        return

    # --- 2. 设置检验参数 ---
    config = load_config(os.path.join(project_root, 'config', 'config.json'))
    
    # 定义一个极其严格的求解器设置，用于"求真"
    strict_solver_options = {'ftol': 1e-12, 'maxiter': 2000}
    
    # 从总数据中随机抽取N个样本进行检验
    if len(workspace_data) < num_samples_to_test:
        logging.warning(f"Requested {num_samples_to_test} samples, but only {len(workspace_data)} available. Testing all.")
        num_samples_to_test = len(workspace_data)
    
    sample_indices = np.random.choice(len(workspace_data), num_samples_to_test, replace=False)
    test_samples = workspace_data[sample_indices]

    logging.info(f"--- Starting Truth Test on {num_samples_to_test} random samples ---")

    # --- 3. 执行检验循环 ---
    errors = []
    failed_refinements = 0

    for i, sample in enumerate(test_samples):
        _, delta_l_fast, q_fast = sample
        
        logging.info(f"[Sample {i+1}/{num_samples_to_test}] Refining solution for a given delta_l...")
        
        # 使用 "快速解" 作为 "严格解" 的初始猜测
        q_refined = solve_static_equilibrium_disp_ctrl(
            q_guess=q_fast, 
            delta_l_motor=delta_l_fast, 
            params=config,
            solver_options=strict_solver_options
        )

        if q_refined is not None:
            # 计算 "快速解" 和 "严格解" 之间的欧氏距离差异
            error = np.linalg.norm(q_refined - q_fast)
            errors.append(error)
            logging.info(f"  -> Refinement successful. Difference between fast and strict solution: {error:.6e}")
        else:
            failed_refinements += 1
            logging.warning(f"  -> Refinement FAILED. The strict solver could not converge from this point.")

    # --- 4. 报告最终结果 ---
    logging.info("--- Truth Test Finished --- ")
    
    if not errors:
        logging.error("No successful refinements to analyze. All strict solves failed.")
        return

    errors = np.array(errors)
    successful_refinements = len(errors)

    print("\n================= TRUTH TEST SUMMARY =================")
    print(f"Total samples tested: {num_samples_to_test}")
    print(f"✅ Successful refinements: {successful_refinements}")
    print(f"❌ Failed refinements:     {failed_refinements}")
    print("-----------------------------------------------------")
    print("Difference between 'fast' and 'strict' solutions (L2 Norm of q):")
    print(f"  - Average Error: {np.mean(errors):.6e}")
    print(f"  - Median Error:  {np.median(errors):.6e}")
    print(f"  - Max Error:     {np.max(errors):.6e}")
    print(f"  - Min Error:     {np.min(errors):.6e}")
    print("=====================================================\n")

    # 根据结果给出一个结论
    if np.mean(errors) < 1e-4 and failed_refinements / num_samples_to_test < 0.1:
        print("CONCLUSION: ✅ The fast solutions are trustworthy. The differences are negligible.")
    else:
        print("CONCLUSION: ⚠️ The fast solutions may not be reliable. The average error or failure rate is significant.")

if __name__ == '__main__':
    run_truth_test()
