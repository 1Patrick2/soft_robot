import numpy as np
import sys
import os
import time
import logging
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.outer_solver import solve_ik_globally
from src.kinematics import forward_kinematics
from src.utils.read_config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def run_ik_validation(num_samples=50):
    """
    Performs a batch validation of the inverse kinematics solver.
    1. Loads known reachable points from a workspace analysis file.
    2. Uses them as targets for the IK solver.
    3. Reports statistics on success rate, accuracy, and performance.
    """
    print("--- Running Inverse Kinematics Batch Validation ---")
    
    # 1. Load Configuration and Workspace Data
    try:
        config = load_config('config/config.json')
        workspace_data = np.load('plots/workspace_points_homotopy.npy', allow_pickle=True)
        if len(workspace_data) < num_samples:
            print(f"❌ ERROR: Not enough data in workspace file. Found {len(workspace_data)}, need {num_samples}.")
            return
        print(f"✅ Loaded {len(workspace_data)} points from workspace_points_homotopy.npy")
    except FileNotFoundError:
        print("❌ ERROR: workspace_points_homotopy.npy not found. Please run workspace analysis first.")
        return

    # 2. Select Random Targets
    np.random.seed(0) # for reproducibility
    random_indices = np.random.choice(len(workspace_data), size=num_samples, replace=False)
    targets = workspace_data[random_indices]

    # 3. Run Batch Validation
    results = []
    for i in tqdm(range(num_samples), desc="IK Validation Progress"):
        target_data = targets[i]
        q_truth = target_data['q_eq']
        
        # The target pose is the one generated from the known good q
        target_pose, _ = forward_kinematics(q_truth, config)

        start_time = time.time()
        ik_result = solve_ik_globally(target_pose, config)
        end_time = time.time()

        ik_result['solve_time_s'] = end_time - start_time
        results.append(ik_result)

    # 4. Analyze and Report Results
    success_count = sum(1 for r in results if r['success'])
    success_rate = (success_count / num_samples) * 100
    
    successful_pos_errors = [r['error_mm'] for r in results if r['success']]
    successful_solve_times = [r['solve_time_s'] for r in results if r['success']]

    mean_pos_error = np.mean(successful_pos_errors) if successful_pos_errors else -1
    median_pos_error = np.median(successful_pos_errors) if successful_pos_errors else -1
    max_pos_error = np.max(successful_pos_errors) if successful_pos_errors else -1
    mean_solve_time = np.mean(successful_solve_times) if successful_solve_times else -1

    print("\n--- IK Validation Summary ---")
    print(f"  - Total Targets: {num_samples}")
    print(f"  - Success Rate: {success_rate:.2f}% ({success_count}/{num_samples})")
    print("\n--- Statistics for Successful Solves ---")
    print(f"  - Position Error (mm):")
    print(f"    - Mean:   {mean_pos_error:.4f}")
    print(f"    - Median: {median_pos_error:.4f}")
    print(f"    - Max:    {max_pos_error:.4f}")
    print(f"  - Solve Time (s):")
    print(f"    - Mean:   {mean_solve_time:.2f}")

    if success_rate < 90.0:
        print("\n- ❌ FINAL VERDICT: FAILED. Success rate is below 90%.")
    elif mean_pos_error > 2.0:
        print("\n- ⚠️ FINAL VERDICT: PASSED WITH WARNINGS. Mean position error is > 2.0 mm.")
    else:
        print("\n- ✅✅✅ FINAL VERDICT: PASSED. The outer solver is robust and accurate.")

if __name__ == '__main__':
    run_ik_validation()
