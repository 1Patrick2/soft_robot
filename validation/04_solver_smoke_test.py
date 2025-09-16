import numpy as np
import sys
import os
import logging
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solver import solve_static_equilibrium_disp_ctrl
from src.utils.read_config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def run_test_scenario(scenario_name, num_samples, params, q_guess_func, delta_l_func):
    print(f"\n--- Running Test Scenario: {scenario_name} ---")
    success_count = 0
    iterations_list = []

    for _ in tqdm(range(num_samples), desc=scenario_name):
        q_guess = q_guess_func()
        delta_l = delta_l_func()
        
        solve_result = solve_static_equilibrium_disp_ctrl(q_guess, delta_l, params)
        
        if solve_result["q_solution"] is not None:
            success_count += 1
            iterations_list.append(solve_result["result"].nit)

    success_rate = (success_count / num_samples) * 100
    avg_iterations = np.mean(iterations_list) if iterations_list else 0

    print(f"\n--- Results for {scenario_name} ---")
    print(f"  Success Rate: {success_rate:.2f}% ({success_count}/{num_samples})")
    print(f"  Average Iterations on Success: {avg_iterations:.2f}")
    
    if success_rate < 95.0:
        print("  - ❌ WARNING: Success rate is below 95%!")
    else:
        print("  - ✅ PASSED: Solver is robust in this scenario.")
    return success_rate

if __name__ == '__main__':
    config = load_config('config/config.json')

    # --- Scenario 1: Zero Gravity, Zero Drive ---
    params_no_gravity = config.copy()
    params_no_gravity['Mass'] = config['Mass'].copy()
    params_no_gravity['Mass']['pss_kg'] = 0
    params_no_gravity['Mass']['cms_proximal_kg'] = 0
    params_no_gravity['Mass']['cms_distal_kg'] = 0
    run_test_scenario(
        "Zero Gravity, Zero Drive",
        num_samples=50,
        params=params_no_gravity,
        q_guess_func=lambda: np.full(6, 1e-6),
        delta_l_func=lambda: np.zeros(8)
    )

    # --- Scenario 2: Small Random Drives ---
    run_test_scenario(
        "Small Random Drives",
        num_samples=100,
        params=config,
        q_guess_func=lambda: np.random.rand(6) * 0.1 - 0.05, # Small random guess
        delta_l_func=lambda: np.random.randn(8) * 0.003 # up to ~3mm
    )

    # --- Scenario 3: Large Boundary Drives ---
    run_test_scenario(
        "Large Boundary Drives",
        num_samples=100,
        params=config,
        q_guess_func=lambda: np.random.rand(6) * 2 - 1, # Wider random guess
        delta_l_func=lambda: np.random.randn(8) * 0.06 # up to ~60mm
    )

    print("\n--- Smoke Test Complete ---")
