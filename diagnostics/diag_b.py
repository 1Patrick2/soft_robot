# 诊断脚本B：从随机κ采样得到Δl_map（映射可行集）
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.cosserat.statics import calculate_drive_mapping
from src.cosserat.kinematics import discretize_robot
from src.utils.read_config import load_config

print("Running Diagnostic B: Mapping from kappa-space to delta_l-space...")
cfg = load_config('config/config.json')
_, (n_pss,n_cms1,n_cms2) = discretize_robot(cfg)
num_el = n_pss + n_cms1 + n_cms2
M = 2000
dl_map = []
for i in range(M):
    # scale can be tuned. 0.2 is a moderate amount of random curvature.
    k_rand = np.random.randn(3, num_el) * 0.2  
    # We need to call the correct function from statics
    # The function in test.md is old. The actual function is calculate_actuation_potential_and_gradient
    # No, wait, the function is calculate_drive_mapping. Let's check statics.py
    # Reading statics.py... ah, it seems calculate_drive_mapping is indeed the one.
    dl_robot = calculate_drive_mapping(k_rand, cfg)
    dl_map.append(dl_robot)
dl_map = np.array(dl_map)

os.makedirs('diagnostics', exist_ok=True)
np.save('diagnostics/dl_map.npy', dl_map)
print(f"Saved dl_map.npy (shape: {dl_map.shape})")
