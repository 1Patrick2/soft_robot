# verification/debug_pss_end_mode.py
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.kinematics import forward_kinematics
from src.statics import (
    calculate_elastic_potential_energy,
    calculate_gravity_potential_energy,
    calculate_drive_mapping,
    expand_diff4_to_motor8,
    smooth_max_zero,
    calculate_elastic_gradient_analytical,
    calculate_gravity_gradient_analytical,
    calculate_actuation_gradient_analytical
)

# Suppress scientific notation for clearer output
np.set_printoptions(suppress=True, precision=8)

print("--- Debugging pss_end Mode Energy and Gradient ---")

# 1. Load config and set to pss_end mode
params = load_config('config/config.json')
params['ModelOptions'] = {"cable_anchor_mode": "pss_end"}

# 2. Define a fixed test case
# Use a simple non-zero q to avoid singularities at q=0
q_test = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
diff4_test = np.array([0.05, 0, 0, 0])
delta_l_motor_test = expand_diff4_to_motor8(diff4_test, params)

print(f"Test Configuration q: {q_test}")
print(f"Test diff4 input: {diff4_test}")
print(f"Resulting delta_l_motor: {delta_l_motor_test}\n")

# 3. Calculate Energy Components
print("--- Energy Components ---")
U_elastic = calculate_elastic_potential_energy(q_test, params)
U_gravity = calculate_gravity_potential_energy(q_test, params)

# Cable energy components (extracted from total energy function)
k_cable = params['Drive_Properties']['cable_stiffness']
f_pre = params['Drive_Properties'].get('pretension_force_N', 0.0)
delta_l_robot = calculate_drive_mapping(q_test, params)
stretch = delta_l_motor_test - delta_l_robot
stretch_tensioned = smooth_max_zero(stretch)
U_cable = 0.5 * k_cable * np.sum(stretch_tensioned**2)
U_pretension = -f_pre * np.sum(delta_l_robot)
U_total = U_elastic + U_gravity + U_cable + U_pretension

print(f"  - U_elastic:    {U_elastic:.8f}")
print(f"  - U_gravity:    {U_gravity:.8f}")
print(f"  - U_cable:      {U_cable:.8f}")
print(f"  - U_pretension: {U_pretension:.8f}")
print(f"  - U_total:      {U_total:.8f}\n")

print("--- Intermediate Cable Values ---")
print(f"  - delta_l_robot: {delta_l_robot}")
print(f"  - stretch (motor - robot): {stretch}")
print(f"  - stretch_tensioned (after smooth_max_zero): {stretch_tensioned}\n")

# 4. Calculate Gradient Components
print("--- Gradient Components (d_Energy / d_q) ---")
grad_e = calculate_elastic_gradient_analytical(q_test, params)
grad_g = calculate_gravity_gradient_analytical(q_test, params)
grad_a = calculate_actuation_gradient_analytical(q_test, delta_l_motor_test, params)
grad_total = grad_e + grad_g + grad_a

print(f"  - Grad_Elastic:   {grad_e}")
print(f"  - Grad_Gravity:   {grad_g}")
print(f"  - Grad_Actuation: {grad_a}")
print(f"  - Grad_Total:     {grad_total}")

print("\n--- Debugging Complete ---")
