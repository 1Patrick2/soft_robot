# verification/verify_actuation_jacobian.py
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.statics import calculate_drive_mapping, calculate_actuation_jacobian_analytical

# Suppress scientific notation for clearer output
np.set_printoptions(suppress=True, precision=6)

print("--- Running verify_actuation_jacobian.py (Step 3 from test.md) ---")

params = load_config('config/config.json')

def numerical_actuation_jacobian(q, params, eps=1e-6):
    base = calculate_drive_mapping(q, params)
    J = np.zeros((8,6))
    for i in range(6):
        dq = np.zeros(6); dq[i] = eps
        plus = calculate_drive_mapping(q + dq, params)
        minus = calculate_drive_mapping(q - dq, params)
        J[:,i] = (plus - minus)/(2*eps)
    return J

# Test at a non-zero configuration to avoid singularities
q_test = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

print(f"\n--- Test Condition ---")
print(f"  - Test configuration q: {q_test}")

J_num = numerical_actuation_jacobian(q_test, params)
J_ana = calculate_actuation_jacobian_analytical(q_test, params)

# The analytical jacobian function in statics.py has a dependency on kinematics.py,
# which might not be in the path when run directly. The sys.path.append should fix this.

print("\n--- Numerical Jacobian (J_num) ---")
print(J_num)

print("\n--- Analytical Jacobian (J_ana) ---")
print(J_ana)

diff_norm = np.linalg.norm(J_num - J_ana)
ref_norm = np.linalg.norm(J_num)
rel_diff = diff_norm / (ref_norm + 1e-12)

print("\n--- Comparison ---")
print(f"  Norm of Numerical J: {ref_norm:.6f}")
print(f"  Norm of Analytical J: {np.linalg.norm(J_ana):.6f}")
print(f"  Absolute Difference Norm: {diff_norm:.6f}")
print(f"  Relative Difference: {rel_diff:.6e}")

# As per test.md, a relative difference < 1e-4 is a good sign.
if rel_diff < 1e-4:
    print("\n✅ Test Passed: Analytical and numerical Jacobians are consistent.")
else:
    print("\n❌ Test Failed: Significant difference between analytical and numerical Jacobians.")

print("\n--- Jacobian verification finished. ---")
