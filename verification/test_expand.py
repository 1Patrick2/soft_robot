# verification/test_expand.py
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.statics import expand_diff4_to_motor8
from src.utils.read_config import load_config

# Suppress scientific notation for clearer output
np.set_printoptions(suppress=True)

print("--- Running test_expand.py with CURRENT implementation ---")
params = load_config('config/config.json')

cases = [
  np.array([0.01, 0, 0, 0]),
  np.array([0, 0.01, 0, 0]),
  np.array([0, 0, 0.01, 0]),
  np.array([0, 0, 0, 0.01]),
  np.array([0.01, 0.01, 0, 0])
]
for d in cases:
    delta_l = expand_diff4_to_motor8(d, params)
    print("diff4:", d, "-> motor8:", delta_l)