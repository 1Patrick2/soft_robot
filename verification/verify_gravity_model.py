import numpy as np
import sys
import os

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.statics import calculate_gravity_potential_energy, calculate_gravity_gradient_analytical

def numerical_gradient(f, x, h=1e-7):
    """
    Computes the gradient of a function f at point x using central differences.
    """
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus_h = np.copy(x).astype(float)
        x_plus_h[i] += h
        
        x_minus_h = np.copy(x).astype(float)
        x_minus_h[i] -= h
        
        grad[i] = (f(x_plus_h) - f(x_minus_h)) / (2 * h)
    return grad

def main():
    print("--- Verification of Gravity Model Gradient ---")
    
    # 1. Load configuration
    config = load_config('config/config.json')
    np.set_printoptions(precision=8, suppress=True)

    # 2. Define a non-trivial test configuration q
    # A bent and twisted configuration
    q_test = np.array([5.0, 0.5, 2.0, -1.0, 1.0, 1.5])
    print(f"Test configuration q = {q_test}\n")

    # 3. Define the objective function for numerical gradient calculation
    def gravity_potential_energy_func(q):
        return calculate_gravity_potential_energy(q, config)

    # 4. Calculate numerical gradient (the "truth")
    print("Calculating numerical gradient (truth)...")
    grad_numerical = numerical_gradient(gravity_potential_energy_func, q_test)
    print(f"  - Numerical Gradient = {grad_numerical}\n")

    # 5. Calculate analytical gradient (our implementation)
    print("Calculating analytical gradient (our implementation)...")
    grad_analytical = calculate_gravity_gradient_analytical(q_test, config)
    print(f"  - Analytical Gradient = {grad_analytical}\n")

    # 6. Compare the results
    print("--- Comparison ---")
    error_abs = np.abs(grad_analytical - grad_numerical)
    error_norm = np.linalg.norm(grad_analytical - grad_numerical)
    
    print(f"Absolute Error Vector = {error_abs}")
    print(f"Norm of Error Vector = {error_norm:.6e}")

    # 7. Verdict
    if error_norm < 1e-5:
        print("\n✅✅✅ Verification PASSED: Analytical gradient matches the numerical truth.")
    else:
        print("\n❌❌❌ Verification FAILED: Analytical gradient does NOT match the numerical truth.")
        print("This strongly suggests a bug in 'calculate_gravity_gradient_analytical' or its dependency 'calculate_com_jacobians_analytical'.")

if __name__ == '__main__':
    main()
