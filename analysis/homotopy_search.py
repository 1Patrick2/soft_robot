# analysis/homotopy_search.py
import numpy as np
from src.solver import solve_static_equilibrium_diff4
from src.utils.read_config import load_config
from src.statics import expand_diff4_to_motor8

cfg = load_config('config/config.json')

def homotopy_from_boundary(diff4_boundary, steps=20):
    q_guess = np.zeros(6)
    found = []
    for alpha in np.linspace(1.0, 0.0, steps):
        diff4_target = diff4_boundary * alpha
        res = solve_static_equilibrium_diff4(q_guess, diff4_target, cfg)
        q_eq = res.get('q_solution')
        if q_eq is None:
            print(f"Failed at alpha={alpha:.3f}")
            break
        found.append({'alpha': alpha, 'diff4': diff4_target.copy(), 'q': q_eq})
        q_guess = q_eq.copy()  # warm start
    return found

# usage example, pick a boundary sample (e.g. [0.12,0,0,0])
if __name__ == '__main__':
    diff4_b = np.array([0.12, 0.0, 0.0, 0.0])
    path = homotopy_from_boundary(diff4_b, steps=30)
    print("found length:", len(path))
    np.save('analysis/homotopy_path.npy', path)
