# analysis/workspace_diagnostics.py
import numpy as np
from src.utils.read_config import load_config
from src.solver import solve_static_equilibrium_diff4
from src.statics import expand_diff4_to_motor8, calculate_drive_mapping, calculate_actuation_jacobian_analytical
from src.kinematics import forward_kinematics
import os

cfg = load_config('config/config.json')

# load or generate some diff4 samples used in the workspace run.
# If you have saved workspace diffs, load that instead; otherwise sample small set.
N = 200
# sample in same bounds used by workspace; try both uniform and center-focused
low, high = cfg['Bounds']['diff4_bounds']
diff4_samples = np.random.uniform(low, high, size=(N,4))

results = []
for i, diff4 in enumerate(diff4_samples):
    delta_l = expand_diff4_to_motor8(diff4, cfg)
    res = solve_static_equilibrium_diff4(np.zeros(6), diff4, cfg)
    q_eq = res.get('q_solution')
    ok = q_eq is not None
    tension_mask = None
    cond_J = None
    grad_norm = None
    if ok:
        # compute robot's delta_l_robot and per-cable stretch -> tensioned?
        delta_l_robot = calculate_drive_mapping(q_eq, cfg)
        stretch = delta_l - delta_l_robot
        # using smooth_max_zero derivative sign: tensioned where smooth_max_zero(stretch)>0
        # but easier: tensioned if stretch>1e-8 (positive indicates motor shortened relative to robot)
        tension_mask = (stretch > 1e-6)
        # jacobian condition
        J_act = calculate_actuation_jacobian_analytical(q_eq, cfg)
        try:
            cond_J = np.linalg.cond(J_act)
        except Exception:
            cond_J = np.nan
        # gradient norm
        from src.statics import calculate_gradient_disp_ctrl
        grad = calculate_gradient_disp_ctrl(q_eq, delta_l, cfg)
        grad_norm = np.linalg.norm(grad)
    results.append({
        'diff4': diff4,
        'ok': ok,
        'q': q_eq,
        'tension_mask': tension_mask,
        'cond_J': cond_J,
        'grad_norm': grad_norm
    })

# Summaries
oks = [r for r in results if r['ok']]
fails = [r for r in results if not r['ok']]
print(f"Total samples: {N}, OK: {len(oks)}, FAIL: {len(fails)}")
# tension stats
if oks:
    tensions = np.array([r['tension_mask'].astype(int) for r in oks])
    print("Tensioned counts per cable (mean over successes):", tensions.mean(axis=0))
    conds = np.array([r['cond_J'] for r in oks])
    print("Jacobian cond stats (successes): min, median, max =", np.nanmin(conds), np.nanmedian(conds), np.nanmax(conds))
    gradnorms = np.array([r['grad_norm'] for r in oks])
    print("Grad norm stats: min, median, max =", np.nanmin(gradnorms), np.nanmedian(gradnorms), np.nanmax(gradnorms))
else:
    print("No successful solves in sample - consider relaxing solver options or using robust fallback")
# save summary
np.save('analysis/workspace_diag_results.npy', results)
print("Saved analysis/workspace_diag_results.npy")
