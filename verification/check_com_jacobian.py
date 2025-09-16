# verification/check_com_jacobian.py
import numpy as np, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kinematics import forward_kinematics, calculate_com_jacobians_analytical
from src.utils.read_config import load_config
from scipy.optimize import approx_fprime

config = load_config('config/config.json')
q = np.array([0.1, 0.2, 0.05, 0.1, 0.02, 0.03])

J_anal = calculate_com_jacobians_analytical(q, config)  # expects dict with keys 'pss','cms1','cms2'
print("J_anal keys:", J_anal.keys())
for body in ['pss','cms1','cms2']:
    Ja = J_anal[body]
    print(f"\n{body} analytical shape:", Ja.shape)
    # compute numeric jacobian of z coordinate: f(q)=COM_z
    def fz(q_in, b=body):
        _, coms = forward_kinematics(q_in, config)
        return coms[b][2]  # z
    eps = 1e-6
    num_jac_z = approx_fprime(q, lambda qq: fz(qq), eps)
    # pick analytic z-row depending on Ja shape
    if Ja.shape[0] >= 3:
        # assume rows are x,y,z
        ana_jac_z = Ja[2, :]  
    else:
        # else maybe jac is transposed
        ana_jac_z = Ja[:, 2]
    print(f"{body} numeric z-jac: {num_jac_z}")
    print(f"{body} analytic z-jac: {ana_jac_z}")
    print("diff:", ana_jac_z - num_jac_z)
