# verification/inspect_units_and_values.py
import numpy as np, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.read_config import load_config
from src.kinematics import forward_kinematics
from src.statics import calculate_drive_mapping

config = load_config('config/config.json')
print("=== config core values ===")
print("g:", config['Mass']['g'])
print("masses (pss,c1,c2):", config['Mass']['pss_kg'], config['Mass']['cms_proximal_kg'], config['Mass']['cms_distal_kg'])
print("geometry lengths (m):", config['Geometry']['PSS_initial_length'],
      config['Geometry']['CMS_proximal_length'], config['Geometry']['CMS_distal_length'])
print("short/long radii (m):", config['Geometry']['short_lines']['diameter_m']/2,
      config['Geometry']['long_lines']['diameter_m']/2)

# forward kinematics test
q = np.array([0.1, 0.2, 0.05, 0.1, 0.02, 0.03])
T_final, coms = forward_kinematics(q, config)
print("\nCOM positions (from FK):")
for k,v in coms.items():
    print(k, v, "units? (look at magnitude; if ~0.0x it's meters; if ~xx it's mm)")

# drive mapping sanity
dl = calculate_drive_mapping(q, config)
print("\ndrive mapping:", dl)
