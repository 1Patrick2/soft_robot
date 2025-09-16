import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.read_config import load_config
from src.statics import calculate_total_potential_energy_disp_ctrl

def analyze_energy_landscape():
    """
    Analyzes and plots slices of the potential energy landscape to diagnose solver issues.
    """
    print("--- Analyzing Potential Energy Landscape ---")
    
    # 1. Load configuration and set parameters
    config = load_config('config/config.json')
    delta_l_zero = np.zeros(8)
    
    # --- Analysis 1: Energy vs. PSS Curvature (kp) ---
    print("\nAnalyzing: Energy vs. PSS Curvature (kp)")
    kp_range = np.linspace(-10, 10, 200)
    energies_kp = []
    q_base = np.zeros(6)

    for kp in tqdm(kp_range, desc="Scanning kp"):
        q = q_base.copy()
        q[0] = kp
        energy = calculate_total_potential_energy_disp_ctrl(q, delta_l_zero, config)
        energies_kp.append(energy)

    # --- Analysis 2: Energy vs. PSS Angle (phip) ---
    print("\nAnalyzing: Energy vs. PSS Angle (phip)")
    phip_range = np.linspace(-np.pi, np.pi, 200)
    energies_phip = []
    # Base q with a small non-zero curvature to avoid singularity for phi
    q_base_bent = np.array([0.1, 0, 0, 0, 0, 0]) 

    for phip in tqdm(phip_range, desc="Scanning phip"):
        q = q_base_bent.copy()
        q[1] = phip
        energy = calculate_total_potential_energy_disp_ctrl(q, delta_l_zero, config)
        energies_phip.append(energy)

    # --- Plotting ---
    print("\nPlotting results...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle("Potential Energy Landscape Slices (delta_l = 0, with Gravity)")

    # Plot for kp
    ax1.plot(kp_range, energies_kp, '-o', markersize=3)
    ax1.set_title("Energy vs. PSS Curvature (kp) at q=[kp, 0, 0, 0, 0, 0]")
    ax1.set_xlabel("PSS Curvature kp (1/m)")
    ax1.set_ylabel("Total Potential Energy (J)")
    ax1.grid(True)

    # Plot for phip
    ax2.plot(phip_range, energies_phip, '-o', markersize=3, color='r')
    ax2.set_title("Energy vs. PSS Angle (phip) at q=[0.1, phip, 0, 0, 0, 0]")
    ax2.set_xlabel("PSS Angle phip (rad)")
    ax2.set_ylabel("Total Potential Energy (J)")
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    plot_path = 'plots/debug_energy_landscape.png'
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_filename = os.path.join(project_root, plot_path)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename)
    
    print(f"Plot saved to {os.path.abspath(output_filename)}")
    # plt.show() # Comment out to prevent hanging in non-interactive environments


if __name__ == '__main__':
    analyze_energy_landscape()
