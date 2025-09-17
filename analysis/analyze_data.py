import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_q_distribution(data_filepath):
    """
    Analyzes the distribution of the configuration vector q, especially kc1 and kc2,
    from a saved workspace data file.
    """
    if not os.path.exists(data_filepath):
        print(f"Error: Data file not found at {data_filepath}")
        return

    try:
        data = np.load(data_filepath, allow_pickle=True)
    except Exception as e:
        print(f"Error loading data file: {e}")
        return

    if data.size == 0:
        print("Error: Data file is empty.")
        return

    # Extract q_eq vectors
    q_eqs = np.array([item['q_eq'] for item in data])

    kp = q_eqs[:, 0]
    phip = q_eqs[:, 1]
    kc1 = q_eqs[:, 2]
    phic1 = q_eqs[:, 3]
    kc2 = q_eqs[:, 4]
    phic2 = q_eqs[:, 5]

    print(f"--- Bending Analysis for: {os.path.basename(data_filepath)} ---")
    print(f"Total points analyzed: {len(q_eqs)}")
    print("\n--- Statistics for kc1 (Proximal CMS) ---")
    print(f"  Mean:   {np.mean(kc1):.4f}")
    print(f"  Std Dev:{np.std(kc1):.4f}")
    print(f"  Min:    {np.min(kc1):.4f}")
    print(f"  Max:    {np.max(kc1):.4f}")
    print(f"  Median: {np.median(kc1):.4f}")

    print("\n--- Statistics for kc2 (Distal CMS) ---")
    print(f"  Mean:   {np.mean(kc2):.4f}")
    print(f"  Std Dev:{np.std(kc2):.4f}")
    print(f"  Min:    {np.min(kc2):.4f}")
    print(f"  Max:    {np.max(kc2):.4f}")
    print(f"  Median: {np.median(kc2):.4f}")
    
    # Plotting histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f'Distribution of CMS Curvatures ({os.path.basename(data_filepath)})')

    axes[0].hist(kc1, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('kc1 (Proximal CMS)')
    axes[0].set_xlabel('Curvature (rad/m)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].hist(kc2, bins=30, color='salmon', edgecolor='black')
    axes[1].set_title('kc2 (Distal CMS)')
    axes[1].set_xlabel('Curvature (rad/m)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plot_filename = os.path.join(os.path.dirname(data_filepath), f'bending_analysis_{os.path.basename(data_filepath).replace(".npy","")}.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_filename)
    print(f"\nHistogram plot saved to {plot_filename}")
    plt.close(fig)

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # The new data file from the pretension continuation run
    data_file = os.path.join(project_root, 'plots', 'workspace_points_pretension_continuation_500.npy')
    
    analyze_q_distribution(data_file)
