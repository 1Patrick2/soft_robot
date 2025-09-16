import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def analyze_workspace_data(data_path, plot_path):
    """
    Analyzes the workspace data to quantitatively assess its "fullness".
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    points = np.load(data_path)
    if points.shape[0] < 10:
        print("Error: Not enough data points to analyze.")
        return

    # 1. Calculate the centroid
    centroid = np.mean(points, axis=0)

    # 2. Calculate the radial distance of each point from the centroid
    radial_distances = np.linalg.norm(points - centroid, axis=1)

    # 3. Calculate key metrics
    mean_radius = np.mean(radial_distances)
    std_dev_radius = np.std(radial_distances) # This is our "Fullness Score"
    min_radius = np.min(radial_distances)
    max_radius = np.max(radial_distances)

    # 4. Print the quantitative analysis
    print("\n--- Quantitative Workspace Analysis ---")
    print(f"Number of points: {len(points)}")
    print(f"Centroid: {centroid}")
    print(f"Mean Radial Distance: {mean_radius:.4f} m")
    print(f"Min / Max Radial Distance: {min_radius:.4f} m / {max_radius:.4f} m")
    print(f"Std Dev of Radial Distance (Fullness Score): {std_dev_radius:.4f}")
    print("---------------------------------------")
    print("Interpretation:")
    print("- A LARGER 'Fullness Score' (Std Dev) suggests a more voluminous, less 'shell-like' workspace.")
    print("- A histogram with a wide spread indicates a 'solid' workspace, while a sharp peak indicates a 'hollow' one.")


    # 5. Generate and save a histogram of the radial distances
    plt.figure(figsize=(10, 6))
    plt.hist(radial_distances, bins=50, alpha=0.75, label='Point Distribution')
    plt.title('Distribution of Workspace Points by Radial Distance from Centroid')
    plt.xlabel('Distance from Centroid (m)')
    plt.ylabel('Number of Points')
    plt.grid(True)
    plt.axvline(mean_radius, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_radius:.4f}m')
    plt.legend()
    
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"\nAnalysis histogram saved to {os.path.abspath(plot_path)}")
    plt.close()


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(project_root, 'plots', 'workspace_points.npy')
    plot_file = os.path.join(project_root, 'plots', 'workspace_fullness_histogram.png')
    analyze_workspace_data(data_file, plot_file)
