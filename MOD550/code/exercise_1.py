import numpy as np
import matplotlib.pyplot as plt

class DatasetGenerator:
    def __init__(self, num_points, x_range=(-10, 10), y_range=(-10, 10)):
        """
        Initialize the DatasetGenerator with parameters for dataset generation.
        :param num_points: Number of data points to generate
        :param x_range: Tuple (min_x, max_x) specifying the range for x-coordinates
        :param y_range: Tuple (min_y, max_y) specifying the range for y-coordinates
        """
        self.num_points = num_points
        self.x_range = x_range
        self.y_range = y_range

    def generate_random_2d(self):
        """
        Generate a dataset of random 2D points.
        :return: A NumPy array of shape (num_points, 2)
        """
        x_values = np.random.uniform(self.x_range[0], self.x_range[1], self.num_points)
        y_values = np.random.uniform(self.y_range[0], self.y_range[1], self.num_points)
        return np.column_stack((x_values, y_values))

    def generate_function_with_noise(self, function, noise_level=1.0):
        """
        Generate a dataset around a given function with added noise.
        :param function: A callable function f(x) that defines the relationship between x and y
        :param noise_level: The standard deviation of the noise to add to y-values
        :return: A NumPy array of shape (num_points, 2)
        """
        x_values = np.linspace(self.x_range[0], self.x_range[1], self.num_points)
        y_values = function(x_values) + np.random.normal(0, noise_level, self.num_points)
        return np.column_stack((x_values, y_values))

    def generate_your_truth(self, clusters=3, spread=1.0):
        """
        Generate a non-linear dataset with clusters or patterns.
        :param clusters: Number of clusters in the dataset
        :param spread: The spread (standard deviation) of each cluster
        :return: A NumPy array of shape (num_points, 2)
        """
        points_per_cluster = self.num_points // clusters
        dataset = []
        for _ in range(clusters):
            cluster_center = (
                np.random.uniform(self.x_range[0], self.x_range[1]),
                np.random.uniform(self.y_range[0], self.y_range[1]),
            )
            cluster_points = np.random.normal(loc=cluster_center, scale=spread, size=(points_per_cluster, 2))
            dataset.append(cluster_points)
        dataset = np.vstack(dataset)
        return dataset[:self.num_points]  # Ensure exact num_points

    def print_dataset(self, dataset):
        """
        Print the dataset in a readable format.
        :param dataset: The dataset to print, as a NumPy array
        """
        print("Generated 2D Dataset:")
        for point in dataset:
            print(f"x: {point[0]:.2f}, y: {point[1]:.2f}")





# Generate the first dataset
generator1 = DatasetGenerator(num_points=100, x_range=(-10, 10), y_range=(-10, 10))
dataset1 = generator1.generate_random_2d()

# Generate the second dataset
generator2 = DatasetGenerator(num_points=50, x_range=(-5, 5), y_range=(-5, 5))
dataset2 = generator2.generate_function_with_noise(lambda x: 2 * x + 1, noise_level=1.5)

# Combine the datasets
combined_dataset = np.vstack((dataset1, dataset2))

# Print the combined dataset
#generator1.print_dataset(combined_dataset)






# Save the dataset to a CSV file
output_path = "combined_dataset.csv"
np.savetxt(output_path, combined_dataset, delimiter=",", header="x,y", comments="")

# Save the plot
plt.figure(figsize=(10, 6))
plt.scatter(dataset1[:, 0], dataset1[:, 1], label="Dataset 1 (Random)", alpha=0.7, c="blue")
plt.scatter(dataset2[:, 0], dataset2[:, 1], label="Dataset 2 (Function + Noise)", alpha=0.7, c="orange")
plt.title("Combined Dataset")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plot_path = "combined_dataset_plot.png"
plt.savefig(plot_path)
plt.close()

# Save metadata to a text file
metadata = """
Metadata for Combined Dataset:

1. Dataset 1:
   - Type: 2D Random Points
   - Number of Points: 100
   - X Range: (-10, 10)
   - Y Range: (-10, 10)

2. Dataset 2:
   - Type: Function with Noise (y = 2x + 1 + noise)
   - Number of Points: 50
   - X Range: (-5, 5)
   - Y Range: Derived from the function
   - Noise Level: 1.5

3. Combined Dataset:
   - Total Points: 150
   - Description: Appended Dataset 1 and Dataset 2
"""
metadata_path = "combined_dataset_metadata.txt"
with open(metadata_path, "w") as file:
    file.write(metadata)

print("Files saved successfully:")
print(f"Dataset: {output_path}")
print(f"Plot: {plot_path}")
print(f"Metadata: {metadata_path}")


