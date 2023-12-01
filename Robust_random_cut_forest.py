import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RCTree:
    def __init__(self, tree_size=256):
        """
        Initialize a Random Cut Tree.

        Parameters:
        - tree_size (int): Size of the tree.
        """
        self.size = tree_size
        self.leaves = []
        self.height = 0

    def update(self, point, index):
        """
        Update the tree with a new point.

        Parameters:
        - point (numpy.ndarray): Data point to be added.
        - index (int): Index of the data point.

        Note: The tree is maintained at a fixed size, and older leaves are removed if necessary.
        """
        leaf = RCTreeNode(point, index)
        self.leaves.append(leaf)
        if len(self.leaves) > 2 * self.size:
            self.leaves = self.leaves[::2]  # Remove older leaves
        self.height = int(np.ceil(np.log2(len(self.leaves))))

    def codisp(self, point):
        """
        Calculate the co-dispersion of a given point in the tree.

        Parameters:
        - point (numpy.ndarray): Data point for co-dispersion calculation.

        Returns:
        - float: Co-dispersion of the point.
        """
        if not self.leaves:
            return 0.0
        else:
            return np.mean([leaf.codisp(point) for leaf in self.leaves])

class RCTreeNode:
    def __init__(self, point, index):
        """
        Initialize a node in the Random Cut Tree.

        Parameters:
        - point (numpy.ndarray): Data point represented by the node.
        - index (int): Index of the data point.
        """
        self.left = None
        self.right = None
        self.point = point
        self.index = index
        self.is_leaf = True

    def codisp(self, point):
        """
        Calculate the co-dispersion of a given point in the node.

        Parameters:
        - point (numpy.ndarray): Data point for co-dispersion calculation.

        Returns:
        - float: Co-dispersion of the point.
        """
        if self.is_leaf:
            return np.linalg.norm(self.point - point)
        else:
            left_codisp = self.left.codisp(point)
            right_codisp = self.right.codisp(point)
            return max(left_codisp, right_codisp)

class RandomForest:
    def __init__(self, num_trees=50, tree_size=256):
        """
        Initialize a Random Forest of Random Cut Trees.

        Parameters:
        - num_trees (int): Number of trees in the forest.
        - tree_size (int): Size of each tree.
        """
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.trees = [RCTree(tree_size) for _ in range(num_trees)]

    def update(self, point):
        """
        Update each tree in the forest with a new point.

        Parameters:
        - point (numpy.ndarray): Data point to be added to each tree.
        """
        for i, tree in enumerate(self.trees):
            tree.update(point, i)

    def codisp(self, point):
        """
        Calculate the average co-dispersion across all trees for a given point.

        Parameters:
        - point (numpy.ndarray): Data point for co-dispersion calculation.

        Returns:
        - float: Average co-dispersion across all trees.
        """
        return np.mean([tree.codisp(point) for tree in self.trees])

class StreamingAnomalyDetector:
    def __init__(self, trees=50, tree_size=256):
        """
        Initialize a streaming anomaly detector using a Random Forest.

        Parameters:
        - trees (int): Number of trees in the forest.
        - tree_size (int): Size of each tree.
        """
        self.forest = RandomForest(num_trees=trees, tree_size=tree_size)

    def update(self, data_point):
        """
        Update the streaming detector with a new data point.

        Parameters:
        - data_point (numpy.ndarray): New data point to update the detector.
        """
        self.forest.update(data_point)

    def detect_anomaly(self, data_point):
        """
        Detect anomaly for a given data point.

        Parameters:
        - data_point (numpy.ndarray): Data point for anomaly detection.

        Returns:
        - float: Anomaly score for the data point.
        """
        score = self.forest.codisp(data_point)
        return score

def generate_data_point(period=50, amplitude=5, noise_factor=1.0):
    """
    Generate a synthetic data point with a regular pattern, seasonal element, and random noise.

    Parameters:
    - period (int): Period of the regular pattern.
    - amplitude (float): Amplitude of the regular pattern.
    - noise_factor (float): Factor for random noise.

    Returns:
    - float: Generated data point.
    """
    # Initialize an empty data stream
    data_stream = []

    # Regular pattern
    regular_pattern = amplitude * np.sin(2 * np.pi * len(data_stream) / period)

    # Seasonal element (e.g., daily fluctuations)
    seasonal_element = 2 * np.sin(2 * np.pi * len(data_stream) / 24)

    # Random noise
    random_noise = noise_factor * np.random.randn()

    # Combine components to generate data point
    return regular_pattern + seasonal_element + random_noise

# Example usage:
streaming_detector = StreamingAnomalyDetector()

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Streaming Moving Anomaly Detection with Random Forest')
ax.set_xlabel('Time')
ax.set_ylabel('Value')

# Lists to store data points and anomaly scores
data_stream_points = []
anomaly_scores = []

# Function to update the plot in each animation frame
def update(frame):
    data_point = generate_data_point()

    # Update the detector with the new data point
    streaming_detector.update(data_point)

    # Detect anomaly
    anomaly_score = streaming_detector.detect_anomaly(data_point)

    # Append data points and anomaly scores to lists
    data_stream_points.append(data_point)
    anomaly_scores.append(anomaly_score)

    # Plot the entire data stream as a line without markers
    ax.clear()
    ax.plot(data_stream_points, label='Data Stream', color='blue')

    # Highlight anomalies
    anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score > 1.5]
    anomaly_points = [data_stream_points[i] for i in anomaly_indices]
    ax.scatter(anomaly_indices, anomaly_points, color='red', label='Anomalies')

    ax.legend()

# Create the animation
ani = FuncAnimation(fig, update, frames=200, interval=100, repeat=False)
plt.show()
