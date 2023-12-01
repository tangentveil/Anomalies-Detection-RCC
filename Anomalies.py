# **************************************** Streaming Moving Trimean

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# def generate_data_point(period=50, amplitude=5, noise_factor=1.0):
#     # Regular pattern
#     regular_pattern = amplitude * np.sin(2 * np.pi * len(data_stream) / period)

#     # Seasonal element (e.g., daily fluctuations)
#     seasonal_element = 2 * np.sin(2 * np.pi * len(data_stream) / 24)

#     # Random noise
#     random_noise = noise_factor * np.random.randn()

#     # Combine components to generate data point
#     return regular_pattern + seasonal_element + random_noise

# class StreamingMovingAverage:
#     def __init__(self) -> None:
#         # Initialize data stream
#         self.data_streaming = []

#     def add_data_point(self, value):
#         # Add new data point to the streaming data
#         self.data_streaming.append(value)

#     def detect_anomaly(self, value):
#         # Override this method in the derived class for specific anomaly detection
#         pass

#     def stream_data_and_detect_anomalies(self, data_stream):
#         anomalies = []
#         for value in data_stream:
#             self.add_data_point(value)
#             anomaly = self.detect_anomaly(value)
#             anomalies.append(anomaly)
#         return anomalies

# class StreamingMovingTrimean(StreamingMovingAverage):
#     def __init__(self, threshold=1.5) -> None:
#         super().__init__()
#         # Parameters
#         self.max_deviation_from_expected = threshold

#     def _enough_data(self) -> bool:
#         '''Check if there is enough data'''
#         return len(self.data_streaming) > 0

#     def _standard_deviation(self) -> float:
#         '''Return the standard deviation'''
#         data = self.data_streaming
#         data = pd.Series(data=data, dtype=float)
#         variance = trimean(data) - data
#         return pow(sum(variance ** 2) / len(data), 1/2)

#     def _expected_value(self, timestamp: int) -> float:
#         '''Return the expected value'''
#         data = self.data_streaming
#         data = pd.Series(data=data, dtype=float)
#         return trimean(data)

#     def detect_anomaly(self, value):
#         if self._enough_data():
#             expected_value = self._expected_value(len(self.data_streaming))
#             deviation = abs(value - expected_value)
#             if deviation > self.max_deviation_from_expected * self._standard_deviation() :
#                 return 1  # Anomaly detected
#         return 0  # No anomaly

# def trimean(values):
#     return (np.quantile(values, 0.25) + (2 * np.quantile(values, 0.50)) + np.quantile(values, 0.75)) / 4

# def continuous_plot_anomalies(algorithm, parameters, iterations=200, pause_duration=0.1):
#     # Initialize the algorithm
#     anomaly_detector = algorithm(**parameters)

#     # Set up the plot
#     plt.figure(figsize=(10, 6))
#     plt.title('Streaming Moving Trimean Anomaly Detection')
#     plt.xlabel('Time')
#     plt.ylabel('Value')

#     # Continuously stream data, detect anomalies, and update the plot
#     for _ in range(iterations):
#         data_point = generate_data_point()
#         anomaly_labels = anomaly_detector.stream_data_and_detect_anomalies([data_point])

#         # Update the streaming data
#         anomaly_detector.add_data_point(data_point)

#         # Update the plot with the current data stream
#         plt.plot(anomaly_detector.data_streaming, label='Data Stream')

#         # Check if there are any anomalies
#         if any(anomaly_labels):
#             # Plot the anomalies
#             plt.scatter(len(anomaly_detector.data_streaming) - 1, data_point, color='red', label='Anomalies')

#         plt.legend()
#         plt.draw()
#         plt.pause(pause_duration)

# # Initialize an empty data stream
# data_stream = []

# # Example parameters for the StreamingMovingTrimean algorithm
# algorithm = StreamingMovingTrimean
# parameters = {'threshold': 2.0}

# # Run the continuous streaming and visualization
# continuous_plot_anomalies(algorithm, parameters)


# **************************************** Streaming Moving MAD (Mean Absolute Deviation)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# class StreamingMovingMAD:
#     '''Moving Mean Absolute Deviation (M.A.D) - using M.A.D instead of Arithmetic Mean (or Average)'''

#     def __init__(self, threshold=1.5) -> None:
#         # Initialize data stream
#         self.data_streaming = []
#         # Parameters
#         self.max_deviation_from_expected = threshold

#     def add_data_point(self, value):
#         # Add new data point to the streaming data
#         self.data_streaming.append(value)

#     def _enough_data(self) -> bool:
#         '''Check if there is enough data'''
#         return len(self.data_streaming) > 0

#     def _median_absolute_deviation(self) -> float:
#         '''Return the median absolute deviation (MAD)'''
#         data = self.data_streaming
#         data = pd.Series(data=data, dtype=float)
#         median = np.median(data)
#         mad = np.median(np.abs(data - median))
#         return mad

#     def _expected_value(self) -> float:
#         '''Return the expected value'''
#         data = self.data_streaming
#         data = pd.Series(data=data, dtype=float)
#         return np.median(data)

#     def detect_anomaly(self, value):
#         if self._enough_data():
#             expected_value = self._expected_value()
#             deviation = np.abs(value - expected_value)
#             if deviation > self.max_deviation_from_expected * self._median_absolute_deviation():
#                 return 1  # Anomaly detected
#         return 0  # No anomaly

# def generate_data_point(period=50, amplitude=5, noise_factor=1.0):
#     # Regular pattern
#     regular_pattern = amplitude * np.sin(2 * np.pi * len(data_stream) / period)

#     # Seasonal element (e.g., daily fluctuations)
#     seasonal_element = 2 * np.sin(2 * np.pi * len(data_stream) / 24)

#     # Random noise
#     random_noise = noise_factor * np.random.randn()

#     # Combine components to generate data point
#     return regular_pattern + seasonal_element + random_noise

# def continuous_plot_anomalies(algorithm, parameters, iterations=200, pause_duration=0.1):
#     # Initialize the algorithm
#     anomaly_detector = algorithm(**parameters)

#     # Set up the plot
#     plt.figure(figsize=(10, 6))
#     plt.title('Streaming Moving MAD Anomaly Detection')
#     plt.xlabel('Time')
#     plt.ylabel('Value')

#     # Continuously stream data, detect anomalies, and update the plot
#     for _ in range(iterations):
#         data_point = generate_data_point()
#         anomaly_labels = anomaly_detector.detect_anomaly(data_point)

#         # Update the streaming data
#         anomaly_detector.add_data_point(data_point)

#         # Update the plot with the current data stream
#         plt.plot(anomaly_detector.data_streaming, label='Data Stream')

#         # Check if there are any anomalies
#         if anomaly_labels:
#             # Plot the anomalies
#             plt.scatter(len(anomaly_detector.data_streaming) - 1, data_point, color='red', label='Anomalies')

#         plt.legend()
#         plt.draw()
#         plt.pause(pause_duration)

# # Initialize an empty data stream
# data_stream = []

# # Example parameters for the StreamingMovingMAD algorithm
# algorithm = StreamingMovingMAD
# parameters = {'threshold': 2.0}

# # Run the continuous streaming and visualization
# continuous_plot_anomalies(algorithm, parameters)

# **************************************** Streaming Moving MAD and Streaming Moving Trimean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StreamingMovingAnomalyDetector:
    def __init__(self, method='MAD', threshold=1.5) -> None:
        # Initialize data stream
        self.data_streaming = []
        # Parameters
        self.max_deviation_from_expected = threshold
        self.method = method.lower()  # Convert method to lowercase for case-insensitivity

    def add_data_point(self, value):
        # Add new data point to the streaming data
        self.data_streaming.append(value)

    def _enough_data(self) -> bool:
        '''Check if there is enough data'''
        return len(self.data_streaming) > 0

    def _calculate_deviation(self) -> float:
        '''Calculate the deviation based on the chosen method'''
        data = pd.Series(data=self.data_streaming, dtype=float)

        if self.method == 'mad':
            # Median Absolute Deviation (MAD)
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            return mad
        elif self.method == 'trimean':
            # Trimean
            return (np.quantile(data, 0.25) + (2 * np.quantile(data, 0.50)) + np.quantile(data, 0.75)) / 4
        else:
            raise ValueError("Invalid method. Choose either 'MAD' or 'Trimean'.")

    def _expected_value(self) -> float:
        '''Return the expected value based on the chosen method'''
        data = pd.Series(data=self.data_streaming, dtype=float)

        if self.method == 'mad':
            # Median
            return np.median(data)
        elif self.method == 'trimean':
            # Trimean
            return (np.quantile(data, 0.25) + np.quantile(data, 0.50) + np.quantile(data, 0.75)) / 3

    def detect_anomaly(self, value):
        if self._enough_data():
            expected_value = self._expected_value()
            deviation = np.abs(value - expected_value)
            if deviation > self.max_deviation_from_expected * self._calculate_deviation():
                return 1  # Anomaly detected
        return 0  # No anomaly

def generate_data_point(period=50, amplitude=5, noise_factor=1.0):
    # Regular pattern
    regular_pattern = amplitude * np.sin(2 * np.pi * len(data_stream) / period)

    # Seasonal element (e.g., daily fluctuations)
    seasonal_element = 2 * np.sin(2 * np.pi * len(data_stream) / 24)

    # Random noise
    random_noise = noise_factor * np.random.randn()

    # Combine components to generate data point
    return regular_pattern + seasonal_element + random_noise

def continuous_plot_anomalies(algorithm, parameters, iterations=100, pause_duration=0.1):
    # Initialize the algorithm
    anomaly_detector = algorithm(**parameters)

    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.title(f'Streaming Moving {anomaly_detector.method.capitalize()} Anomaly Detection')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # Continuously stream data, detect anomalies, and update the plot
    for _ in range(iterations):
        data_point = generate_data_point()
        anomaly_labels = anomaly_detector.detect_anomaly(data_point)

        # Update the streaming data
        anomaly_detector.add_data_point(data_point)

        # Update the plot with the current data stream
        plt.plot(anomaly_detector.data_streaming, label='Data Stream')

        # Check if there are any anomalies
        if anomaly_labels:
            # Plot the anomalies
            plt.scatter(len(anomaly_detector.data_streaming) - 1, data_point, color='red', label='Anomalies')

        plt.legend()
        plt.draw()
        plt.pause(pause_duration)

# Initialize an empty data stream
data_stream = []

# Example parameters for the StreamingMovingAnomalyDetector
algorithm = StreamingMovingAnomalyDetector
parameters = {'method': 'MAD', 'threshold': 2.0}

# Run the continuous streaming and visualization
continuous_plot_anomalies(algorithm, parameters)

# **************************************** Streaming Exponential Average

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# class StreamingExponentialMovingAverage:
#     '''Exponential Weighted Moving Average (EWMA) algorithm'''
#     # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ewm.html

#     def __init__(self, threshold=1.5, alpha=0.3) -> None:
#         # Initialize data stream
#         self.data_streaming = []
#         # Parameters
#         self.max_deviation_from_expected = threshold
#         self.alpha = alpha

#     def add_data_point(self, value):
#         # Add new data point to the streaming data
#         self.data_streaming.append(value)

#     def _enough_data(self) -> bool:
#         '''Check if there is enough data'''
#         return len(self.data_streaming) > 0

#     def _expected_value(self, timestamp: int) -> float:
#         '''Return the expected value'''
#         data = pd.Series(data=self.data_streaming, dtype=float)
#         return data.ewm(alpha=self.alpha, adjust=True).mean().iloc[-1]

#     def detect_anomaly(self, value):
#         if self._enough_data():
#             expected_value = self._expected_value(len(self.data_streaming))
#             deviation = abs(value - expected_value)
#             if deviation > self.max_deviation_from_expected * np.std(self.data_streaming):
#                 return 1  # Anomaly detected
#         return 0  # No anomaly

#     def stream_data_and_detect_anomalies(self, data_stream):
#         anomalies = []
#         for value in data_stream:
#             self.add_data_point(value)
#             anomaly = self.detect_anomaly(value)
#             anomalies.append(anomaly)
#         return anomalies

# def generate_data_point(period=50, amplitude=5, noise_factor=1.0):
#     # Regular pattern
#     regular_pattern = amplitude * np.sin(2 * np.pi * len(data_stream) / period)

#     # Seasonal element (e.g., daily fluctuations)
#     seasonal_element = 2 * np.sin(2 * np.pi * len(data_stream) / 24)

#     # Random noise
#     random_noise = noise_factor * np.random.randn()

#     # Combine components to generate data point
#     return regular_pattern + seasonal_element + random_noise

# class StreamingMovingAverage:
#     def __init__(self) -> None:
#         # Initialize data stream
#         self.data_streaming = []

#     def add_data_point(self, value):
#         # Add new data point to the streaming data
#         self.data_streaming.append(value)

#     def detect_anomaly(self, value):
#         # Override this method in the derived class for specific anomaly detection
#         pass

# def continuous_plot_anomalies(algorithm, parameters, iterations=200, pause_duration=0.1):
#     # Initialize the algorithm
#     anomaly_detector = algorithm(**parameters)

#     # Set up the plot
#     plt.figure(figsize=(10, 6))
#     plt.title('Streaming Moving EWMA Anomaly Detection')
#     plt.xlabel('Time')
#     plt.ylabel('Value')

#     # Continuously stream data, detect anomalies, and update the plot
#     for _ in range(iterations):
#         data_point = generate_data_point()
#         anomaly_labels = anomaly_detector.stream_data_and_detect_anomalies([data_point])

#         # Update the streaming data
#         anomaly_detector.add_data_point(data_point)

#         # Update the plot with the current data stream
#         plt.plot(anomaly_detector.data_streaming, label='Data Stream')

#         # Check if there are any anomalies
#         if any(anomaly_labels):
#             # Plot the anomalies
#             plt.scatter(len(anomaly_detector.data_streaming) - 1, data_point, color='red', label='Anomalies')

#         plt.legend()
#         plt.draw()
#         plt.pause(pause_duration)

# # Initialize an empty data stream
# data_stream = []

# # Example parameters for the StreamingExponentialMovingAverage algorithm
# algorithm = StreamingExponentialMovingAverage
# parameters = {'threshold': 2.0, 'alpha': 0.3}

# # Run the continuous streaming and visualization
# continuous_plot_anomalies(algorithm, parameters)

# **************************************** Robust Random Cut Forest (RRCF)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RCTree:
    def __init__(self, tree_size=256):
        self.size = tree_size
        self.leaves = []
        self.height = 0

    def update(self, point, index):
        leaf = RCTreeNode(point, index)
        self.leaves.append(leaf)
        if len(self.leaves) > 2 * self.size:
            self.leaves = self.leaves[::2]
        self.height = int(np.ceil(np.log2(len(self.leaves))))

    def codisp(self, point):
        if not self.leaves:
            return 0.0
        else:
            return np.mean([leaf.codisp(point) for leaf in self.leaves])

class RCTreeNode:
    def __init__(self, point, index):
        self.left = None
        self.right = None
        self.point = point
        self.index = index
        self.is_leaf = True

    def codisp(self, point):
        if self.is_leaf:
            return np.linalg.norm(self.point - point)
        else:
            left_codisp = self.left.codisp(point)
            right_codisp = self.right.codisp(point)
            return max(left_codisp, right_codisp)

class RandomForest:
    def __init__(self, num_trees=50, tree_size=256):
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.trees = [RCTree(tree_size) for _ in range(num_trees)]

    def update(self, point):
        for i, tree in enumerate(self.trees):
            tree.update(point, i)

    def codisp(self, point):
        return np.mean([tree.codisp(point) for tree in self.trees])

class StreamingAnomalyDetector:
    def __init__(self, trees=50, tree_size=256):
        self.forest = RandomForest(num_trees=trees, tree_size=tree_size)

    def update(self, data_point):
        # Update the forest with the new data point
        self.forest.update(data_point)

    def detect_anomaly(self, data_point):
        # Compute anomaly score for the data point
        score = self.forest.codisp(data_point)
        return score

def generate_data_point(period=50, amplitude=5, noise_factor=1.0):
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
