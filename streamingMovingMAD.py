# MAD (Mean Absolute Deviation)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StreamingMovingMAD:
    '''Moving Mean Absolute Deviation (M.A.D) - using M.A.D instead of Arithmetic Mean (or Average)'''

    def __init__(self, threshold=1.5) -> None:
        # Initialize data stream
        self.data_streaming = []
        # Parameters
        self.max_deviation_from_expected = threshold

    def add_data_point(self, value):
        # Add new data point to the streaming data
        self.data_streaming.append(value)

    def _enough_data(self) -> bool:
        '''Check if there is enough data'''
        return len(self.data_streaming) > 0

    def _median_absolute_deviation(self) -> float:
        '''Return the median absolute deviation (MAD)'''
        data = self.data_streaming
        data = pd.Series(data=data, dtype=float)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return mad

    def _expected_value(self) -> float:
        '''Return the expected value'''
        data = self.data_streaming
        data = pd.Series(data=data, dtype=float)
        return np.median(data)

    def detect_anomaly(self, value):
        if self._enough_data():
            expected_value = self._expected_value()
            deviation = np.abs(value - expected_value)
            if deviation > self.max_deviation_from_expected * self._median_absolute_deviation():
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

def continuous_plot_anomalies(algorithm, parameters, iterations=200, pause_duration=0.1):
    # Initialize the algorithm
    anomaly_detector = algorithm(**parameters)

    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.title('Streaming Moving MAD Anomaly Detection')
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

# Example parameters for the StreamingMovingMAD algorithm
algorithm = StreamingMovingMAD
parameters = {'threshold': 2.0}

# Run the continuous streaming and visualization
continuous_plot_anomalies(algorithm, parameters)