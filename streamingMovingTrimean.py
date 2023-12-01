import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_data_point(period=50, amplitude=5, noise_factor=1.0):
    # Regular pattern
    regular_pattern = amplitude * np.sin(2 * np.pi * len(data_stream) / period)

    # Seasonal element (e.g., daily fluctuations)
    seasonal_element = 2 * np.sin(2 * np.pi * len(data_stream) / 24)

    # Random noise
    random_noise = noise_factor * np.random.randn()

    # Combine components to generate data point
    return regular_pattern + seasonal_element + random_noise

class StreamingMovingAverage:
    def __init__(self) -> None:
        # Initialize data stream
        self.data_streaming = []

    def add_data_point(self, value):
        # Add new data point to the streaming data
        self.data_streaming.append(value)

    def detect_anomaly(self, value):
        # Override this method in the derived class for specific anomaly detection
        pass

    def stream_data_and_detect_anomalies(self, data_stream):
        anomalies = []
        for value in data_stream:
            self.add_data_point(value)
            anomaly = self.detect_anomaly(value)
            anomalies.append(anomaly)
        return anomalies

class StreamingMovingTrimean(StreamingMovingAverage):
    def __init__(self, threshold=1.5) -> None:
        super().__init__()
        # Parameters
        self.max_deviation_from_expected = threshold

    def _enough_data(self) -> bool:
        '''Check if there is enough data'''
        return len(self.data_streaming) > 0

    def _standard_deviation(self) -> float:
        '''Return the standard deviation'''
        data = self.data_streaming
        data = pd.Series(data=data, dtype=float)
        variance = trimean(data) - data
        return pow(sum(variance ** 2) / len(data), 1/2)

    def _expected_value(self, timestamp: int) -> float:
        '''Return the expected value'''
        data = self.data_streaming
        data = pd.Series(data=data, dtype=float)
        return trimean(data)

    def detect_anomaly(self, value):
        if self._enough_data():
            expected_value = self._expected_value(len(self.data_streaming))
            deviation = abs(value - expected_value)
            if deviation > self.max_deviation_from_expected * self._standard_deviation() :
                return 1  # Anomaly detected
        return 0  # No anomaly

def trimean(values):
    return (np.quantile(values, 0.25) + (2 * np.quantile(values, 0.50)) + np.quantile(values, 0.75)) / 4

def continuous_plot_anomalies(algorithm, parameters, iterations=200, pause_duration=0.1):
    # Initialize the algorithm
    anomaly_detector = algorithm(**parameters)

    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.title('Streaming Moving Trimean Anomaly Detection')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # Continuously stream data, detect anomalies, and update the plot
    for _ in range(iterations):
        data_point = generate_data_point()
        anomaly_labels = anomaly_detector.stream_data_and_detect_anomalies([data_point])

        # Update the streaming data
        anomaly_detector.add_data_point(data_point)

        # Update the plot with the current data stream
        plt.plot(anomaly_detector.data_streaming, label='Data Stream')

        # Check if there are any anomalies
        if any(anomaly_labels):
            # Plot the anomalies
            plt.scatter(len(anomaly_detector.data_streaming) - 1, data_point, color='red', label='Anomalies')

        plt.legend()
        plt.draw()
        plt.pause(pause_duration)

# Initialize an empty data stream
data_stream = []

# Example parameters for the StreamingMovingTrimean algorithm
algorithm = StreamingMovingTrimean
parameters = {'threshold': 2.0}

# Run the continuous streaming and visualization
continuous_plot_anomalies(algorithm, parameters)