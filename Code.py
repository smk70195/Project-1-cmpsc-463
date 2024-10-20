import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math


# Custom Merge Sort Function
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        # Merge the two halves
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
    return arr


# Preprocess the data (cleaning, sorting by date)
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
    if df['Price'].dtype == 'object':
        df['Price'] = df['Price'].str.replace(',', '').astype(float)
    if df['Open'].dtype == 'object':
        df['Open'] = df['Open'].str.replace(',', '').astype(float)
    if df['High'].dtype == 'object':
        df['High'] = df['High'].str.replace(',', '').astype(float)
    if df['Low'].dtype == 'object':
        df['Low'] = df['Low'].str.replace(',', '').astype(float)

    # Custom merge sort for sorting the Date column
    sorted_dates = merge_sort(df['Date'].tolist())
    df['Date'] = sorted_dates  # Sorted dates in ascending order

    return df


# Load and preprocess data
solana_data_sorted = preprocess_data('/Users/syedkazmi/Desktop/Solana Historical Data.csv')


# Kadane's Algorithm to find maximum subarray (price changes)
def max_subarray(arr):
    max_ending_here = max_so_far = arr[0]
    start = end = s = 0
    for i in range(1, len(arr)):
        if arr[i] > max_ending_here + arr[i]:
            max_ending_here = arr[i]
            s = i
        else:
            max_ending_here += arr[i]
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            start = s
            end = i
    return max_so_far, start, end


# 2D Kadane's Algorithm for finding the maximum sum subarray in a matrix
def kadane_2d(matrix):
    if not matrix:
        return 0, None, None, None, None

    rows, cols = len(matrix), len(matrix[0])
    max_sum = float('-inf')
    final_left = final_right = final_top = final_bottom = 0

    # Iterate over all pairs of columns
    for left in range(cols):
        temp = [0] * rows

        for right in range(left, cols):
            # Summing up all rows between columns left and right
            for i in range(rows):
                temp[i] += matrix[i][right]

            # Apply 1D Kadane's algorithm on this row sum array
            current_sum, top, bottom = kadane_1d(temp)

            if current_sum > max_sum:
                max_sum = current_sum
                final_left = left
                final_right = right
                final_top = top
                final_bottom = bottom

    return max_sum, final_left, final_right, final_top, final_bottom


# Kadane's Algorithm for 1D arrays (used in 2D Kadane's)
def kadane_1d(arr):
    max_sum = float('-inf')
    max_ending_here = 0
    start = end = s = 0

    for i in range(len(arr)):
        max_ending_here += arr[i]
        if max_ending_here > max_sum:
            max_sum = max_ending_here
            start = s
            end = i
        if max_ending_here < 0:
            max_ending_here = 0
            s = i + 1

    return max_sum, start, end


# Calculate daily price changes
solana_data_sorted['Price_Change'] = solana_data_sorted['Price'].diff().fillna(0)

# Find the period of max gain
max_gain, start_idx, end_idx = max_subarray(solana_data_sorted['Price_Change'].tolist())
max_gain_period = solana_data_sorted.iloc[start_idx:end_idx + 1]

# Extract detailed information about the maximum gain period
start_price = max_gain_period['Price'].iloc[0]
end_price = max_gain_period['Price'].iloc[-1]
percentage_gain = ((end_price - start_price) / start_price) * 100

print(
    f"Maximum gain period: {max_gain_period['Date'].iloc[0]} to {max_gain_period['Date'].iloc[-1]} with a gain of {max_gain}")
print(f"Starting price: ${start_price:.2f}, Ending price: ${end_price:.2f}, Percentage gain: {percentage_gain:.2f}%")


# Function to calculate Euclidean distance between two points
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Closest pair algorithm (1D price anomaly detection)
def closest_pair(points):
    if len(points) < 2:
        return float('inf'), None, None
    points = sorted(points)
    return closest_pair_recursive(points)


def closest_pair_recursive(points):
    n = len(points)
    if n <= 3:
        return brute_force_closest_pair(points)
    mid = n // 2
    left_closest = closest_pair_recursive(points[:mid])
    right_closest = closest_pair_recursive(points[mid:])
    min_distance = min(left_closest[0], right_closest[0])
    split_closest = closest_split_pair(points, min_distance)
    return min(left_closest, right_closest, split_closest, key=lambda x: x[0])


def brute_force_closest_pair(points):
    min_dist = float('inf')
    pair = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = distance(points[i], points[j])
            if dist < min_dist:
                min_dist = dist
                pair = (points[i], points[j])
    return min_dist, pair[0], pair[1]


def closest_split_pair(points, delta):
    mid_x = points[len(points) // 2][0]
    in_strip = [p for p in points if abs(p[0] - mid_x) < delta]
    in_strip.sort(key=lambda p: p[1])
    min_dist = delta
    pair = None
    for i in range(len(in_strip)):
        for j in range(i + 1, min(i + 7, len(in_strip))):
            dist = distance(in_strip[i], in_strip[j])
            if dist < min_dist:
                min_dist = dist
                pair = (in_strip[i], in_strip[j])
    if pair is None:
        return delta, None, None
    else:
        return min_dist, pair[0], pair[1]


# Prepare the points (index, price)
points = list(zip(solana_data_sorted.index, solana_data_sorted['Price']))

# Find anomalies
closest_pair_dist, p1, p2 = closest_pair(points)

if p1 is not None and p2 is not None:
    print(
        f"Closest anomaly points: {solana_data_sorted['Date'].iloc[p1[0]]} and {solana_data_sorted['Date'].iloc[p2[0]]} with a distance of {closest_pair_dist}")
    print(f"Anomaly prices: ${solana_data_sorted['Price'].iloc[p1[0]]} and ${solana_data_sorted['Price'].iloc[p2[0]]}")
else:
    print("No anomalies detected")


# Enhanced Report: Price trend over time, moving averages, and anomaly detection
def generate_enhanced_report(solana_data_sorted, max_gain_period, p1, p2):
    # Plotting Price with Maximum Gain Period and Anomalies
    plt.figure(figsize=(10, 6))
    plt.plot(solana_data_sorted['Date'], solana_data_sorted['Price'], label='Price')
    plt.axvspan(max_gain_period['Date'].iloc[0], max_gain_period['Date'].iloc[-1], color='yellow', alpha=0.3,
                label='Max Gain Period')
    if p1 is not None and p2 is not None:
        plt.scatter([solana_data_sorted['Date'].iloc[p1[0]], solana_data_sorted['Date'].iloc[p2[0]]],
                    [p1[1], p2[1]], color='red', label='Anomalies')

    # Adding Moving Averages
    solana_data_sorted['7-day Moving Avg'] = solana_data_sorted['Price'].rolling(window=7).mean()
    solana_data_sorted['30-day Moving Avg'] = solana_data_sorted['Price'].rolling(window=30).mean()
    plt.plot(solana_data_sorted['Date'], solana_data_sorted['7-day Moving Avg'], label='7-day Moving Avg',
             linestyle='--', color='orange')
    plt.plot(solana_data_sorted['Date'], solana_data_sorted['30-day Moving Avg'], label='30-day Moving Avg',
             linestyle='--', color='green')

    # Add Annotations for key events or spikes
    max_price_date = solana_data_sorted.loc[solana_data_sorted['Price'].idxmax()]['Date']
    max_price_value = solana_data_sorted['Price'].max()
    plt.annotate(f'Highest Price: ${max_price_value:.2f}', xy=(max_price_date, max_price_value),
                 xytext=(max_price_date, max_price_value + 50),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Solana Price Trend, Moving Averages, and Anomaly Detection')
    plt.legend()
    plt.show()


# Generate the enhanced report
generate_enhanced_report(solana_data_sorted, max_gain_period, p1, p2)

# Bar Plot: Weekly Price Gains/Losses
solana_data_sorted['Weekly Change'] = solana_data_sorted['Price'].diff(periods=7)
plt.figure(figsize=(10, 6))
plt.bar(solana_data_sorted['Date'], solana_data_sorted['Weekly Change'], color='blue', alpha=0.7)
plt.axhline(0, color='red', linewidth=1.5)
plt.title('Solana Weekly Price Gains/Losses')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.show()

# Additional Bar Plot: Monthly Price Gains/Losses
solana_data_sorted['Monthly Change'] = solana_data_sorted['Price'].diff(periods=30)
plt.figure(figsize=(10, 6))
plt.bar(solana_data_sorted['Date'], solana_data_sorted['Monthly Change'], color='purple', alpha=0.7)
plt.axhline(0, color='red', linewidth=1.5)
plt.title('Solana Monthly Price Gains/Losses')
plt.xlabel('Date')
plt.ylabel('Price Change')
plt.show()

# Verification of Code Functionality

# Load and preprocess data
solana_data_sorted = preprocess_data('/Users/syedkazmi/Desktop/Solana Historical Data.csv')
print(solana_data_sorted.head())

#merge sort for time series data example for testing
sample_dates = [datetime(2020, 1, 2), datetime(2019, 12, 31), datetime(2020, 1, 1)]
sorted_dates = merge_sort(sample_dates)
print(sorted_dates)

#Kadane’s Algorithm for Maximum Gain example for testing
price_changes = [1, -3, 4, -1, 2, 1, -5, 4]
max_gain, start_idx, end_idx = max_subarray(price_changes)
print(f"Max gain: {max_gain}, Start index: {start_idx}, End index: {end_idx}")

#Closest Pair of Points for Anomaly Detection
points = [(1, 1), (2, 2), (3, 3), (4, 5)]
dist, p1, p2 = closest_pair(points)
print(f"Closest points: {p1} and {p2}, Distance: {dist}")

# Weekly change plot
plt.bar(solana_data_sorted['Date'], solana_data_sorted['Weekly Change'], color='blue')
plt.show()

# Monthly change plot
plt.bar(solana_data_sorted['Date'], solana_data_sorted['Monthly Change'], color='purple')
plt.show()


