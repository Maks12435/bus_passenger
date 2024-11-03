import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation_matrix(data):
    corr = data.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

def plot_dwell_and_travel_times(sample):
    hours = []
    average_dwell_times = []
    average_travel_times = []

    for i in range(6, 22):
        filtered_data = sample[sample['hour'] == i]
        avg_dwell_time = filtered_data['dwell_time_in_seconds'].mean() or 0
        avg_travel_time = filtered_data['travel_time'].mean() or 0

        hours.append(i)
        average_dwell_times.append(avg_dwell_time)
        average_travel_times.append(avg_travel_time)

    plt.figure(figsize=(12, 6))
    plt.bar(hours, average_dwell_times, width=0.4, label='Average Dwell Time (s)', color='b', align='center')
    plt.bar(np.array(hours) + 0.4, average_travel_times, width=0.4, label='Average Travel Time (s)', color='r', align='center')
    plt.title('Average Dwell and Travel Times by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Time in Seconds')
    plt.xticks(hours)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
