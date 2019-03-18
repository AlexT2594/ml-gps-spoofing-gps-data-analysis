from utils.data import get_lat_long_entries_from_file
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from utils.data import nmea_log_to_entry

import multiprocessing as mp
from threading import Thread


from kafka import KafkaConsumer

topic_name = 'raw_nmea'

fig, axes = plt.subplots(nrows=2, ncols=1)
plt.subplots_adjust(left=0.15, hspace=0.5)

def main():
    print("==> Lat/Long Analysis")
    print("\tDifference between adjacent entries analysis")

    lat_long_entries = get_lat_long_entries_from_file("../data/true_data.txt")

    entries_length = len(lat_long_entries)
    print("==> Number of entries: " + str(entries_length))

    entry_index = 0

    # we get the module of the difference between each adjacent entry
    lat_long_entries_difference = []
    lat_entries_difference = []
    long_entries_difference = []

    while entry_index < entries_length - 1:
        lat_value_1 = float(lat_long_entries[entry_index][0])
        long_value_1 = float(lat_long_entries[entry_index][1])

        lat_value_2 = float(lat_long_entries[entry_index + 1][0])
        long_value_2 = float(lat_long_entries[entry_index + 1][1])

        single_entry = [abs(lat_value_1 - lat_value_2), abs(long_value_1 - long_value_2)]

        lat_long_entries_difference.append(single_entry)
        lat_entries_difference.append(single_entry[0])
        long_entries_difference.append(single_entry[1])

        entry_index += 1

    lat_and_long_entries_difference = list()
    lat_and_long_entries_difference.append(lat_entries_difference)
    lat_and_long_entries_difference.append(long_entries_difference)

    # here we'll have the lat entries of both spoofed and not spoofed values

    axes[0].set_title("Lat differences")
    axes[0].boxplot(lat_entries_difference, vert=False, showfliers=False)
    axes[0].set_yticklabels([''])

    print("==> Lat values:")
    print("==> \tMedian: " + str(np.median(lat_entries_difference)))
    print("==> \tUpper percentile: " + str(np.percentile(lat_entries_difference, 75)))

    axes[1].set_title("Long differences")
    axes[1].boxplot(long_entries_difference, vert=False, showfliers=False)
    axes[1].set_yticklabels([''])

    print("==> Long values:")
    print("==> \tMedian: " + str(np.median(long_entries_difference)))
    print("==> \tUpper percentile: " + str(np.percentile(long_entries_difference, 75)))

    queue = mp.Queue()

    lat_long_vals = [[], []]
    lat_long_diff_vals = [[], []]

    thread = Thread(target=consumeData, args=(queue,))
    thread.start()

    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate, fargs=(queue, [lat_entries_difference, long_entries_difference],
                                                       lat_long_vals, lat_long_diff_vals), interval=1000)
    plt.show()


def animate(i, q, ground_truth, observation_vals, observation_diff_vals):

    lat_long = q.get()

    observation_vals[0].append(lat_long[0])
    observation_vals[1].append(lat_long[1])

    if len(observation_vals[0]) < 2:
        return

    lat_diff = abs(lat_long[0] - observation_vals[0][-2])
    long_diff = abs(lat_long[1] - observation_vals[1][-2])

    observation_diff_vals[0].append(lat_diff)
    observation_diff_vals[1].append(long_diff)

    axes[0].clear()
    axes[0].boxplot([observation_diff_vals[0], ground_truth[0]], vert=False, showfliers=False)
    axes[0].set_yticklabels(['Monitoring', 'Ground\nTruth'])

    axes[1].clear()
    axes[1].boxplot([observation_diff_vals[1], ground_truth[1]], vert=False, showfliers=False)
    axes[1].set_yticklabels(['Monitoring', 'Ground\nTruth'])


def consumeData(queue):

    consumer = KafkaConsumer(topic_name, auto_offset_reset='earliest', bootstrap_servers=['localhost:9092'])
                             #,consumer_timeout_ms=1000)

    print("Started the consumer")

    for msg in consumer:
        entry = msg.value.decode('utf-8')

        print("==> Entry")
        print(entry)

        entry = nmea_log_to_entry(entry)

        if len(entry) == 0:
            continue

        GGA_entry = entry['$GPGGA']
        lat = GGA_entry[1]
        long = GGA_entry[2]

        print("Latitude " + lat)
        print("Longitude " + long)

        queue.put([float(lat), float(long)])

    consumer.close()


if __name__ == '__main__':
    main()
