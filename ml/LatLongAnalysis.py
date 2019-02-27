from utils.data import get_lat_long_entries_from_file
import matplotlib.pyplot as plt
import numpy as np


def main():
    print("==> Lat/Long Analysis")
    print("\tDifference between adjacent entries analysis")

    lat_long_entries = get_lat_long_entries_from_file("../data/work_route_sample.txt")

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
    lat_entries = list()
    lat_entries.append(lat_entries_difference)
    lat_entries.append(lat_entries_difference)

    long_entries = list()
    long_entries.append(long_entries_difference)
    long_entries.append(long_entries_difference)

    fig_lat, ax_lat = plt.subplots()
    ax_lat.set_title("Lat differences")
    lat_boxplot = ax_lat.boxplot(lat_entries)
    ax_lat.set_xticklabels(['Spoofed', 'Not Spoofed'])

    print("==> Lat values:")
    print("==> \tMedian: " + str(np.median(lat_entries[0])))
    print("==> \tUpper percentile: " + str(np.percentile(lat_entries[0], 75)))

    fig_long, ax_long = plt.subplots()
    ax_long.set_title("Long differences")
    ax_long.boxplot(long_entries)
    ax_long.set_xticklabels(['Spoofed', 'Not Spoofed'])

    print("==> Long values:")
    print("==> \tMedian: " + str(np.median(long_entries[0])))
    print("==> \tUpper percentile: " + str(np.percentile(long_entries[0], 75)))

    plt.show()


if __name__ == '__main__':
    main()
