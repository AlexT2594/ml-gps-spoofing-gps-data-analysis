from utils.data import get_lat_long_entries_from_file
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("==> Lat/Long Analysis")
    print("\tActual position analysis")

    lat_long_entries = get_lat_long_entries_from_file("../data/work_route_sample.txt")

    lat_entries = []
    long_entries = []
    for lat_long_entry in lat_long_entries:
        lat_entries.append(float(lat_long_entry[0]))
        long_entries.append(float(lat_long_entry[1]))

    fig_lat, ax_lat = plt.subplots()
    ax_lat.set_title("Lat entries")
    ax_lat.boxplot(lat_entries)
    ax_lat.set_xticklabels(['Latitude'])

    print("==> Lat values:")
    print("==> \tMedian: " + str(np.median(lat_entries)))
    print("==> \tLower percentile: " + str(np.percentile(lat_entries, 25)))
    print("==> \tUpper percentile: " + str(np.percentile(lat_entries, 75)))

    fig_long, ax_long = plt.subplots()
    ax_long.set_title("Long entries")
    ax_long.boxplot(long_entries)
    ax_long.set_xticklabels(['Longitude'])

    print("==> Long values:")
    print("==> \tMedian: " + str(np.median(long_entries)))
    print("==> \tLower percentile: " + str(np.percentile(long_entries, 25)))
    print("==> \tUpper percentile: " + str(np.percentile(long_entries, 75)))

    plt.show()


if __name__ == '__main__':
    main()
