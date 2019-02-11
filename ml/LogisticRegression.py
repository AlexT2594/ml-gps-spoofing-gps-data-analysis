from utils.data import get_lat_long_entries_from_file, lat_long_to_sin_cos

def main():
    lat_long_entries = get_lat_long_entries_from_file("../data/work_route_sample_stable.txt")
    print("==> Entry:")
    print("==>\tLat: " + lat_long_entries[0][0])
    print("==>\tLong: " + lat_long_entries[0][1])

    print("==> Lat in sin/cos: " + str(lat_long_to_sin_cos(lat_long_entries[0][0], "lat")))

if __name__ == '__main__':
    main()
