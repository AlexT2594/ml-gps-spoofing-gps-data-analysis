import random


def get_data_from_file(file):
    sample_file = open(file, 'r')
    dataset = []
    dataset_labels = []
    for line in sample_file:
        fields = line.split(',')
        message_ID = fields[0]
        if message_ID == "$GPGGA":

            entry = get_GGA_entry_as_array(line)

            dataset.append(entry)
            dataset_labels.append(random.randint(0, 1))

    return dataset, dataset_labels


def get_GGA_entry_as_array(entry):
    fields = entry.split(',')

    utc = check_if_null(fields[1])
    lat = check_if_null(fields[2])
    lat_dir = check_if_null(fields[3])
    long = check_if_null(fields[4])
    long_dir = check_if_null(fields[5])
    gps_qi = check_if_null(fields[6])
    sat_num = check_if_null(fields[7])
    hor_dilution = check_if_null(fields[8])
    antenna_alt = check_if_null(fields[9])
    units_antenna_alt = check_if_null(fields[10])
    geoidal_sep = check_if_null(fields[11])
    units_geoidal_sep = check_if_null(fields[12])
    age_of_diff_gps_data = check_if_null(fields[13])
    diff_ref_station_id = check_if_null(fields[14].split('*')[0])

    entry_array = [gps_qi, sat_num,
             hor_dilution, antenna_alt, units_antenna_alt,
             geoidal_sep, units_geoidal_sep, age_of_diff_gps_data,
             diff_ref_station_id]

    '''
    entry = [lat, lat_dir, long, long_dir, gps_qi, sat_num,
             hor_dilution, antenna_alt, units_antenna_alt,
             geoidal_sep, units_geoidal_sep, age_of_diff_gps_data,
             diff_ref_station_id]   
    '''

    return entry_array

def get_RMC_entry_as_array(entry):
    fields = entry.split(",")

    utc = check_if_null(fields[1])
    status = check_if_null(fields[2])
    lat = check_if_null(fields[3])
    lat_dir = check_if_null(fields[4])
    long = check_if_null(fields[5])
    long_dir = check_if_null(fields[6])
    speed_over_ground = check_if_null(fields[7])
    course_over_ground = check_if_null(fields[8])
    date = check_if_null(fields[9])
    magnetic_variation = check_if_null(fields[10])
    mode = check_if_null(fields[11].split('*')[0])

    entry_array = [status, speed_over_ground, course_over_ground, magnetic_variation, mode]

    return entry_array


def get_GSA_entry_as_array(entry):
    fields = entry.split(",")

    mode_1 = check_if_null(fields[1])
    mode_2 = check_if_null(fields[2])

    satellite_used_1 = check_if_null(fields[3])
    satellite_used_2 = check_if_null(fields[4])
    satellite_used_3 = check_if_null(fields[5])
    satellite_used_4 = check_if_null(fields[6])
    satellite_used_5 = check_if_null(fields[7])
    satellite_used_6 = check_if_null(fields[8])
    satellite_used_7 = check_if_null(fields[9])
    satellite_used_8 = check_if_null(fields[10])
    satellite_used_9 = check_if_null(fields[11])
    satellite_used_10 = check_if_null(fields[12])
    satellite_used_11 = check_if_null(fields[13])
    satellite_used_12 = check_if_null(fields[14])

    pdop = check_if_null(fields[15])
    hdop = check_if_null(fields[16])
    vdop = check_if_null(fields[17].split('*')[0])

    entry_array = [mode_1, mode_2, satellite_used_1, satellite_used_2, satellite_used_3, satellite_used_4, satellite_used_5,
             satellite_used_6, satellite_used_7, satellite_used_8, satellite_used_9, satellite_used_10, satellite_used_11,
             satellite_used_12, pdop, hdop, vdop]

    return entry_array

def get_VTG_entry_as_array(entry):
    fields = entry.split(",")

    course_1 = check_if_null(fields[1])
    reference1 = check_if_null(fields[2])
    course_2 = check_if_null(fields[3])
    reference2 = check_if_null(fields[4])
    speed1 = check_if_null(fields[5])
    units1 = check_if_null(fields[6])
    speed2 = check_if_null(fields[7])
    units2 = check_if_null(fields[8])
    mode = check_if_null(fields[9].split('*')[0])

    entry_array = [course_1, course_2, speed1, speed2, mode]

    return entry_array


def get_single_GSV_entry_as_array(entry):

    MAX_SATELLITES_FOR_ENTRY = 4

    fields = entry.split(",")

    n_of_messages = fields[1]
    message_number = fields[2]
    total_satellites = fields[3]

    if message_number < n_of_messages:
        satellites_for_entry = MAX_SATELLITES_FOR_ENTRY
    else:
        satellites_for_entry = total_satellites % n_of_messages

    entry_array = []

    for i in range(satellites_for_entry):
        entry_array.append(fields[4 + 4 * i])

    return entry_array


def get_GGA_entry_as_string(entry):
    entry_array = get_GGA_entry_as_array(entry)
    return " ".join(entry_array)


def check_if_null(elem):
    if elem == '':
        return '-1'
    else:
        return elem
