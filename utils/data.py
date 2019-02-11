from random import randint
from math import sin, cos, pi

def get_data_from_file(file, spoofed=False):
    sample_file = open(file, 'r')
    dataset = []
    dataset_labels = []
    single_entry = []
    stable = False

    for line in sample_file:
        fields = line.split(',')
        message_ID = fields[0]

        # check if we have become stable
        if not stable:
            # we'll never become stable if not GPGGA
            if message_ID != "$GPGGA":
                continue
            else:
                gps_qi = int(check_if_null(fields[6]))
                if gps_qi != 0:
                    stable = True
                else:
                    continue
        # we don't use an else since it could happen that we became stable and need to start immediately
        # an else would make us loose the first elements
        if stable:
            if message_ID == "$GPGGA":
                single_entry = get_GGA_entry_as_array(line)
            elif message_ID == "$GPGSV" or message_ID == "$GLGSV":
                single_entry = single_entry + get_single_GSV_entry_as_array(line)
            elif message_ID == "$GPGSA" or message_ID == "$GNGSA":
                single_entry = single_entry + get_GSA_entry_as_array(line)
            elif message_ID == "$GPRMC":
                single_entry = single_entry + get_RMC_entry_as_array(line)

                dataset.append(single_entry)
                dataset_labels.append(randint(0, 1))

    return dataset, dataset_labels


def get_data_from_file_2(file, spoofed=False):
    sample_file = open(file, 'r')
    dataset = []
    dataset_labels = []
    single_entry = []
    stable = False

    lines = sample_file.readlines()
    line_index = 0
    lines_count = len(lines)

    while line_index < lines_count:
        line = lines[line_index]
        fields = line.split(',')
        message_ID = fields[0]

        # check if we have become stable
        if not stable:
            # we'll never become stable if not GPGGA
            if message_ID != "$GPGGA":
                line_index += 1
                continue
            else:
                gps_qi = int(check_if_null(fields[6]))
                if gps_qi != 0:
                    stable = True
                else:
                    line_index += 1
                    continue
        # we don't use an else since it could happen that we became stable and need to start immediately
        # an else would make us loose the first elements
        if stable:
            if message_ID == "$GPGGA":
                single_entry = get_GGA_entry_as_array(line)
                line_index += 1
            elif message_ID == "$GPGSV" or message_ID == "$GLGSV":
                fields = line.split(',')
                total_number_of_messages = int(fields[1])
                GSV_messages = []
                for GSV_index in range(line_index, line_index + total_number_of_messages):
                    GSV_messages.append(lines[GSV_index])

                single_entry += get_GSV_entry_as_array(GSV_messages)
                line_index += total_number_of_messages
            elif message_ID == "$GPGSA" or message_ID == "$GNGSA":
                single_entry = single_entry + get_GSA_entry_as_array(line)
                line_index += 1
            elif message_ID == "$GPRMC":
                single_entry = single_entry + get_RMC_entry_as_array(line)
                line_index += 1

                dataset.append(single_entry)
                dataset_labels.append(random.randint(0, 1))
            else:
                line_index += 1

    return dataset, dataset_labels


def get_lat_long_entries_from_file(file):
    sample_file = open(file, 'r')
    lat_long_entries = []
    stable = False

    for line in sample_file:
        fields = line.split(",")
        message_ID = fields[0]

        # check if we have become stable
        if not stable:
            # we'll never become stable if not GPGGA
            if message_ID != "$GPGGA":
                continue
            else:
                gps_qi = int(check_if_null(fields[6]))
                if gps_qi != 0:
                    stable = True
                else:
                    continue
        # we don't use an else since it could happen that we became stable and need to start immediately
        # an else would make us loose the first elements
        if stable:
            if message_ID == "$GPGGA":
                entry = [check_if_null(fields[2]), check_if_null(fields[4])]
                lat_long_entries.append(entry)
            else:
                continue

    return lat_long_entries


def get_time_entries_from_file(file):
    sample_file = open(file, 'r')
    time_entries = []
    stable = False

    for line in sample_file:
        fields = line.split(",")
        message_ID = fields[0]

        # check if we have become stable
        if not stable:
            # we'll never become stable if not GPGGA
            if message_ID != "$GPGGA":
                continue
            else:
                gps_qi = int(check_if_null(fields[6]))
                if gps_qi != 0:
                    stable = True
                else:
                    continue
        # we don't use an else since it could happen that we became stable and need to start immediately
        # an else would make us loose the first elements
        if stable:
            if message_ID == "$GPGGA":
                time_entries.append(check_if_null(fields[1]))

            else:
                continue

    return time_entries


def get_single_elem_from_file(file):
    sample_file = open(file, 'r')
    dataset = []
    dataset_labels = []

    stable = False

    lines = sample_file.readlines()
    line_index = 0
    lines_count = len(lines)

    while line_index < lines_count:
        line = lines[line_index]
        fields = line.split(',')
        message_ID = fields[0]

        if message_ID == "$GPGSV" or message_ID == "$GLGSV":
            fields = line.split(',')
            total_number_of_messages = int(fields[1])
            GSV_messages = []
            for GSV_index in range(line_index, line_index + total_number_of_messages):
                GSV_messages.append(lines[GSV_index])

            single_entry = get_GSV_entry_as_array(GSV_messages)
            line_index += total_number_of_messages

            dataset.append(single_entry)
            dataset_labels.append(random.randint(0, 1))
        else:
            line_index += 1

    return dataset, dataset_labels


def transform_data_into_CSV(file):
    dataset, dataset_labels = get_data_from_file_2(file)
    csv_file_name = ".." + file.split(".")[2] + ".csv"
    csv_file = open(csv_file_name, "w")
    GPGSV_SVS = 16
    GLGSV_SVS = 12
    GSA_SAT_USED = 12

    labels = "spoofed,GGA_gps_qi,GGA_sat_num,GGA_hdop,GGA_antenna_alt,GGA_geoidal_sep,GGA_age_of_diff_gps_data," + \
             "GGA_diff_ref_station_id,"

    labels += "GPGSV_sat_in_view,"

    for i in range(GPGSV_SVS):
        labels += "GPGSV_prn_" + str(i) + ","
        labels += "GPGSV_elevation_" + str(i) + ","
        labels += "GPGSV_azimuth_" + str(i) + ","
        labels += "GPGSV_snr_" + str(i) + ","

    labels += "GLGSV_sat_in_view,"

    for i in range(GLGSV_SVS):
        labels += "GLGSV_prn_" + str(i) + ","
        labels += "GLGSV_elevation_" + str(i) + ","
        labels += "GLGSV_azimuth_" + str(i) + ","
        labels += "GLGSV_snr_" + str(i) + ","

    labels += "GPGSA_mode1,GPGSA_mode2,"

    for i in range(GSA_SAT_USED):
        labels += "GPGSA_sat_used_" + str(i) + ","

    labels += "GPGSA_pdop,GPGSA_hdop,GPGSA_vdop,"

    for i in range(2):
        labels += "GNGSA" + str(i) + "_mode1,GNGSA" + str(i) + "_mode2,"

        for j in range(GSA_SAT_USED):
            labels += "GNGSA" + str(i) + "_sat_used_" + str(j) + ","

        labels += "GNGSA" + str(i) + "_pdop,GNGSA" + str(i) + "_hdop,GNGSA" + str(i) + "_vdop,"

    labels += "RMC_status,RMC_speed,RMC_course,RMC_mode\n"

    csv_file.write(labels)

    sample_index = 0
    features_length = len(dataset[0])
    for sample in dataset:
        entry = ""
        counter = 0

        for feature in sample:
            if counter < features_length - 1:
                entry += feature + ","
            else:
                entry += feature + "\n"

            counter += 1

        csv_file.write(str(dataset_labels[sample_index]) + ",")
        sample_index += 1
        csv_file.write(entry)

    csv_file.close()


def get_GGA_entry_as_array(entry):
    fields = entry.split(',')

    utc = check_if_null(fields[1])
    lat = check_if_null(fields[2])
    lat_dir = check_if_null(fields[3])
    long = check_if_null(fields[4])
    long_dir = check_if_null(fields[5])
    gps_qi = check_if_null(fields[6])  # ordinal
    sat_num = check_if_null(fields[7])  # categorical
    hor_dilution = check_if_null(fields[8])  # interval
    antenna_alt = check_if_null(fields[9])  # interval
    units_antenna_alt = check_if_null(fields[10])
    geoidal_sep = check_if_null(fields[11])  # interval
    units_geoidal_sep = check_if_null(fields[12])
    age_of_diff_gps_data = check_if_null(fields[13])  # interval
    diff_ref_station_id = check_if_null(fields[14].split('*')[0])

    # len(entry_array) = 7
    entry_array = [gps_qi, sat_num, hor_dilution, antenna_alt,
                   geoidal_sep, age_of_diff_gps_data, diff_ref_station_id]

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
    status = check_if_null(fields[2])  # categorical
    lat = check_if_null(fields[3])
    lat_dir = check_if_null(fields[4])
    long = check_if_null(fields[5])
    long_dir = check_if_null(fields[6])
    speed_over_ground = check_if_null(fields[7])  # interval
    course_over_ground = check_if_null(fields[8])  # interval
    date = check_if_null(fields[9])
    #    magnetic_variation = check_if_null(fields[10])
    mode = check_if_null(fields[12].split('*')[0])  # categorical

    # len(entry_array) = 4
    entry_array = [status, speed_over_ground, course_over_ground, mode]

    return entry_array


def get_GSA_entry_as_array(entry):
    fields = entry.split(",")

    mode_1 = check_if_null(fields[1])  # categorical
    mode_2 = check_if_null(fields[2])  # ordinal

    satellite_used_0 = check_if_null(fields[3])  # categorical
    satellite_used_1 = check_if_null(fields[4])  # categorical
    satellite_used_2 = check_if_null(fields[5])  # categorical
    satellite_used_3 = check_if_null(fields[6])  # categorical
    satellite_used_4 = check_if_null(fields[7])  # categorical
    satellite_used_5 = check_if_null(fields[8])  # categorical
    satellite_used_6 = check_if_null(fields[9])  # categorical
    satellite_used_7 = check_if_null(fields[10])  # categorical
    satellite_used_8 = check_if_null(fields[11])  # categorical
    satellite_used_9 = check_if_null(fields[12])  # categorical
    satellite_used_10 = check_if_null(fields[13])  # categorical
    satellite_used_11 = check_if_null(fields[14])  # categorical

    pdop = check_if_null(fields[15])  # interval
    hdop = check_if_null(fields[16])  # interval
    vdop = check_if_null(fields[17].split('*')[0])  # interval

    # len(entry_array) = 17

    entry_array = [mode_1, mode_2, satellite_used_0, satellite_used_1, satellite_used_2, satellite_used_3,
                   satellite_used_4,
                   satellite_used_5, satellite_used_6, satellite_used_7, satellite_used_8, satellite_used_9,
                   satellite_used_10, satellite_used_11, pdop, hdop, vdop]

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

    n_of_messages = int(fields[1])
    message_number = int(fields[2])
    total_satellites = int(fields[3])

    if message_number < n_of_messages:
        satellites_for_entry = MAX_SATELLITES_FOR_ENTRY
    else:
        satellites_for_entry = total_satellites - (MAX_SATELLITES_FOR_ENTRY * (n_of_messages - 1))

    # we consider only once the number of satellites since it could be
    # an important attribute

    # since we could have multiple GSV sentences, we consider it only at the first message
    if message_number == 1:
        entry_array = [str(total_satellites)]  # categorical
    else:
        entry_array = []

    # we will append, for each, satellite, 4 attributes
    # SV PRN number -> categorical
    # elevation -> interval
    # azimuth -> interval
    # SNR -> interval

    max_range = 4 + (4 * satellites_for_entry)
    for i in range(4, max_range):
        if i != max_range - 1:
            entry_array.append(check_if_null(fields[i]))
        else:
            entry_array.append(check_if_null(fields[i].split('*')[0]))

    return entry_array


def get_GSV_entry_as_array(messages):
    """
    Gets as input an array of GSV messages and returns an array of
    [total_SVs, prn_0, elevation_0, azimuth_0, snr_0, ..., prn_15, elevation_15, azimuth_15, snr_15]
    :param messages: array of strings
    :return: array of strings
    """

    LEN_GPGSV_ENTRY = 65
    LEN_GLGSV_ENTRY = 49
    entry_type = messages[0].split(',')[0]
    if entry_type == "$GPGSV":
        LEN_ENTRY = LEN_GPGSV_ENTRY
    else:
        LEN_ENTRY = LEN_GLGSV_ENTRY
    entry_array = []

    for message in messages:
        single_message_as_array = get_single_GSV_entry_as_array(message)
        entry_array += single_message_as_array

    while len(entry_array) != LEN_ENTRY:
        entry_array.append('-1')

    return entry_array


def get_GGA_entry_as_string(entry):
    entry_array = get_GGA_entry_as_array(entry)
    return " ".join(entry_array)


def utc_to_sin_cos(utc):
    """
    Transforms a time value expressed in UTC into sin, cos values
    :param utc: String
    :return: Float, Float
    """

    # in order to convert to polar coordinates we do the following reasoning:
    # rads : 2PI = secs : total_secs
    # rads = (secs * 2PI) / total_secs
    # rads = (secs * PI) / (total_secs / 2)
    TOTAL_SECS_PER_DAY = 24 * 60 * 60 / 2

    hours = int(utc[:2])
    mins = int(utc[2:4])
    secs = int(utc[4:6])
    total_secs = hours*60 + mins*60 + secs

    return sin(pi * total_secs / TOTAL_SECS_PER_DAY), cos(pi * total_secs / TOTAL_SECS_PER_DAY)


def lat_long_to_sin_cos(nmea_lat, type="lat", direction="N"):
    """
    Transforms NMEA latitude (ddmm.mmmm) into sin, cos values
    :param nmea_lat: String
    :return: Float, float
    """

    if type == "lat":
        if direction == "S":
            degrees = (float(nmea_lat[0:2]) + float(nmea_lat[2:]) / 60) * -1
        else:
            degrees = float(nmea_lat[0:2]) + float(nmea_lat[2:]) / 60
        degrees += 90
        TOTAL = 180 / 2
    else:
        if direction == "W":
            degrees = (float(nmea_lat[0:3]) + float(nmea_lat[3:]) / 60) * -1
        else:
            degrees = float(nmea_lat[0:3]) + float(nmea_lat[3:]) / 60
        degrees += 180
        TOTAL = 360 / 2

    return sin(pi * degrees / TOTAL), cos(pi * degrees / TOTAL)


def transform_data_into_numeric(file):
    sample_file = open(file, 'r')
    entries = []
    stable = False

    for line in sample_file:
        fields = line.split(",")
        message_ID = fields[0]

        # check if we have become stable
        if not stable:
            # we'll never become stable if not GPGGA
            if message_ID != "$GPGGA":
                continue
            else:
                gps_qi = int(check_if_null(fields[6]))
                if gps_qi != 0:
                    stable = True
                else:
                    continue
        # we don't use an else since it could happen that we became stable and need to start immediately
        # an else would make us loose the first elements
        if stable:
            if message_ID == "$GPGGA":
                sin_time, cos_time = utc_to_sin_cos(check_if_null(fields[1]))
                sin_lat, cos_lat = lat_long_to_sin_cos(check_if_null(fields[2]), "lat", fields[3])
                sin_long, cos_long = lat_long_to_sin_cos(check_if_null(fields[4]), "long", fields[5])


            else:
                continue


def check_if_null(elem):
    """
    Checks if a string is '', if so, it returns '-1'
    :param elem: string
    :return: string
    """
    if elem == '':
        return '-1'
    else:
        return elem
