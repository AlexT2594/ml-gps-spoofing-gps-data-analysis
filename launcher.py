import argparse
import ml.LatLongAnalysis2
import ml.RealTimeAttackSim
import utils.data as utils

import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPS Spoofing Detection System',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', type=str, metavar='filename', nargs='+', required=True,
                        help='filename of NMEA logs with the same label')
    parser.add_argument('-l', type=int, metavar='integer_label', nargs='+', choices=[0, 1], required=True,
                        help='integer label for corresponding log\noptions: 0 (not spoofed)\n\t 1 (spoofed)')
    parser.add_argument('-m', type=str, metavar='mode', nargs=1, required=True, choices=['latlong', 'numeric'],
                        help='options: latlong\n\t numeric')

    args = parser.parse_args()
    print(args)

    OUTPUT_FILENAME = "data.txt"
    OUTPUT_FILENAME_CSV = "data.csv"
    filenames = args.f
    labels = args.l
    mode = args.m

    if len(filenames) != len(labels):
        raise Exception('Number of provided filenames and labels must be the same.')

    if mode[0] == 'latlong':
        print("==> Latitude/Longitude analysis mode")
        print("    Description: Not spoofed logs are used for generating a ground truth.\n"
              "                 At least one such log file is needed.")

        if 0 not in labels:
            raise Exception('At least one not spoofed log is needed.')

        fnames = []

        for fname, label in zip(filenames, labels):
            if label == 0:
                fnames.append(fname)

        utils.concatenate_files(fnames, OUTPUT_FILENAME)
        ml.LatLongAnalysis2.main(OUTPUT_FILENAME)
        os.remove(OUTPUT_FILENAME)

    else:
        print("==> Numeric analysis mode")
        print("    Does a real-time analysis on relevant numeric attributes of the signal.")

        utils.transform_data_for_sv_info_DOP_analysis_into_CSV(filenames, labels, csv_name=OUTPUT_FILENAME_CSV)
        ml.RealTimeAttackSim.main(OUTPUT_FILENAME_CSV)
        os.remove(OUTPUT_FILENAME_CSV)


