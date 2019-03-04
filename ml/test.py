from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.data import get_sv_DOP_info_from_file, transform_data_for_numeric_into_CSV, \
    transform_data_for_sv_info_DOP_analysis_into_CSV, get_DOP_info_from_file, \
    transform_data_for_DOP_analysis_into_CSV, nmea_log_to_entry, get_sv_info_from_single_entry, \
    concatenate_files
import numpy as np

def main():

    concatenate_files(['../data/true_data.txt', '../data/day39spoof11utc.txt'], '../data/true_with_spoof.txt')


if __name__ == '__main__':
    main()
