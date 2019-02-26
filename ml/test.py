from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.data import get_data_from_file_2, concatenate_files, transform_data_for_numeric_into_CSV, get_sv_info_from_file
import numpy as np

def main():

    transform_data_for_numeric_into_CSV(["../data/true_data.txt", "../data/day39spoof11utc.txt"], [0, 1], True, csv_name="../data/day39spoofwithtruedata.csv")

    return


if __name__ == '__main__':
    main()
