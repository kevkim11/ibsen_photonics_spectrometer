import matplotlib.pyplot as plt
import pandas as pd

import itertools

# Custom Modules
from matricies import *
# from baseline_subtraction_variables import *
from k_baseline import line_scanner, K_Baseline, baseline_subtraction_class

from os.path import join

from datetime import datetime

# folder = '../csv_files'
# folder_name = 'baseline_subtraction.csv'
#
# file_dir = join(folder, folder_name)
#
# a = pd.read_csv(file_dir, index_col=0)
#
# data = a.transpose()
# cxr = data.iloc[0].tolist()
#
# flu = []
# joe = []
# tmr = []
# wen = []
# l_of_l = [flu, joe, tmr, cxr, wen]
startTime = datetime.now()

def load_file(csv):
    """
    :param csv is a string
    Loads a csv file and converts the column_labels into increments starting from 1
    and the index (row_label) values to a float rounded to the nearest tenth place.
    """
    a = pd.read_csv(csv, index_col=0)
    data = a.transpose()
    # Convert the columns into just increments
    times = list(data.columns.values)
    steps = []
    for i in range(1, len(times) + 1):
        steps.append(i)
    data.columns = steps  # assign data's time into steps
    return data

def k_filter4(panda_table, pixel_steps, time_steps, flu=905, joe=956, tmr=1000, cxr=1037, wen=1167):
    """
    Combination of time_averaging and k_filter.
    Needed to combine so that I can make the filtering efficient by only
    taking the subsets out.

    Made the k_filter2 more efficient by just combining k_filter2 and get_five_dyes
    :param panda_table:
    :param pixel_steps:
    :param flu:
    :param joe:
    :param tmr:
    :param cxr:
    :param wen:
    :return:
    """
    list_of_pixel_numbers = [flu, joe, tmr, cxr, wen]
    filtered_data = []
    for i in list_of_pixel_numbers:
        row = panda_table[i - pixel_steps:i + pixel_steps + 1]
        time_filtered_data = row.copy()
        list_of_row_values = list(row.columns.values)
        for i in range(1, (len(list_of_row_values) - time_steps* 2) + 1):
            columns = row.ix[:, i: i + (time_steps* 2)]
            time_filtered_data.ix[:, i + time_steps] = columns.mean(axis=1)
        # """ Output filtered Data"""
        # time_filtered_data.to_csv('k4_filtered_' + file_name)  # output
        time_averaged = list(time_filtered_data.mean(axis=0))
        filtered_data.append(time_averaged)
    filtered_df = pd.DataFrame(filtered_data, index=list_of_pixel_numbers)
    """ Output filtered Data"""
    filtered_df.to_csv('../csv_files/k4_filtered_'+ str(pixel_steps)+"X"+str(time_steps) + "_" +file_name)  # output
    print "k_filter4 and output is done " + str(datetime.now() - startTime)
    return filtered_data

def load_filtered_csv(csv):
    data = pd.read_csv(csv, index_col=0)
    # Convert the columns into just increments
    # times = list(data.columns.values)
    # steps = []
    # for i in range(1, len(times) + 1):
    #     steps.append(i)
    # data.columns = steps  # assign data's time into steps
    return data

def get_five_dyes(data, flu=905, joe=956, tmr=1000, cxr=1037, wen=1167):
    """
    Returns five lists with flu, joe, tmr, and wen plots at their respective pixels/wavelength
    :param data: pandas dataframe
    :param flu: ints/pixels
    :param joe: ints/pixels
    :param tmr: ints/pixels
    :param cxr: ints/pixels
    :param wen: ints/pixels
    :return: lists
    """
    a1 = data.ix[flu].tolist()
    a2 = data.ix[joe].tolist()
    a3 = data.ix[tmr].tolist()
    a4 = data.ix[cxr].tolist()
    a5 = data.ix[wen].tolist()
    return [a1, a2, a3, a4, a5]

def main(file_dir):
    """
    The correct order of how to view the allelic ladder.
    1) Load file
    2) Time Averaging/K_filter/Get 5 dyes
    3) Matrix Correction
    4) K_Baseline Subtraction
    5) Plot
    :param file_dir:
    :return:
    """
    print "Starting at " + str(startTime)
    """1) load file"""
    # pd = load_file(file_dir)
    print "File done loading " + str(datetime.now() - startTime)
    # pd1 = data_compression(pd)
    # print "Compression done at " + str(datetime.now() - startTime)
    """2) Time Averaging/K_filter/Get 5 dyes"""
    # kfiltered_list_of_list = k_filter4(pd, pixel_steps=8, time_steps=4)
    pad = pd.read_csv(file_dir, index_col=0)
    none_filtered_list_of_list = get_five_dyes(pad)
    print "k_filter4 is done " + str(datetime.now() - startTime)
    print none_filtered_list_of_list
    # """3) Matrix Correction"""
    # matrix_corrected_list_of_list = matrix_correction(kfiltered_list_of_list, matrix_MOD_AL)
    # """ set_threshold..."""
    # matrix_corrected_list_of_list = set_threshold(matrix_corrected_list_of_list, 250)
    # # matrix_corrected_list_of_list2 = matrix_correction(kfiltered_list_of_list, ZERO_matrix)
    # # matrix_corrected_list_of_list2 = matrix_correction(none_filtered_list_of_list, matrix_MOD_AL)
    # print "Matrix Correction is done " + str(datetime.now() - startTime)
    # """4) K baseline_subtraction"""
    # line_scanner1 = line_scanner(matrix_corrected_list_of_list[3])
    # l1 = line_scanner1.find_all_local_min()
    #
    # # line_scanner2 = line_scanner(matrix_corrected_list_of_list2[3])
    # # l2 = line_scanner2.find_all_local_min()
    #
    # # k_baseline1 = K_Baseline(l1)
    # # x_and_y_dict = k_baseline1.populate_x_and_y(matrix_corrected_list_of_list[3])
    #
    # bs1 = baseline_subtraction_class(matrix_corrected_list_of_list[3])
    # a = bs1.perform_baseline_subtraction()
    #
    # print "k_baseline subtraction is done " + str(datetime.now() - startTime)
    # """5) plot"""
    # p1 = plot_dyes(matrix_corrected_list_of_list,
    #                list_of_baseline_x=l1["x/quarter seconds"],
    #                list_of_baseline_y=l1["y/best-fit line"],
    #                scatter=True)
    # p1.set_title(str("baseline subtracted"))
    # # p2 = plot_dyes(matrix_corrected_list_of_list2,
    # #                list_of_baseline_x=l2["x/quarter seconds"],
    # #                list_of_baseline_y=l2["y/best-fit line"],
    # #                scatter=True)
    # # p2.set_title(str("Not filtered"))
    # print "done " + str(datetime.now() - startTime)
    # plt.show()

if __name__ == "__main__":
    '''
    *** File Directory
    '''
    # Folder where csv files are
    folder = '../csv_files'
    # file name variables
    AL = 'k4_filtered_8X4_10_13_AL_new_ibsen_modified.csv'
    # mat = '10_14_matrix.csv'
    AL2 = '10_26_9mW_AL_(actual).csv'
    file_name = AL
    file_name2 = AL2

    file_dir = join(folder, file_name)
    file_dir2 = join(folder, file_name2)
    main(file_dir)
