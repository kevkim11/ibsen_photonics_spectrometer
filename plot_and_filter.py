import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

from matricies import *
from baseline_subtraction_variables import *
from os.path import join

from datetime import datetime
"""
Author: Kevin
Date: 10/13/16
Designed to use with the new ibsen output file
Still need to convert to csv file though.
"""
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

def k_filter(panda_table, steps):
    """
    FINISHED
    takes in a table and adds new columns to that table.
    """
    # Local Variable
    list_of_column_values = list(panda_table.columns.values) # change this to index/row
    # Copy the table
    df = panda_table.copy()
    for i in range(1, (len(list_of_column_values)-steps*2)+1):
        columns = panda_table.ix[:, i: i+(steps*2)]
        df[i+steps] = columns.mean(axis=1)
    # Filter the new DataFrame
    # Drop the left side of the DataFrame
    for i in range(1, steps+1):
        # print i
        df.drop(i, axis=1, inplace=True)
    # Drop the right side of the DataFrame
    for i in range(len(list_of_column_values), (len(list_of_column_values)-steps), -1):
        # print i
        df.drop(i, axis=1, inplace=True)
    return df

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def time_filter_single(l, steps):
    list_of_time_filtered = []
    for i in range(len(l)):
        ave = l[i:i+(steps*2)]
        aved = mean(ave)
        list_of_time_filtered.append(aved)
    return list_of_time_filtered

def time_filter(list_of_list, steps):
    new_list_of_list = []
    for i in list_of_list:
        new_list_of_list.append(time_filter_single(i, steps))
    return new_list_of_list

def k_filter2(panda_table, steps):
    """
    :param panda_table - Pandas dataframe w/ column_label = time and row_label = wavelengths (float)
    :param steps - the number of rows used to calculate the average. Formula is new_ave = (steps*2) +1

    """
    # Local Variable
    list_of_index_values = list(panda_table.index.values)
    # Copy the table
    df = panda_table.copy()
    # Convert the wavelength into increments starting at 1, then convert back later
    increments = []
    for i in range(1, len(list_of_index_values)+1):
        increments.append(i)
    df.index = increments
    panda_table.index = increments
    # Filter
    for i in range(1, (len(list_of_index_values)-steps*2)+1):
        row = panda_table.loc[i: i+(steps*2)]
        print i
        print i+(steps*2)
        df.loc[i+steps] = row.mean(axis=0)
    # Drop the left side of the DataFrame
    for i in range(1, steps+1):
        df.drop(i, axis=0, inplace=True)
        list_of_index_values.pop(0)
    # Drop the right side of the DataFrame
    for i in range(len(list_of_index_values), (len(list_of_index_values)-steps), -1):
        df.drop(i, axis=0, inplace=True)
        list_of_index_values = list_of_index_values[:-1]
    df.index = list_of_index_values
    return df


def k_filter3(panda_table, steps, flu=905, joe=956, tmr=1000, cxr=1037, wen=1167):
    """
    Made the k_filter2 more efficient by just combining k_filter2 and get_five_dyes
    :param panda_table:
    :param steps:
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
        a = i-steps
        b = i+steps
        row = panda_table.iloc[i-steps:i+steps+1]
        filtered_data.append(list(row.mean(axis=0)))
    return filtered_data

def plot_dyes(list_list_dyes):
    fig = plt.figure()
    plot = fig.add_subplot(111)

    x_axis = [x for x in range(len(list_list_dyes[0]))]
    # Plotting the actual color. Comment out if you don't want to plot one of them
    # plot.plot(x_axis, list_list_dyes[0], c="blue", label='Flu')
    # plot.plot(x_axis, list_list_dyes[1], c="green", label='Joe')
    # plot.plot(x_axis, list_list_dyes[2], c="orange", label='TMR')
    # plot.plot(x_axis, list_list_dyes[3], c="red", label='CXR')
    plot.plot(x_axis, list_list_dyes[4], c="black", label='WEN')
    """
    Set the x and y coordinate labels
    """
    plot.set_xlabel('quarter Seconds')
    plot.set_ylabel('ADC-Counts')
    """
    delta click event function
    """
    # Keep track of x/y coordinates, part of the find_delta_onclick
    xcoords = []
    ycoords = []
    def find_delta_onclick(event):
        global ix, iy
        global coords
        ix, iy = event.xdata, event.ydata
        xcoords.append(ix)
        ycoords.append(iy)
        print 'x = %s, y = %s' % (ix, iy)
        if len(xcoords) % 2 == 0:
            delta_x = abs(xcoords[-1] - xcoords[-2])
            delta_y = abs(ycoords[-1] - ycoords[-2])
            print 'delta_x = %d, delta_y = %d' % (delta_x, delta_y)
        coords = [ix, iy]
        return coords
    # connect the onclick function to the to mouse press
    fig.canvas.mpl_connect('button_press_event', find_delta_onclick)
    """
    add a for each plot
    """
    legend = plt.legend(loc='upper left', fontsize='small')
    return plot

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

def baseline_subtraction(list_list_dyes, sub):
    """
    Baseline subtraction based on lists of each dye and a list of the subtraction number.
    :param r1: list - Flu
    :param r2: list - Joe
    :param r3: list - TMR
    :param r4: list - CXR
    :param r5: list - WEN
    :param sub: list of ints to subtract from each element in the dye
    :return: each element in l1 with the baseline subtraction.
    """
    flu = []
    joe = []
    tmr = []
    cxr = []
    wen = []
    for a, b, c, d, e in itertools.izip(list_list_dyes[0], list_list_dyes[1], list_list_dyes[2], list_list_dyes[3], list_list_dyes[4]):
        flu.append(a-sub[0])
        joe.append(b-sub[1])
        tmr.append(c-sub[2])
        cxr.append(d-sub[3])
        wen.append(e-sub[4])

    return [flu, joe, tmr, cxr, wen]


def matrix_correction(list_list_dyes, matrix):
    """
    Matrix Correction (5 x 5 matrix).
    :param r1: list - Flu
    :param r2: list - Joe
    :param r3: list - TMR
    :param r4: list - CXR
    :param r5: list - WEN
    :param matrix: numpy array matrix
    :return:
    """
    flu = []
    joe = []
    tmr = []
    cxr = []
    wen = []
    for a, b, c, d, e in itertools.izip(list_list_dyes[0], list_list_dyes[1], list_list_dyes[2], list_list_dyes[3], list_list_dyes[4]):
        vector = np.array([a, b, c, d, e])
        dot_product = matrix.dot(vector)
        flu.append(dot_product[0])
        joe.append(dot_product[1])
        tmr.append(dot_product[2])
        cxr.append(dot_product[3])
        wen.append(dot_product[4])
    return [flu, joe, tmr, cxr, wen]


def set_threshold(list_list_dyes, threshold_value):
    """
    Set a threshold so that all values below that is threshold is set to the threshold value.
    :param r1: list - Flu
    :param r2: list - Joe
    :param r3: list - TMR
    :param r4: list - CXR
    :param r5: list - WEN
    :param threshold_value:
    :return: Modified data with the threshold value
    """
    flu2 = []
    joe2 = []
    tmr2 = []
    cxr2 = []
    wen2 = []
    [flu2.append(threshold_value) if i < threshold_value else flu2.append(i) for i in list_list_dyes[0]]
    [joe2.append(threshold_value) if i < threshold_value else joe2.append(i) for i in list_list_dyes[1]]
    [tmr2.append(threshold_value) if i < threshold_value else tmr2.append(i) for i in list_list_dyes[2]]
    [cxr2.append(threshold_value) if i < threshold_value else cxr2.append(i) for i in list_list_dyes[3]]
    [wen2.append(threshold_value) if i < threshold_value else wen2.append(i) for i in list_list_dyes[4]]

    return [flu2, joe2, tmr2, cxr2, wen2]

def k_baseline_subtraction(list_list_dyes):
    """
    Kevin's baseline subtraction algorithm.
    :param list_list_dyes:
    :return:
    """


if __name__ == "__main__":
    # Folder where csv files are
    folder = 'csv_files'
    # file name variables
    AL = '10_13_AL_new_ibsen.csv'
    mat = '10_14_matrix.csv'
    AL2 = '10_26_9mW_AL_(actual).csv'
    file_name = AL
    file_name2 = AL2

    file_dir = join(folder, file_name)
    file_dir2 = join(folder, file_name2)
    """1 load file"""
    pd1 = load_file(file_dir)
    pd2 = load_file(file_dir2)

    """2 Filter"""
    startTime = datetime.now()
    print "filter start time is "+str(startTime)
    filtered_data1 = k_filter3(pd1, 8)
    filtered_data2 = k_filter3(pd2, 8)
    # filtered_data3 = k_filter3(pd1, 20)
    # filtered_data3 = k_filter2(pd, 20)
    print "filter end time is " + str(datetime.now() - startTime)

    """3 Get 5 dyes"""
    x1 = get_five_dyes(pd1)
    twentysix = get_five_dyes(pd2)
    # x9 = get_five_dyes(filtered_data1)
    # x2 = get_five_dyes(filtered_data2)
    # x3 = get_five_dyes(filtered_data3)
    # a1 = get_five_dyes(filtered_data1)
    # b1 = get_five_dyes(filtered_data2)
    # c1 = get_five_dyes(filtered_data3)
    print "get dyes end time is " + str(datetime.now() - startTime)

    """4 Baseline Subtraction"""
    y1 = baseline_subtraction(filtered_data1, ZERO_subtraction)
    y2 = baseline_subtraction(filtered_data2, ZERO_subtraction)
    # y4 = baseline_subtraction(filtered_data3, ZERO_subtraction)

    # r1 = baseline_subtraction(a1, ZERO_subtraction)
    # s1 = baseline_subtraction(b1, ZERO_subtraction)
    # r1 = baseline_subtraction(filtered_data1, matrix_subtraction_10mW)
    # s1 = baseline_subtraction(filtered_data2, matrix_subtraction_9mW)

    # t1 = baseline_subtraction(c1, ZERO_subtraction)
    print "baseline_subtraction end time is " + str(datetime.now() - startTime)

    """5 Matrix Correction"""
    mat1 = matrix_correction(y1, matrix_MOD)
    mat2 = matrix_correction(y2, matrix_MOD2)
    # mat11 = matrix_correction(twentysix, matrix_MOD2)
    print "matrix correction end time is " + str(datetime.now() - startTime)

    """5.5 Time averaging/filter"""
    # t1 = time_filter(y1, 8)
    # t2 = time_filter(mat1, 8)
    # t3 = time_filter(mat1, 14)
    # t4 = time_filter(mat1, 20)
    """6 Plot Data"""
    # p1 = plot_dyes(z)
    # p1.set_title(file_name + "__No filter")
    p4 = plot_dyes(mat1)
    p4.set_title(file_name + "8_step_filter, no baseline subtraction")
    p4.grid(True)
    p2=plot_dyes(mat2)
    p2.set_title(file_name2 + "8_step_filter, no baseline subtraction")
    p2.grid(True)
    # p3=plot_dyes(mat3)
    # p3.set_title(file_name + "__20")
    # p8 = plot_dyes(twentysix)
    # p8.set_title(file_name2 + " Raw")
    # p7 = plot_dyes(mat11)
    # p7.set_title(file_name2 + " Matrix correction")

    # p10 = plot_dyes()
    # p10.set_title()


    """Filter w/ Timing"""
    # startTime = datetime.now()
    # print startTime
    # Filter
    # new_data2 = k_filter2(new_data, 7) # filter
    # new_data2.to_csv('filtered_'+file_name) # output
    # print datetime.now() - startTime

    print "done"
    plt.show()