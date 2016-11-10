import matplotlib.pyplot as plt
import pandas as pd

import itertools

# Custom Modules
from matricies import *
# from baseline_subtraction_variables import *
from k_baseline import line_scanner, K_Baseline, baseline_subtraction_class

from os.path import join

from datetime import datetime

"""
Author: Kevin Kim
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

"""
*** Time Filter functions
"""


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


def k_filter(panda_table, steps):
    """
    FINISHED
    takes in a table and adds new columns to that table.
    """
    # Local Variable
    list_of_column_values = list(panda_table.columns.values) # change this to index/row
    # Copy the table
    df = panda_table.copy()
    # Convert  the time into incremenets starting at 1, then convert back later...
    increments = []
    for i in range(1,len(list_of_column_values)+1):
        increments.append(i)
    df.columns = increments
    panda_table.columns = increments
    # Filter
    for i in range(1, (len(list_of_column_values)-steps*2)+1):
        columns = panda_table.ix[:, i: i+(steps*2)]
        df.loc[i+steps] = columns.mean(axis=1)
    # Filter the new DataFrame
    # Drop the left side of the DataFrame
    for i in range(1, steps+1):
        # print i
        df.drop(i, axis=1, inplace=True)
        list_of_column_values.pop(0)
    # Drop the right side of the DataFrame
    for i in range(len(list_of_column_values), (len(list_of_column_values)-steps), -1):
        # print i
        df.drop(i, axis=1, inplace=True)
        list_of_column_values = list_of_column_values[:-1]
    df.columns = list_of_column_values
    return df


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
        # a = i - pixel_steps
        # b = i + pixel_steps
        row = panda_table[i - pixel_steps:i + pixel_steps + 1]
        time_filtered_data = row.copy()
        list_of_row_values = list(row.columns.values)
        for i in range(1, (len(list_of_row_values) - time_steps* 2) + 1):
            columns = row.ix[:, i: i + (time_steps* 2)]
            time_filtered_data.ix[:, i + time_steps] = columns.mean(axis=1)
        time_averaged = list(time_filtered_data.mean(axis=0))
        filtered_data.append(time_averaged)
    return filtered_data


def plot_one_line(list_of_values, color="blue", label="label"):
    """

    :param list_of_values:
    :return:
    """
    fig = plt.figure()
    plot = fig.add_subplot(111)

    x_axis = [x for x in range(len(list_of_values))]
    plot.plot(x_axis, list_of_values, c=color, label=label)
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

def plot_dyes(list_list_dyes, list_of_baseline_x = [], list_of_baseline_y = [], scatter = False):
    """

    :param list_list_dyes:
    :param list_of_baseline_x: optional list - list of x's to plot for the baseline to be subtracted
    :param list_of_baseline_y: optional list - list of y's to plot for the baseline to be subtracted
    :param scatter:
    :return:
    """
    fig = plt.figure()
    plot = fig.add_subplot(111)

    x_axis = [x for x in range(len(list_list_dyes[3]))]

    if scatter == True and len(list_of_baseline_x)!= 0:
        plot.scatter(list_of_baseline_x, list_of_baseline_y, c="black", label='Scatter')
    elif scatter == False and len(list_of_baseline_x)!= 0:
        plot.plot(list_of_baseline_x, list_of_baseline_y, c="black", label='Plot')
    # plot.plot(x_axis, list_list_dyes[0], c="blue", label='Flu')
    # plot.plot(x_axis, list_list_dyes[1], c="green", label='Joe')
    # plot.plot(x_axis, list_list_dyes[2], c="orange", label='TMR')
    plot.plot(x_axis, list_list_dyes[3], c="red", label='CXR')
    # plot.plot(x_axis, list_list_dyes[4], c="black", label='WEN')
    # plot.plot(x_axis, y_axis, c="blue", label='Flu')

    # plot.scatter(list_of_baseline_x, list_of_baseline_y, c="black", label='New Baseline')
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

    Subtracts each dye in list_list_dyes with the respective ints in sub
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

def baseline_subtraction_steps_main(file_dir):
    """

    :param file_dir:
    :return:
    """
    """1) load file"""
    pd = load_file(file_dir)
    """2) Get 5 dyes"""
    list_of_list_dyes = get_five_dyes(pd)
    """3) K_Baseline Subtraction"""
    line_scanner1 = line_scanner(list_of_list_dyes[3])
    l1 = line_scanner1.find_all_local_min(threshold_value=50000)
    k_baseline1 = K_Baseline(l1)
    x_and_y_dict = k_baseline1.populate_x_and_y(list_of_list_dyes[3])
    bs1 = baseline_subtraction_class(list_of_list_dyes[3])
    a = bs1.perform_baseline_subtraction()
    """4) Plot Data"""
    q1 = plot_dyes(list_of_list_dyes, list_of_baseline_x=x_and_y_dict["x/quarter seconds"],
                   list_of_baseline_y=x_and_y_dict["y/best-fit line"], scatter=False)
    q1.set_title(str(file_dir))
    q2 = plot_one_line(a, color="red")
    q2.set_title("Baseline subtracted")
    print "done"
    plt.show()

def previous_main(file_dir):
    """1 load file"""
    pd1 = load_file(file_dir)
    # pd2 = load_file()

    """2 Filter"""
    startTime = datetime.now()
    print "filter start time is " + str(startTime)
    # filtered_data1 = k_filter3(pd1, 8)
    # filtered_data2 = k_filter3(pd2, 8)
    # filtered_data3 = k_filter3(pd1, 20)
    # filtered_data3 = k_filter2(pd, 20)
    print "filter end time is " + str(datetime.now() - startTime)

    """3 Get 5 dyes"""
    x1 = get_five_dyes(pd1)

    line_scanner1 = line_scanner(x1[3])
    l1 = line_scanner1.find_all_local_min()
    k_baseline1 = K_Baseline(l1)
    x_and_y_dict = k_baseline1.populate_x_and_y(x1[3])

    bs1 = baseline_subtraction_class(x1[3])
    a = bs1.perform_baseline_subtraction()

    # q1 = plot_dyes(x1)
    # q1 = plot_dyes(x1, list_of_baseline_x=x_and_y_dict["x/quarter seconds"],
    #                list_of_baseline_y=x_and_y_dict["y/best-fit line"], scatter=False)
    q1 = plot_dyes(x1, list_of_baseline_x=x_and_y_dict["x/quarter seconds"],
                   list_of_baseline_y=x_and_y_dict["y/best-fit line"], scatter=False)
    p2 = plot_one_line(a, label="k_baseline subtracted")
    p2.set_title("Actually subtracted")
    q1.set_title("k_baseline")
    twentysix = get_five_dyes(pd2)
    # x9 = get_five_dyes(filtered_data1)
    # x2 = get_five_dyes(filtered_data2)
    # x3 = get_five_dyes(filtered_data3)
    # a1 = get_five_dyes(filtered_data1)
    # b1 = get_five_dyes(filtered_data2)
    # c1 = get_five_dyes(filtered_data3)
    print "get dyes end time is " + str(datetime.now() - startTime)

    """4 Baseline Subtraction"""
    # y1 = baseline_subtraction(filtered_data1, ZERO_subtraction)
    # y2 = baseline_subtraction(filtered_data2, ZERO_subtraction)
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
    # p4 = plot_dyes(mat1)
    # p4.set_title(file_name + "8_step_filter, no baseline subtraction")
    # p4.grid(True)
    # p2=plot_dyes(mat2)
    # p2.set_title(file_name2 + "8_step_filter, no baseline subtraction")
    # p2.grid(True)


    """Filter w/ Timing"""
    # startTime = datetime.now()
    # print startTime
    # Filter
    # new_data2 = k_filter2(new_data, 7) # filter
    # new_data2.to_csv('filtered_'+file_name) # output
    # print datetime.now() - startTime

    print "done"
    plt.show()

def proper_main(file_dir):
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

    """1) load file"""
    pd = load_file(file_dir)
    print "File done loading"
    """2) Time Averaging/K_filter/Get 5 dyes"""
    kfiltered_list_of_list = k_filter4(pd, pixel_steps=8, time_steps=2)
    print "k_filter4 is done"
    """3) Matrix Correction"""
    matrix_corrected_list_of_list = matrix_correction(kfiltered_list_of_list, matrix_MOD2)
    print "Matrix Correction is done"
    """4) baseline_subtraction"""
    bs1_subs = []
    for i in matrix_corrected_list_of_list:
        bs1 = baseline_subtraction_class(i)
        bs1_sub = bs1.perform_baseline_subtraction()
        bs1_subs.append((bs1_sub))
    print "k_baseline subtraction is done"
    """5) plot"""
    p1 = plot_dyes(matrix_corrected_list_of_list)
    print "done"
    plt.show()

def main(file_dir):
    """

    :param file_dir:
    :return:
    """
    """1) load file"""
    pd = load_file(file_dir)
    """2) Get 5 dyes"""
    list_of_list_dyes = get_five_dyes(pd)
    """3) K_Baseline Subtraction"""
    line_scanner1 = line_scanner(list_of_list_dyes[3])
    l1 = line_scanner1.find_all_local_min()
    k_baseline1 = K_Baseline(l1)
    x_and_y_dict = k_baseline1.populate_x_and_y(list_of_list_dyes[3])

    bs1 = baseline_subtraction_class(list_of_list_dyes[3])
    a = bs1.perform_baseline_subtraction()
    """4) Plot Data"""
    plot1 = plot_dyes(list_of_list_dyes, list_of_baseline_x=l1["x/quarter seconds"], list_of_baseline_y=l1["y/best-fit line"], scatter=True)
    plot1.set_title(str(file_dir))
    # q2 = plot_one_line(a, color="red")
    # q2.set_title("Baseline subtracted")
    print "done"
    plt.show()

if __name__ == "__main__":
    '''
    *** File Directory
    '''
    # Folder where csv files are
    folder = '../csv_files'
    # file name variables
    AL = '10_13_AL_new_ibsen.csv'
    # mat = '10_14_matrix.csv'
    AL2 = '10_26_9mW_AL_(actual).csv'
    file_name = AL
    file_name2 = AL2

    file_dir = join(folder, file_name)
    file_dir2 = join(folder, file_name2)

    main(file_dir)

    # baseline_subtraction_steps_main(file_dir)
