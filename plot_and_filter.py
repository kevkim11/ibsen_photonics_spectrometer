"""
Author: Kevin Kim
Date: October 2016
Designed to use with the new ibsen output file

"""

# import itertool
from datetime import datetime
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import csv

from SG1_writer import SG1_Writer
from k_baseline import *
from matricies import *

from DataCursor import DataCursor

# variable that can keep track of running time
startTime = datetime.now()
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

# plotly api
import plotly
plotly.tools.set_credentials_file(username='kevkim11', api_key='eXCerKP3UZXa3I1mvVcf')
import plotly.plotly as py
import plotly.graph_objs as go

def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

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
"""
*** k_filter - Moving average filters
"""

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
    Moving average (None Efficient)
    Does a moving average on the entire dataframe.
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
    Moving average (Efficient)
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
        # a = i-steps
        # b = i+steps
        row = panda_table.iloc[i-steps:i+steps+1]
        filtered_data.append(list(row.mean(axis=0)))
    return filtered_data


def k_filter4(panda_table, pixel_steps, time_steps, flu=905, joe=956, tmr=1000, cxr=1037, wen=1167):
    """
    Combination of time_averaging and k_filter.
    Needed to combine so that I can make the filtering efficient by only
    taking the subsets out.

    Made the k_filter2 more efficient by just combining k_filter2 and get_five_dyes

    Also outputs a csv file of the filtered data. It's a 5 row df.
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

def k_peak_detection(list_of_list):
    """

    :param list_of_list:
    :return: list that contains two lists (x and y peaks)
    """
    list_of_peaks = [[],[]]
    threhold_value = 65
    for list in list_of_list:
        maxtab, mintab = peakdet(list, 10)
        local_max_x = array(maxtab)[:, 0]
        local_max_y = array(maxtab)[:, 1]
        print "found local max"
        # Filter the local_max_x and local_max_y
        peak_x = []
        peak_y = []
        for x, y in zip(local_max_x, local_max_y):
            if y > threhold_value:
                peak_x.append(x)
                peak_y.append(y)
        peak_x.pop(0)
        peak_y.pop(0)
        list_of_peaks[0].append(peak_x)
        list_of_peaks[1].append(peak_y)
    return list_of_peaks


def plot_ibsen_spectrum(list_of_values, color="blue", label="label"):
    """

    :param list_of_values:
    :return:
    """

    # fig, ax = plt.subplots()
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


def quart_sec_to_bp(quarter_seconds):

    return

def plotly_dyes(list_list_dyes):
    """

    :param list_list_dyes:
    :return:
    """
    x_axis = [x for x in range(len(list_list_dyes[3]))]

    data = [
        go.Scatter(
            x=x_axis,
            y=list_list_dyes[3],
            name='Plotly Plot of my data.',
        )
    ]

    layout = go.Layout(
        title="Number of songs listed in the Guardian's<br><em>Top 1,000 Songs to Hear Before You Die</em> per artist with 4 or more songs",
        font=dict(
            family='Georgia, serif',
            color='#635F5D'
        ),
        plot_bgcolor='#EFECEA'
    )
    fig = go.Figure(data=data, layout=layout)
    plot_url = py.plot(fig)

def plot_dyes(list_list_dyes, list_of_baseline_x = [], list_of_baseline_y = [], scatter = False):
    """
    Takes a list of list of the five dyes and then plots each of them.
    Commenting out some of the lines if I don't want to plot them.

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
        # Plot the annotations / coordinates that show up on the plot
        # for x_coord, y_coord, in zip(list_of_baseline_x, list_of_baseline_y):
        #     plot.annotate("("+str(x_coord) + ")", xy=(x_coord, y_coord))
    elif scatter == False and len(list_of_baseline_x)!= 0:
        plot.plot(list_of_baseline_x, list_of_baseline_y, c="red", label='Plot')
    plot.plot(x_axis, list_list_dyes[0], c="blue", label='Flu')
    plot.plot(x_axis, list_list_dyes[1], c="green", label='Joe')
    plot.plot(x_axis, list_list_dyes[2], c="orange", label='TMR')
    plot.plot(x_axis, list_list_dyes[3], c="red", label='CXR')
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

def plot_dyes2(list_list_dyes, list_of_peaks):
    """
    Plots the 5 dyes along with their peaks.

    :param list_list_dyes:
    :param list_of_peaks: list that contains two lists (x and y peaks)
    :param scatter:
    :return:
    """
    fig = plt.figure()
    plot = fig.add_subplot(111)

    x_axis = [x for x in range(len(list_list_dyes[3]))]

    plot.plot(x_axis, list_list_dyes[0], c="blue", label='Flu')
    plot.plot(x_axis, list_list_dyes[1], c="green", label='Joe')
    plot.plot(x_axis, list_list_dyes[2], c="orange", label='TMR')
    plot.plot(x_axis, list_list_dyes[3], c="red", label='CXR')
    plot.plot(x_axis, list_list_dyes[4], c="black", label='WEN')
    plot.scatter(list_of_peaks[0][0], list_of_peaks[1][0], c="red")
    plot.scatter(list_of_peaks[0][1], list_of_peaks[1][1], c="orange")
    plot.scatter(list_of_peaks[0][2], list_of_peaks[1][2], c="green")
    plot.scatter(list_of_peaks[0][3], list_of_peaks[1][3], c="blue")
    plot.scatter(list_of_peaks[0][4], list_of_peaks[1][4], c="purple")

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

def plot_dye_cursor(list_list_dyes, list_of_peaks):
    fig = plt.figure()
    plot = fig.add_subplot(111)

    x_axis = [x for x in range(len(list_list_dyes[0]))]

    plot.plot(x_axis, list_list_dyes[0], c="blue", label='Flu')

    scat = plot.scatter(list_of_peaks[0][0], list_of_peaks[1][0], c="red")
    DataCursor(scat, list_of_peaks[0][0], list_of_peaks[1][0])

    plot.set_xlabel('quarter Seconds')
    plot.set_ylabel('ADC-Counts')
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
    list_of_list = []
    for i in [flu, joe, tmr, cxr, wen]:
        list_of_list.append(data.ix[i].tolist())
    return list_of_list


def get_five_dyes2(data, flu=905, joe=956, tmr=1000, cxr=1037, wen=1167):
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
    list_of_list = []
    for i in [flu, joe, tmr, cxr, wen]:
        list_of_list.append(data.loc[str(i)].tolist())
    return list_of_list

def baseline_subtraction(list_list_dyes, sub):
    """
    Baseline subtraction based on lists of each dye and a list of the subtraction number.

    Subtracts each dye in list_list_dyes with the respective list of ints in sub
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
    :param list_list_dyes: contains 5 lists that correspond with the 5 dyes (Flu, Joe, TMR, CXR, and Wen)
    :param matrix: numpy array matrix in order to matrix correction
    :return: a modified
    """
    # Initialize lists for the matrix corrected values
    flu = []
    joe = []
    tmr = []
    cxr = []
    wen = []
    # Matrix Correction
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
    :param list_list_dyes: contains 5 lists that correspond with the 5 dyes (Flu, Joe, TMR, CXR, and Wen)
    :param threshold_value:
    :return: Modified data with the threshold value
    """
    flu2 = []
    joe2 = []
    tmr2 = []
    cxr2 = []
    wen2 = []
    [flu2.append(0) if i < threshold_value else flu2.append(i-threshold_value) for i in list_list_dyes[0]]
    [joe2.append(0) if i < threshold_value else joe2.append(i-threshold_value) for i in list_list_dyes[1]]
    [tmr2.append(0) if i < threshold_value else tmr2.append(i-threshold_value) for i in list_list_dyes[2]]
    [cxr2.append(0) if i < threshold_value else cxr2.append(i-threshold_value) for i in list_list_dyes[3]]
    [wen2.append(0) if i < threshold_value else wen2.append(i-threshold_value) for i in list_list_dyes[4]]
    # [modified_dye.append(0) if i < threshold_value else modified_dye.append(i) for i in dye]
    return [flu2, joe2, tmr2, cxr2, wen2]

"""
Main Functions
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
    line_scanner1 = line_scanner(list_of_list_dyes[4])
    # l1 = line_scanner1.find_all_local_min(threshold_value=50000)
    """3.1) Trying out peakdet"""
    maxtab, mintab = peakdet(list_of_list_dyes[4], 10)
    # l1 = line_scanner1.find_all_global_max()
    # k_baseline1 = K_Baseline(l1)
    # x_and_y_dict = k_baseline1.populate_x_and_y(list_of_list_dyes[3])
    # bs1 = baseline_subtraction_class(list_of_list_dyes[3])
    # a = bs1.perform_baseline_subtraction()
    """4) Plot Data"""
    # q1 = plot_dyes(list_of_list_dyes, list_of_baseline_x=l1["x/quarter seconds"],
    #                list_of_baseline_y=l1["y/best-fit line"], scatter=True)
    q1 = plot_dyes(list_of_list_dyes, list_of_baseline_x=array(maxtab)[:, 0],
                   list_of_baseline_y=array(maxtab)[:, 1], scatter=True)
    q1.set_title(str(file_dir))
    # q2 = plot_ibsen_spectrum(a, color="red")
    # q2.set_title("Baseline subtracted")
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


def main2(file_dir):
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
    l1 = line_scanner1.find_all_local_max()

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
    pd = load_file(file_dir)
    print "File done loading " + str(datetime.now() - startTime)
    # pd1 = data_compression(pd)
    # print "Compression done at " + str(datetime.now() - startTime)
    """2) Time Averaging/K_filter/Get 5 dyes"""
    kfiltered_list_of_list = k_filter4(pd, pixel_steps=8, time_steps=10)
    # none_filtered_list_of_list = get_five_dyes(pd)
    print "k_filter4 is done " + str(datetime.now() - startTime)
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

def main4(filtered_file_dir, raw_file_dir):
    """
    Main to load processed data's into and plot
    """
    print "Starting at " + str(startTime)
    """1) load file"""
    processed_data = pd.read_csv(filtered_file_dir, index_col=0)
    print "File done loading " + str(datetime.now() - startTime)
    kfiltered_list_of_list = get_five_dyes(processed_data)
    print "k_filter4 is done " + str(datetime.now() - startTime)
    """2) load raw data"""
    raw_data = load_file(raw_file_dir)
    kfiltered3_list_of_list = k_filter3(raw_data, steps=8)
    """3) Matrix Correction"""
    matrix_corrected_list_of_list = matrix_correction(kfiltered_list_of_list, matrix_MOD_AL)
    matrix_corrected_list_of_list_RAW = matrix_correction(kfiltered3_list_of_list, matrix_MOD_AL)
    """ set_threshold..."""
    matrix_corrected_list_of_list = set_threshold(matrix_corrected_list_of_list, 250)
    # matrix_corrected_list_of_list2 = matrix_correction(kfiltered_list_of_list, ZERO_matrix)
    # matrix_corrected_list_of_list2 = matrix_correction(none_filtered_list_of_list, matrix_MOD_AL)
    print "Matrix Correction is done " + str(datetime.now() - startTime)
    """4) K baseline_subtraction"""
    line_scanner1 = line_scanner(matrix_corrected_list_of_list[3])
    l1 = line_scanner1.find_all_local_min()
    # l2 = line_scanner1.find_all_local_max()


    # line_scanner2 = line_scanner(matrix_corrected_list_of_list2[3])
    # l2 = line_scanner2.find_all_local_min()

    k_baseline1 = K_Baseline(l1)
    x_and_y_dict = k_baseline1.populate_x_and_y(matrix_corrected_list_of_list[3])

    bs1_subs = []
    for i in matrix_corrected_list_of_list_RAW:
        bs1 = baseline_subtraction_class2(i, x_and_y_dict)
        bs1_sub = bs1.perform_baseline_subtraction()
        bs1_subs.append((bs1_sub))

    # bs1 = baseline_subtraction_class(matrix_corrected_list_of_list_RAW[3])
    # a = bs1.perform_baseline_subtraction()

    print "k_baseline subtraction is done " + str(datetime.now() - startTime)
    """5) plot"""
    p1 = plot_dyes(matrix_corrected_list_of_list,
                   list_of_baseline_x=l1["x/quarter seconds"],
                   list_of_baseline_y=l1["y/best-fit line"],
                   scatter=True)
    p1.set_title(str(filtered_file_dir))
    p2 = plot_dyes(bs1_subs, scatter=False)
    p2.set_title(str(filtered_file_dir+" BS1_sub"))

    p3 = plot_dyes(matrix_corrected_list_of_list_RAW)
    p3.set_title("Raw")

    print "done " + str(datetime.now() - startTime)
    return


def main_load_one(file_dir):
    """
    Just for uploading and plotting one file_dir
    :param file_dir:
    :return:
    """
    print "Starting at " + str(startTime)
    """1) load file"""
    processed_data = pd.read_csv(file_dir, index_col=0)
    print "File done loading " + str(datetime.now() - startTime)
    kfiltered_list_of_list = get_five_dyes(processed_data)
    print "k_filter4 is done " + str(datetime.now() - startTime)
    """3) Matrix Correction"""
    matrix_corrected_list_of_list = matrix_correction(kfiltered_list_of_list, matrix_MOD_AL)
    """4) K baseline_subtraction"""
    line_scanner1 = line_scanner(matrix_corrected_list_of_list[3])
    l1 = line_scanner1.find_all_local_min()
    p1 = plot_dyes(matrix_corrected_list_of_list,list_of_baseline_x=l1["x/quarter seconds"], list_of_baseline_y=l1["y/best-fit line"],
                   scatter=True)
    p1.set_title(file_dir)


def find_new_baseline_minimums(dye, x_and_y_dict):
    """

    :param dye:
    :param x_and_y_dict:
    :return:
    """
    new_x_and_y_minimums = {
        "x/quarter seconds": x_and_y_dict["x/quarter seconds"],
        "y/best-fit line": []
    }
    for i in x_and_y_dict["x/quarter seconds"]:
        new_x_and_y_minimums["y/best-fit line"].append(dye[i])
    return new_x_and_y_minimums

def main1(filtered_file_dir, raw_file_dir):
    """
        Main to find x from time averaged data and then find local min and basline subtract from non time averaged data.
    """
    print "Starting at " + str(startTime)
    """1) load file"""
    processed_data = pd.read_csv(filtered_file_dir, index_col=0)
    raw_data = load_file(raw_file_dir)
    print "File done loading " + str(datetime.now() - startTime)
    """2) Time/Moving Average - Get 5 Dyes"""
    time_ave_data_list_of_list = get_five_dyes(processed_data)
    non_time_ave_data_list_of_list = k_filter3(raw_data, steps=8)
    print "k_filter4 is done " + str(datetime.now() - startTime)
    """3) Matrix Correction"""
    matrix_corrected_time_ave = matrix_correction(time_ave_data_list_of_list, matrix_MOD_AL)
    matrix_corrected_non_time_ave = matrix_correction(non_time_ave_data_list_of_list, matrix_MOD_AL)
    """ set_threshold..."""
    time_ave_list_of_list = set_threshold(matrix_corrected_time_ave, 250)
    non_time_ave_list_of_list = set_threshold(matrix_corrected_non_time_ave, 250)
    # p0 = plot_dyes(non_time_ave_list_of_list)
    # p0.set_title("non_time_ave_data_list_of_list")
    print "Matrix Correction is done " + str(datetime.now() - startTime)
    """4) K baseline_subtraction"""
    list_of_x_and_y = []
    for i in range(5):
        line_scanner1 = line_scanner(time_ave_list_of_list[i])
        local_min_dict1 = line_scanner1.find_all_local_min()
        local_min_dict2 = find_new_baseline_minimums(non_time_ave_list_of_list[i], local_min_dict1)

        # p00 = plot_dyes(time_ave_list_of_list,
        #                 list_of_baseline_x=local_min_dict1["x/quarter seconds"],
        #                 list_of_baseline_y=local_min_dict1["y/best-fit line"],
        #                 scatter=True)
        # p00.set_title("time_ave_data_list_of_list")

        k_baseline1 = K_Baseline(local_min_dict2)
        x_and_y_dict = k_baseline1.populate_x_and_y(non_time_ave_list_of_list[i])
        list_of_x_and_y.append(x_and_y_dict)
        #
    bs1_subs = []
    count = 0
    for i in non_time_ave_list_of_list:
        bs1 = baseline_subtraction_class2(i, list_of_x_and_y[count])
        bs1_sub = bs1.perform_baseline_subtraction()
        bs1_subs.append((bs1_sub))
        count += 1

    # bs1 = baseline_subtraction_class(matrix_corrected_list_of_list_RAW[3])
    # a = bs1.perform_baseline_subtraction()

    print "k_baseline subtraction is done " + str(datetime.now() - startTime)
    """5) plot"""
    # p1 = plot_dyes(non_time_ave_list_of_list,
    #                list_of_baseline_x=local_min_dict2["x/quarter seconds"],
    #                list_of_baseline_y=local_min_dict2["y/best-fit line"],
    #                scatter=True)
    # p1.set_title("Local Minimums Found")
    # p2 = plot_dyes(non_time_ave_list_of_list,
    #                list_of_baseline_x=x_and_y_dict["x/quarter seconds"],
    #                list_of_baseline_y=x_and_y_dict["y/best-fit line"],
    #                scatter=False)
    # p2.set_title("Baseline Drawn")
    p3 = plot_dyes(bs1_subs)
    p3.set_title("Baseline Subtracted")


    print "done " + str(datetime.now() - startTime)
    return

def main_get_five_dyes_and_convert_to_csv(file_dir):
    """

    :param file_dir: reference spectrum
    :return:
    """
    data = load_file(file_dir)
    # Reference spectrum
    list_of_list = get_five_dyes2(data)
    rows = zip(list_of_list[0], list_of_list[1], list_of_list[2], list_of_list[3], list_of_list[4])
    with open("LUNA_CUSTOMER_ALLELIC_LADDER.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(["Flu", "Joe", "TMR", "CXR", "WEN"])
        for row in rows:
            writer.writerow(row)

def main_get_five_dyes_and_plot(file_dir):
    """

    :param file_dir: reference spectrum
    :return:
    """
    # data = pd.read_csv(file_dir, index_col=0)
    data1 = load_file(file_dir)
    # Reference spectrum
    list_of_list = get_five_dyes2(data1)
    """ MATRIX CORRECTION (DELETE THIS WHEN NOT USING)"""
    # matrix_corrected = matrix_correction(list_of_list, matrix_MOD2)
    # p1 = plot_dyes(matrix_corrected)
    p1 = plot_dyes(list_of_list)
    p1.set_title(file_dir)

def main_with_just_kfilter4(file_dir, threhold_value=600):
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
    # matrix_corrected_list_of_list = matrix_correction(kfiltered_list_of_list, matrix_MOD2)
    """3) K_Baseline Subtraction"""
    # l1 = line_scanner1.find_all_local_min(threshold_value=50000)
    """3.1) Trying out peakdet"""
    list_of_peaks = k_peak_detection(kfiltered_list_of_list)
    print "Matrix Correction is done"
    """4) baseline_subtraction"""
    # bs1_subs = []
    # for i in matrix_corrected_list_of_list:
    #     bs1 = baseline_subtraction_class(i)
    #     bs1_sub = bs1.perform_baseline_subtraction()
    #     bs1_subs.append((bs1_sub))
    print "k_baseline subtraction is done"
    """5) plot"""
    print "plotting"
    # p1 = plot_dyes(kfiltered_list_of_list)
    # plotly_dyes(kfiltered_list_of_list)
    p2 = plot_dye_cursor(kfiltered_list_of_list, list_of_peaks)
    print "done"
    # plt.show()

def main_with_cursor():
    pass


def main_get_five_dark_spec_dyes_and_plot(file_dir):
    """

    :param file_dir: dark spectrum
    :return:
    """
    # data = pd.read_csv(file_dir, index_col=0)
    data1 = load_file(file_dir)
    # Reference spectrum
    # list_of_list = get_five_dyes2(data1)
    # Dark spectrum
    list_of_list = reference_sepectrum(data1)
    """ MATRIX CORRECTION (DELETE THIS WHEN NOT USING)"""
    # matrix_corrected = matrix_correction(list_of_list, matrix_MOD2)
    # p1 = plot_dyes(matrix_corrected)
    p1 = plot_dyes(list_of_list)
    p1.set_title(file_dir)


def main_conversion_310(file_dir, file_name):
    """

    """
    """
    Convert csv files into 310 format
    :param file_dir:
    :return:
    """
    data = load_file(file_dir)
    five_dyes = get_five_dyes(data)
    df = pd.DataFrame(five_dyes)
    df_transpose = df.transpose()
    df_transpose.to_csv("../csv_files/310_converted_"+file_name)
    print "a"

def main_conversion_sg1(file_dir):
    """
    Convert csv files into sg1 format
    :param file_dir:
    :return:
    """
    data = load_file(file_dir)
    five_dyes = get_five_dyes(data)
    SG1_Writer("../csv_files/CUSTOMER_LADDER.sg1", five_dyes)
    # df = pd.DataFrame(five_dyes)
    # df_transpose = df.transpose()
    # df_transpose.to_csv("../csv_files/310_converted_"+file_name)
    # print "a"

def reference_sepectrum(data):
    """
    inputs a file directory and subtracts the dark spectrum from each of the

    :param file_dir: (dark spectrum data set)
    :return: a new list of list that contains the reference spectrum value for each dye
    """
    dark_spectrum = data.iloc[:,1].tolist()
    list_of_dark_to_sub = []
    list_of_dark_to_sub.append(dark_spectrum[904])
    list_of_dark_to_sub.append(dark_spectrum[955])
    list_of_dark_to_sub.append(dark_spectrum[999])
    list_of_dark_to_sub.append(dark_spectrum[1036])
    list_of_dark_to_sub.append(dark_spectrum[1166])

    list_of_list_5_dyes_dark = get_five_dyes(data)
    new_l = [[] for x in xrange(5)] # initlaized a new list
    pos = 0
    for list_of_dye_dark, b in itertools.izip(list_of_list_5_dyes_dark, list_of_dark_to_sub):
        for a in list_of_dye_dark:
            new_l[pos].append(a-b)
        pos+=1
    # for pos, list_of_dye_dark in enumerate(list_of_list_5_dyes_dark):
    #     for a, b in itertools.izip(list_of_dye_dark, dark_spectrum):
    #         new_l[pos].append(a-b)
    return new_l

    # # dark_spectrum_list = dark_spectrum.tolist()
    # reference_sepectrum = pd.DataFrame()
    # # data.sub()
    # for index,row in data.iterrows():
    #     row_list = row.tolist()
    #     print "a"


if __name__ == "__main__":
    '''
    *** File Directory
    '''
    # Folder where csv files are
    folder = '../csv_files'
    # file name variables
    AL2 = '10_26_9mW_AL_(actual).csv'
    file_name = 'luna_experiment_1_18_AL_manual_run_trial_2.csv'
    # file_name2 = '10_13_AL_new_ibsen_modified.csv'
    file_name2 = 'luna_experiment_1_20_CUSTOMER_LADDER_manual_run_trial_2.csv'
    file_name3 = 'luna_experiment_1_23_CUSTOMER_LADDER_manual_run_DARK_ONLY.csv'

    five_color_matrix1 = "luna_experiment_1_12_AL_manual_run_trial_3.csv"

    Three_1 = 'Allelic_ladder_310.csv'
    Kevins_conv = '10_13__AL_RAW.csv'

    print "a"

    file_dir = join(folder, file_name)
    file_dir2 = join(folder, file_name2)
    file_dir3 = join(folder, file_name3)

    file_dir_luna = join(folder,file_dir)

    #ibsen spectrum
    # df = load_file(file_dir3)
    # dark_spectrum = df.iloc[:,1].tolist()
    # ibsen_plot = plot_ibsen_spectrum(dark_spectrum)

    # main_get_five_dyes_and_plot(join(folder, file_dir))
    # main_with_just_kfilter4(join(folder, file_dir2))
    main_with_just_kfilter4(join(folder, file_dir))

    # baseline_subtraction_steps_main(join(folder, file_dir2))
    # main_get_five_dyes_and_convert_to_csv(join(folder, file_dir2))
    # reference_sepectrum(file_dir3)
    # main_get_five_dark_spec_dyes_and_plot(join(folder,file_dir3))

    plt.show()
    # main(file_dir2)
    # baseline_subtraction_steps_main(file_dir)
    # """
    # Variables
    # """
    # # Directory variables
    # # directory_sg = '/Users/kevkim/GitHub/CSV-to-FSA-script/CSV FOLDER/KevinTxtAllelicLadder.sg1'
    # # directory_sg_out = '/Users/kevkim/GitHub/CSV-to-FSA-script/CSV FOLDER/output.sg1'
    # # destination = '/Users/kevkim/GitHub/CSV-to-FSA-script/CSV FOLDER'
    # # name = 'Allelic ladder - 10-14-16-5-21 PM.fsa'
    #
    # sg1_1 = '/Users/kevkim/GitHub/CSV-to-FSA-script/CSV FOLDER/KevinTxtAllelicLadder.sg1'
    # sg1_2 = '/Users/kevkim/GitHub/CSV-to-FSA-script/CSV FOLDER/310_converted_10_14_matrix.txt.sg1'
    # sg1_3 = '/Users/kevkim/GitHub/CSV-to-FSA-script/CSV FOLDER/310_converted_11_9_Matrix_10mW.txt.sg1'
    #
    # # Dataframe/list_to_list variable
    # df = pd.read_csv('/Users/kevkim/GitHub/CSV-to-FSA-script/CSV FOLDER/data_to_csv.csv', index_col=0)
    # # Contains a list of list of the dye values.
    # list_of_list = df.values.tolist()
    #
    # #######################
    # """
    # Write
    # """
    #
    # SG1_Writer(directory_sg_out, list_of_list)
    # SG1_reader = SG1_Reader(directory_sg_out)
    # a = SG1_reader.storeEntries()
    # print "a"

    """
    Read
    """
    # SG1_1_reader = SG1_Reader(sg1_1)
    # SG1_2_reader = SG1_Reader(sg1_2)
    # SG1_3_reader = SG1_Reader(sg1_3)
    # a = SG1_1_reader.storeEntries()
    # b = SG1_2_reader.storeEntries()
    # c = SG1_3_reader.storeEntries()
    # SG1_reader.showEntries()
    #
    # SG1_reader.storeEntries()


    # data = [SG1_reader.getData('TRAC', 1), SG1_reader.getData('TRAC', 2), SG1_reader.getData('TRAC', 3),
    #         SG1_reader.getData('TRAC', 4), SG1_reader.getData('TRAC', 105)]
    # one = SG1_reader.getData('TRAC', 1)
    # two = SG1_reader.getData('TRAC', 2)
    # Dye_num = SG1_reader.getData('Dye#', 1)
    # DATE1 = SG1_reader.getData('RUND', 1)
    # DATE2 = SG1_reader.getData('RUND', 2)
    # TIME1 = SG1_reader.getData('RUNT', 1)
    # TIME2 = SG1_reader.getData('RUNT', 2)
    # a = pd.DataFrame(data)
    # a.to_csv('/Users/kevkim/GitHub/CSV-to-FSA-script/CSV FOLDER/data_to_csv.csv')
    # for i in data:
    #     print i

    print "a"
    #
    # plot_dyes(data)
    plt.show()

    # with open(directory_sg_out, 'rb') as f:
    #     a = f.read(4)
    #     print a

    # myArr = bytearray(f.read(4))
