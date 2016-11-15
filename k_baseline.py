import numpy as np
import itertools
"""
Line Scanner Class
Author: Kevin Kim
Date: 11/1/16
Designed to create a baseline-subtraction algorithm
Have two classes.
1) Line_scanner
2) K_Baseline (Kevin's Baseline)
3) baseline_subtraction_class - Combine line_scanner and K_Baseline
"""


class line_scanner:
    def __init__(self, object):
        """
        Scans the line and returns a dictionary containing lists of local min
        :param object: list - a list of ADC-Count at a specific pixel/wavelength.
        For example: line_scanner1 = line_scanner(x1[4])
        """
        self.local_min_dict = {
            "x/quarter seconds": [],
            "y/best-fit line": []
        }
        self.local_max_dict = {
            "x/quarter seconds": [],
            "y/best-fit line": []
        }
        self.reading_line = object

    def find_all_local_min(self, threshold_value=100):
        """
        function that let's you read the reading_line.
        :param threshold_value=5500 - value that sets the min number in order to be labeled as a min.
        :return: list of 2-list
                self.local_min_dict[0] = all the iterations (x)
                self.local_min_dict[1] = all the values of the iterations (y)
        """
        i_counter = 0
        pointer = 1
        uno = float
        dos = float
        tres = float
        # if threshold_baseline:
        #     self.reading_line[:] = [x-250 for x in self.reading_line]

        # [less_than_250.append(i)  for i in self.reading_line]
        for i in self.reading_line:
            if i_counter == 0:
                uno = i
                i_counter+=1
            elif i_counter == 1:
                dos = i
                i_counter+=1
            elif i_counter == 2:
                tres = i
                i_counter+=1
            else:
                uno = dos
                dos = tres
                tres = i
                i_counter += 1
                pointer+=1
                if uno > dos and tres > dos and dos > threshold_value: # If local min, append to local_min_dict
                    self.local_min_dict["x/quarter seconds"].append(pointer)
                    self.local_min_dict["y/best-fit line"].append(dos)
        return self.local_min_dict

    def find_all_local_max(self):
        """
        function that let's you read the reading_line.
        :param threshold_value=5500 - value that sets the min number in order to be labeled as a min.
        :return: list of 2-list
                self.local_min_dict[0] = all the iterations (x)
                self.local_min_dict[1] = all the values of the iterations (y)
        """
        i_counter = 0
        pointer = 1
        uno = float
        dos = float
        tres = float
        for i in self.reading_line:
            if i_counter == 0:
                uno = i
                i_counter += 1
            elif i_counter == 1:
                dos = i
                i_counter += 1
            elif i_counter == 2:
                tres = i
                i_counter += 1
            else:
                uno = dos
                dos = tres
                tres = i
                i_counter += 1
                pointer += 1
                if uno < dos and tres < dos:  # If local min, append to local_min_dict
                    self.local_max_dict["x/quarter seconds"].append(pointer)
                    self.local_max_dict["y/best-fit line"].append(dos)
        return self.local_max_dict

"""
k_baseline class
"""
class K_Baseline:
    def __init__(self, local_min_dict):
        """

        :param local_min_dict: dict - A dictionary that contains two lists with the corresponding times and min value
        """
        # Dictionary that contain the time and value of the k_baseline found.
        self.x_and_y = {
            "x/quarter seconds": [],
            "y/best-fit line": []
        }
        # Dictionary that holds all the local minimum x and y.
        self.dict_of_mins = local_min_dict

    def populate_x_and_y(self, dye):
        i_counter = 0
        uno = None
        dos = None
        tres = None
        for i0, i1 in itertools.izip(self.dict_of_mins["x/quarter seconds"], self.dict_of_mins["y/best-fit line"]):
            if i_counter == 0:
                x0 = i0
                uno = i1
                i_counter+=1
            elif i_counter == 1:
                x1 = i0
                dos = i1
                i_counter+=1
            elif i_counter == 2:
                x2 = i0
                tres = i1
                i_counter+=1
            # Top if/else are just to set up the algorithm.
            # The else is where the actual filter comes in.
            else:
                uno = dos
                dos = tres
                tres = i1

                x0 = x1
                x1 = x2
                x2 = i0

                # Equation
                x_s = np.array([x0, x1, x2])
                ave = np.array([uno, dos, tres])

                a = np.polyfit(x_s, ave, 2)

                if i_counter == 3:
                    for x in range(0, x0):
                        self.x_and_y["x/quarter seconds"].append(x)
                        self.x_and_y["y/best-fit line"].append(0)

                i_counter+=1

                if self.dict_of_mins["x/quarter seconds"][-1] == i0:
                    for x in range(x0, x2+1):
                        y = (a[0] * (x ** 2)) + (a[1] * x) + a[2]
                        self.x_and_y["x/quarter seconds"].append(x)
                        self.x_and_y["y/best-fit line"].append(y)
                    for x in range(x2+1, len(dye)):
                        self.x_and_y["x/quarter seconds"].append(x)
                        self.x_and_y["y/best-fit line"].append(0)
                else:
                    for x in range(x0, x1):
                        y = (a[0]*(x**2))+(a[1]*x)+a[2]
                        self.x_and_y["x/quarter seconds"].append(x)
                        self.x_and_y["y/best-fit line"].append(y)

                # if i_counter < len()
        return self.x_and_y


class baseline_subtraction_class:
    def __init__(self, dye):
        """
        class that does the k_baseline subtraction
        :param dye: list of ADC-count for a specific dye.
        """
        self.dye = dye
        self.line_scanner = line_scanner(self.dye)
        self.l1 = self.line_scanner.find_all_local_min()
        self.k_baseline = K_Baseline(self.l1)

    def perform_baseline_subtraction(self):
        """

        :return: list - list of new dye values with the k_baseline subtraction applied.
        """
        """
        1) Scan line
        2) Get Filtered Baseline
        3) Subtract from dye.
        """
        x_and_y = self.k_baseline.populate_x_and_y(self.dye)
        baseline_subtracted_dye = []
        for i0, i1 in itertools.izip(self.dye, x_and_y["y/best-fit line"]):
            baseline_sub_value = i0-i1
            baseline_subtracted_dye.append(baseline_sub_value)
        return baseline_subtracted_dye

class baseline_subtraction_class2:
    """
    Class that can take in different dye and x_and_y_baseline dict.
    Dye = Matrix Corrected and Moving Average data
    x_and_y_baseline_dict = dictionary containing the lines.
    """
    def __init__(self, dye, x_and_y_baseline_dict):
        self.dye = dye
        self.x_and_y_baseline_dict = x_and_y_baseline_dict

    def perform_baseline_subtraction(self):
        baseline_subtracted_dye = []
        for i0, i1 in itertools.izip(self.dye, self.x_and_y_baseline_dict["y/best-fit line"]):
            baseline_sub_value = i0 - i1
            baseline_subtracted_dye.append(baseline_sub_value)
        return baseline_subtracted_dye

