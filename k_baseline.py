import numpy as np
import itertools
"""
Line Scanner Class
"""
class line_scanner:
    def __init__(self, object):
        # self.local_max = np.array([])
        # self.local_min = np.array([])
        # self.local_max = []
        self.local_min = [[] , []]
        # self.local_max_dict = {}
        self.local_min_dict = {}
        self.reading_line = object

    def read_line(self):
        """
        function that let's you read the reading_line.
        Outputs a two lists (self.local_min, self.local_max)
        :return: self.local_min, self.local_max
        """
        i_counter = 0
        uno = float
        dos = float
        tres = float
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
                if uno > dos and tres > dos: # If local min, append to local_min
                    self.local_min[0].append(i_counter)
                    self.local_min[1].append(dos)
        return self.local_min
    # def get_reading_line(self):
"""
k_baseline class
"""
class K_Baseline:
    def __init__(self, list_of_local_mins):
        self.x_and_y = {
            "x/quarter seconds": [],
            "y/best-fit line": []
        }
        self.list_of_mins = list_of_local_mins

    def populate_x_and_y(self):
        i_counter = 0
        uno = float
        dos = float
        tres = float
        for i0, i1 in itertools.izip(self.list_of_mins[0], self.list_of_mins[1]):
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

                i_counter+=1
                # Equation
                x_s = np.array([x0, x1, x2])
                ave = np.array([uno, dos, tres])

                a = np.polyfit(x_s, ave, 2)
                for x in range(x0, x1):
                    y = a[0]*(x**2)+a[1]*x+a[2]
                    self.x_and_y["x/quarter seconds"].append(x)
                    self.x_and_y["y/best-fit line"].append(y)
        return self.x_and_y








def find_coeff(ave):
    """
    :param ave: np array of ave error [95, 72, 55]
    :return:
    """
    actual_temp = np.array([55.0, 72.0, 95.0])
    x = actual_temp
    y = actual_temp-ave
    return np.polyfit(x, y, 2)
#
# # Initialize np.array for each reading and then calculate the mean.
# Ther1_1 = np.array([54.2, 70.2, 92.7])
# Ther1_2 = np.array([54.2, 70.4, 92.3])
# Ther1_mean = np.mean([Ther1_1, Ther1_2], axis=0)
# # print Ther1_mean
#
# Ther2_1 = np.array([54.3, 70.2, 93.4])
# Ther2_2 = np.array([54.3, 71.1, 93.4])
# Ther2_3 = np.array([54.3, 70.9, 93.4])
# Ther2_mean = np.mean([Ther2_1, Ther2_2, Ther2_3], axis=0)
# # print Ther2_mean
#
# Iuliu_1 = np.array([54.4, 71.5, 94.0])
# Iuliu_2 = np.array([54.6, 71.4, 93.9])
# Iuliu_mean = np.mean([Iuliu_1, Iuliu_2], axis=0)
# # print Iuliu_mean
#
# Ther3_1 = np.array([52.8, 69.9, 92.0])
# Ther3_2 = np.array([54.1, 70.1, 91.8])
# Ther3_3 = np.array([54.0, 70.0, 91.6])
# Ther3_mean = np.mean([Ther3_1, Ther3_2, Ther3_3], axis=0)
# # print Ther3_mean
#
# Thermister1 = find_coeff(Ther1_mean)
# Thermister2 = find_coeff(Ther2_mean)
# Iuliu = find_coeff(Iuliu_mean)
# Thermister3 = find_coeff(Ther3_mean)
# print Thermister1
# print Thermister1
# print Iuliu
# print Thermister3


