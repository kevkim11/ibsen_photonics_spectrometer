import numpy as np
import itertools
"""
Line Scanner Class
"""
class line_scanner:
    def __init__(self, object):
        # self.local_max = np.array([])
        # self.local_min_dict = np.array([])
        # self.local_max = []
        # self.local_max_dict = {}
        self.local_min_dict = {
            "x/quarter seconds": [],
            "y/best-fit line": []
        }
        self.reading_line = object

    def read_line(self):
        """
        function that let's you read the reading_line.
        :return: list of 2-list
                self.local_min_dict[0] = all the iterations (x)
                self.local_min_dict[1] = all the values of the iterations (y)
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
                if uno > dos and tres > dos and dos < 5500: # If local min, append to local_min_dict
                    self.local_min_dict["x/quarter seconds"].append(i_counter)
                    self.local_min_dict["y/best-fit line"].append(dos)
        return self.local_min_dict
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
        uno = None
        dos = None
        tres = None
        for i0, i1 in itertools.izip(self.list_of_mins["x/quarter seconds"], self.list_of_mins["y/best-fit line"]):
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
                    y = (a[0]*(x**2))+(a[1]*x)+a[2]
                    self.x_and_y["x/quarter seconds"].append(x)
                    self.x_and_y["y/best-fit line"].append(y)
        return self.x_and_y