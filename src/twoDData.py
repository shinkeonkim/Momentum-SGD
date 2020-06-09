import numpy as np
import matplotlib.pyplot as plt

class twoDData:
    def __init__(self, num_points, x_mid, x_range, y_mid, y_range):
        self.num_points = num_points
        self.x_mid = x_mid
        self.x_range = x_range
        self.y_mid = y_mid
        self.y_range = y_range
        self.x_data = []
        self.y_data = []

    def dataGeneration(self):
        vectors_set = []
        for i in np.arange(self.num_points):
            x = np.random.normal(self.x_mid, self.x_range)
            y = np.random.normal(self.y_mid, self.y_range)
            vectors_set.append([x, y])

        x_data = [v[0] for v in vectors_set]
        y_data = [v[1] for v in vectors_set]

        self.x_data = x_data
        self.y_data = y_data
        
        return  x_data, y_data

    def dataDraw(self):
        plt.plot(self.x_data, self.y_data,'ro')
        plt.ylim([min(self.y_data)-10,max(self.y_data) +10])
        plt.xlim([min(self.x_data)-10,max(self.x_data) +10])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()