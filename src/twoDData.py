import numpy as np
import matplotlib.pyplot as plt

class twoDData:
    def __init__(self, num_points, x_mid, x_range, y_weight, noise_range):
        self.num_points = num_points
        self.x_mid = x_mid
        self.x_range = x_range
        self.y_weight = y_weight
        self.noise_range = noise_range
        self.x_data = []
        self.y_data = []

    def dataGeneration(self):
        data_set = []
        for i in np.arange(self.num_points):
            x = np.random.normal(self.x_mid, self.x_range)
            y = x * self.y_weight + np.random.randint(-self.noise_range, self.noise_range)
            data_set.append([x, y])

        x_data = [i[0] for i in data_set]
        y_data = [i[1] for i in data_set]

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