import unittest
import Momentum
import Rosenbrock
import numpy as np
from MomentumLinearRegression import  dataLearning
from twoDData import twoDData
from Momentum import Momentum

class TestMomentum(unittest.TestCase):

    def testMomentuminit(self):
        m = Momentum(1,1)
        self.assertEqual(1, m.momentum)
        self.assertEqual(1, m.lr)

        m2 = Momentum()
        self.assertEqual(0.01, m2.lr)
        self.assertEqual(0.9, m2.momentum)

    def testMomentumLinearRegression(self):
        # learning rate
        rate = 0.0001
        # 관성
        momentum = 0.9
        # 시작점
        a = -20.0
        b = -20.0
        # 점의 개수
        num_points=50
        # 데이터 생성
        data = twoDData(num_points, 5, 5, 10, 5)
        x_data, y_data=data.dataGeneration()

        last_W, last_v = dataLearning(x_data, y_data, rate,momentum,1000,a,b)

        expected_W = (num_points * np.sum(x_data * y_data) - (np.sum(x_data) * np.sum(y_data))) / (num_points* np.sum(x_data**2) - (np.sum(x_data))**2)
        expected_v = (np.sum(y_data) - expected_W*np.sum(x_data))/num_points

        self.assertAlmostEqual(expected_W, last_W, delta = 0.001)
        self.assertAlmostEqual(expected_v, last_v, delta = 0.001)

    def testMomentumLinearRegression2(self):
        # learning rate
        rate = 0.0001
        # 관성
        momentum = 0.9
        # 시작점
        a = 0.0
        b = 0.0
        # 점의 개수
        num_points=100
        # 데이터 생성
        data = twoDData(num_points, 10, 5, 10, 5)
        x_data, y_data=data.dataGeneration()

        last_W, last_v = dataLearning(x_data, y_data, rate,momentum,1000,a,b)

        expected_W = (num_points * np.sum(x_data * y_data) - (np.sum(x_data) * np.sum(y_data))) / (num_points* np.sum(x_data**2) - (np.sum(x_data))**2)
        expected_v = (np.sum(y_data) - expected_W*np.sum(x_data))/num_points

        self.assertAlmostEqual(expected_W, last_W, delta = 0.001)
        self.assertAlmostEqual(expected_v, last_v, delta = 0.001)
if __name__ == '__main__':
    unittest.main()