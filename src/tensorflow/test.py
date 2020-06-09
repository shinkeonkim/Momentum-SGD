import unittest

from twoDData import twoDData

class TestTwoDData(unittest.TestCase):

    def test_init(self):
        data = twoDData(1000,0,10,0,10)
        self.assertEqual(1000,data.num_points)
        self.assertEqual(0,data.x_mid)
        self.assertEqual(10,data.x_range)
        self.assertEqual(0,data.y_mid)
        self.assertEqual(10,data.y_range)


    def test_datageneration(self):
        data = twoDData(1000,0,10,0,10)
        data.dataGeneration()
        self.assertEqual(len(data.y_data), 1000)
        self.assertEqual(len(data.x_data), 1000)

if __name__ == '__main__':
    unittest.main()