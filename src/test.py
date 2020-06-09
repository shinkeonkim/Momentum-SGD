import unittest
import Momentum
import Rosenbrock

class TestMomentum(unittest.TestCase):

    def test_momentuminit(self):
        m = Momentum.Momentum(1,1)
        self.assertEqual(1, m.momentum)
        self.assertEqual(1, m.lr)

        m2 = Momentum.Momentum()
        self.assertEqual(0.01, m2.lr)
        self.assertEqual(0.9, m2.momentum)

# class TestRosenbrock(unittest.TestCase):

#     def test_f2(self):
#         m = Rosenbrock.Rosenbrock()
#         self.assertEqy


if __name__ == '__main__':
    unittest.main()