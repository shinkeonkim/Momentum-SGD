import numpy as np

class Rosenbrock:
    # 2차원 로젠브록 함수
    def f2(self, x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2

    # 2차원 로젠브룩 함수의 도함수
    def f2g(self, x, y):
        return np.array((2.0 * (x - 1) - 400.0 * x * (y - x**2), 200.0 * (y - x**2)))