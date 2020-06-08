import numpy as np

# 2차원 로젠브록 함수
def basic_f2(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# 2차원 로젠브룩 함수의 도함수
def basic_f2g(x, y):
    return np.array((2.0 * (x - 1) - 400.0 * x * (y - x**2), 200.0 * (y - x**2)))

