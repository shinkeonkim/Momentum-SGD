import numpy as np
import matplotlib.pyplot as plt
from Rosenbrock import basic_f2, basic_f2g
from mpl_toolkits.mplot3d import Axes3D
import random

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

mu = 8e-4  # step size
learning_rate = 0.3
s = 0.95  # for arrowhead drawing
x, y = 0,0
past_velocity = [0,0]
momentum = 0.8     # 모멘텀 상수


#다음 그림에  x=−1,y−1 에서 시작하여
# 최대경사법으로 최적점을 찾아나가는 과정을 그레디언트 벡터 화살표와 함께 보였다.

xx = np.linspace(-4, 4, 800)
yy = np.linspace(-3, 3, 600)
X, Y = np.meshgrid(xx, yy)
Z = basic_f2(X, Y)

fig = plt.figure(1, figsize=(12, 8))
fig.suptitle('learning rate: %.2f method:momentum SGD'%(learning_rate), fontsize=15)

# 绘制图1的曲面
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.set_top_view()
ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap='rainbow')

plt.subplot(2,2,2)
levels = np.logspace(-1, 3, 10)

plt.contourf(X, Y, Z, alpha=0.2, levels=levels)
plt.contour(X, Y, Z, colors="green", levels=levels, zorder=0)
plt.plot(1, 1, 'ro', markersize=10)
#plt.show() # debug 시에만 사용할 것

plt.ion()

for i in range(2000): # 5000 
    gradient = basic_f2g(x, y)
    
    velocity = [momentum * past_velocity[i] -  mu * learning_rate * gradient[i] for i in range(2)]
    next_x = x + velocity[0]
    next_y = y + velocity[1]

    plt.arrow(x, y, s * (velocity[0]), s * (velocity[1]),
              head_width=0.04, head_length=0.04, fc='k', ec='k', lw=2)
    x = next_x
    y = next_y
    past_velocity = velocity[:]

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xticks(np.linspace(-2, 2, 10))
plt.yticks(np.linspace(-2, 2, 10))
plt.xlabel("x")
plt.ylabel("y")
plt.title("모멘텀 최적화" )
plt.show()
print(x,y)
plt.pause(999999)