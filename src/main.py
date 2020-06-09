import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Rosenbrock import  Rosenbrock
from SGD import SGD
from Momentum import Momentum

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

momentum = 0.9
learning_rate = 0.00005

x, y = -2,-2 # starting_point

Momentum_model = Momentum(lr = learning_rate, momentum = momentum)
# Momentum_model = SGD(lr = learning_rate)
Rosenbrock_model = Rosenbrock()


xx = np.linspace(-5, 5, 100)
yy = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(xx, yy)
Z = Rosenbrock_model.f2(X, Y)


fig = plt.figure(1, figsize=(12, 8))
fig.suptitle('momentum SGD', fontsize=15)
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.set_top_view()
ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap='rainbow')

plt.subplot(2,2,2)
plt.plot(x, y, 'bo', markersize=10)
plt.plot(1, 1, 'ro', markersize=10)
levels = np.logspace(-1, 5, 10)
plt.contourf(X, Y, Z, alpha=0.2, levels=levels)
plt.contour(X, Y, Z, colors="green", levels=levels, zorder=0)
plt.xticks(np.linspace(-5, 5, 11))
plt.yticks(np.linspace(-5, 5, 11))
plt.xlabel("x")
plt.ylabel("y")


all_loss = []
all_step = []
all_x = [x]
all_y = [y]
last_x = x
last_y = y
loss = 0
vx, vy = [0, 0]

plt.ion()
xy = {'x': x, 'y': y}
for step in range(1,1001):
    g = Rosenbrock_model.f2g(xy['x'], xy['y'])
    Momentum_model.update(xy, {"x":g[0], "y": g[1]})
    
    ax.plot3D(all_x, all_y, Rosenbrock_model.f2(xy['x'], xy['y']), 'gray')
    
    plt.subplot(2,2,2)
    plt.scatter(xy['x'],xy['y'],s=5,color='blue')
    plt.plot([last_x,xy['x']],[last_y,xy['y']],color='aqua')
    
    loss = (xy['x'] - 1) ** 2 + (xy['y'] - 1) ** 2
    
    print(xy['x'],xy['y'], loss)

    all_loss.append(loss)
    all_step.append(step)

    plt.subplot(2,2,4)
    plt.plot(all_step,all_loss,color='orange')
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True)
    last_x = xy['x']
    last_y = xy['y']
    all_x.append(last_x)
    all_y.append(last_y)

    plt.show()
    plt.pause(0.0001)

plt.show()    
plt.pause(9999)