import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Rosenbrock import  Rosenbrock

learning_rate = 0.000015
a = -5.0
b = -5.0
step = 5000

x = tf.Variable(([[a], [b]]))
c = tf.placeholder(tf.float32, [4, 1])
coefficients = np.array([[1.0], [-1.0], [100.0], [-1.0]])
rosenbrock = (c[0][0] + c[1][0]*x[0][0])**2 + c[2][0]*((x[1][0] + c[3][0]*x[0][0]*x[0][0])**2)

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(rosenbrock)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
session.run(x)

session.run(optimizer, feed_dict={c:coefficients})

Rosenbrock_model = Rosenbrock()
xx = np.linspace(-10, 10, 200)
yy = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(xx, yy)
Z = Rosenbrock_model.f2(X, Y)
last_x = [[a], [b]]

fig = plt.figure(1, figsize=(16, 10))
fig.suptitle('momentum SGD', fontsize=15)
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.set_top_view()
ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap='rainbow')

plt.subplot(2,2,2)
plt.plot(last_x[0][0], last_x[1][0], 'bo', markersize=10)
plt.plot(1, 1, 'ro', markersize=10)
levels = np.logspace(-1, 10, 10)
plt.contourf(X, Y, Z, alpha=0.2, levels=levels)
plt.contour(X, Y, Z, colors="green", levels=levels, zorder=0)
plt.xticks(np.linspace(-10, 10, 21))
plt.yticks(np.linspace(-10, 10, 21))
plt.xlabel("x")
plt.ylabel("y")
plt.ion()

all_loss = []

for i in range(step):
    session.run(optimizer, feed_dict={c:coefficients})
    ret = session.run(x)
    
    loss = (ret[0][0] - 1) ** 2 + (ret[1][0] - 1) ** 2
    all_loss.append(loss)
    ax.plot3D(ret[0][0], ret[1][0], Rosenbrock_model.f2(ret[0][0], ret[1][0]), 'bo')

    plt.subplot(2,2,2)
    plt.scatter(ret[0][0],ret[1][0],s=5,color='blue')
    plt.plot([last_x[0][0],ret[0][0]],[last_x[1][0], ret[1][0]],color='aqua')
    last_x = ret

    plt.subplot(2,2,4)
    plt.cla()
    
    plt.grid(True)
    plt.xscale('log')
    plt.plot(np.linspace(0,len(all_loss),len(all_loss)), all_loss, color = 'orange')
    plt.xlabel("step")
    plt.ylabel("loss")

    plt.show()
    # plt.pause(0.00001)

plt.show()
plt.pause(999)