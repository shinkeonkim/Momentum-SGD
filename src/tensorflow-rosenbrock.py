import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Rosenbrock import  Rosenbrock

# Define minimum variable x
x = tf.Variable(([[-1.], [-1.]]))
# Define the Rosenbrock function coefficeints as placeholder c
c = tf.placeholder(tf.float32, [5, 1])

# Define the Rosenbrock function (symbolically) we are trying to minimize
rosen = c[0][0]*(c[1][0] + c[2][0]*x[0][0])**2 + (c[3][0]*x[1][0] + c[4][0]*x[0][0]*x[0][0])**2

# Define the Tensorflow optimizer, we are going to use the Gradient Descent find the optimal (minimum) point
# of this function (using a learning rate of 0.01)
learn_rate = 0.01
optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(rosen)

# Define the coefficient c
coefficients = np.array([[1.0], [1.0], [-1.0], [1.0], [-1.0]])

# Initialize Tensorflow
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print("Initial values of x are: {}".format(session.run(x)))

# Now let us run the 1st iteration of this computation:
session.run(optimizer, feed_dict={c:coefficients})
print("1st iteration value of x are: \n{}\n".format(session.run(x)))


Rosenbrock_model = Rosenbrock()
xx = np.linspace(-5, 5, 100)
yy = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(xx, yy)
Z = Rosenbrock_model.f2(X, Y)
last_x = [[-1], [-1]]

fig = plt.figure(1, figsize=(12, 8))
fig.suptitle('momentum SGD', fontsize=15)
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.set_top_view()
ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap='rainbow')

plt.subplot(2,2,2)
plt.plot(last_x[0][0], last_x[1][0], 'bo', markersize=10)
plt.plot(1, 1, 'ro', markersize=10)
levels = np.logspace(-1, 5, 10)
plt.contourf(X, Y, Z, alpha=0.2, levels=levels)
plt.contour(X, Y, Z, colors="green", levels=levels, zorder=0)
plt.xticks(np.linspace(-5, 5, 11))
plt.yticks(np.linspace(-5, 5, 11))
plt.xlabel("x")
plt.ylabel("y")
plt.ion()

# Now let us see the results of this computation for many iterations
max_iter = 2000
for i in range(max_iter):
    session.run(optimizer, feed_dict={c:coefficients})
    ret = session.run(x)
    
    if i%200 == 0:
        print("{0}th iteration value of x are: \n{1}\n".format(i, ret))

    plt.subplot(2,2,2)
    plt.scatter(ret[0][0],ret[1][0],s=5,color='blue')
    plt.plot([last_x[0][0],ret[0][0]],[last_x[1][0], ret[1][0]],color='aqua')
    last_x = ret
    plt.show()
    plt.pause(0.0001)

plt.show()
plt.pause(10)