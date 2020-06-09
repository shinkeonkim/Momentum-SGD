import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from twoDData import twoDData

def dataLearning(x_data, y_data):
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b
    
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)

    train = optimizer.minimize(loss)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    train_set = [] 

    for step in np.arange(100):
        sess.run(train)
        train_set.append([sess.run(W), sess.run(b), sess.run(loss)])

    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    W_data = [t[0] for t in train_set]
    v_data = [t[1] for t in train_set]
    Loss_data= [t[2] for t in train_set]

    return W_data,v_data, Loss_data


if __name__ == '__main__':
    num_points=50
    data = twoDData(num_points, 5, 10, 5 , 1)
    x_data, y_data=data.dataGeneration()
    data.dataDraw()

    W_data, v_data, Loss_data = dataLearning(x_data, y_data)

    print('W_data = ', W_data)
    print('v_data = ', v_data)
    print('Loss_data = ', Loss_data)