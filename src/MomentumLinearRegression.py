import numpy as np
import matplotlib.pyplot as plt
from twoDData import twoDData
from mpl_toolkits.mplot3d import Axes3D
from Momentum import Momentum

def da(y,y_p,x):
    return np.sum((y-y_p)*(-x))

def db(y,y_p):
    return np.sum((y-y_p)*(-1))

# MSE를 반환하는 함수
def linear_loss(a, b, x, y):
    SSE = np.sum((y - (a * x + b))**2) / (2 * len(x))
    return SSE / len(x)

# 기울기, y절편, MSE를 3차원 평면으로 표시하는 함수
def show_surface(x,y):
    a = np.linspace(-50,50,500)
    b = np.linspace(-50,50,500)
    x = np.array(x)
    y = np.array(y)

    allMSE = np.zeros(shape=(len(a),len(b)))
    for i in range(0,len(a)):
        for j in range(0,len(b)):
            a0 = a[i]
            b0 = b[j]
            MSE = linear_loss(a=a0,b=b0,x=x,y=y)
            allMSE[i][j] = MSE

    a,b = np.meshgrid(a, b)

    return [a,b,allMSE]

def dataLearning(x_data, y_data, learning_rate, momentum, step, a, b):
    num_points = len(x_data)
    
    # 그래프 출력에 사용할 x 최소값, 최대값
    min_x = min(x_data)
    max_x = max(x_data)

    # momentum optimizer
    momentumOptimizer = Momentum(lr= learning_rate, momentum=momentum)
    optimizerParams = {"a": a, "b":b}
    optimizerGrads = {"a":0 ,"b": 0}

    all_loss = []
    all_step = []
    last_a = a
    last_b = b

    for step in range(1,step+1):
        loss = 0
        all_da = 0
        all_db = 0
        
        y_star = optimizerParams["a"] * x_data + optimizerParams["b"]
        loss = linear_loss(optimizerParams["a"],optimizerParams["b"],x_data,y_data)

        optimizerGrads["a"] = da(y_data, y_star, x_data)
        optimizerGrads["b"] = db(y_data, y_star)
            
        all_loss.append(loss)
        all_step.append(step)

        last_a = optimizerParams["a"]
        last_b = optimizerParams["b"]

        momentumOptimizer.update(params=optimizerParams, grads=optimizerGrads)

    return [last_a, last_b]


if __name__ == "__main__":
    
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
    x, y=data.dataGeneration()

    # 그래프 출력에 사용할 x 최소값, 최대값
    min_x = min(x)
    max_x = max(x)

    # momentum optimizer
    momentumOptimizer = Momentum(lr= rate, momentum=momentum)
    optimizerParams = {"a": a, "b":b}
    optimizerGrads = {"a":0 ,"b": 0}


    # ----- 1,1 subplot 부분 ------

    # 평면 출력
    [aa,bb,MSE] = show_surface(x,y)
    # 전치 행렬로 바꾸기
    SSE = MSE.T

    fig = plt.figure(1, figsize=(12, 8))
    fig.suptitle('momentum SGD learning rate: %.4f' %(rate), fontsize=20)

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_surface(aa, bb, MSE, rstride=2, cstride=2, cmap='rainbow')

    # ----- 1,1 subplot 부분 end -----

    plt.subplot(2,2,2)
    ta = np.linspace(-50, 50, 500)
    tb = np.linspace(-50, 50, 500)
    plt.contourf(aa,bb,SSE,15,alpha=0.5)
    C = plt.contour(aa,bb,SSE,15,colors="green")
    plt.clabel(C,inline=True)
    plt.xlabel('a')
    plt.ylabel('b')


    plt.ion()

    all_loss = []
    all_step = []
    last_a = a
    last_b = b

    for step in range(1,200):
        loss = 0
        all_da = 0
        all_db = 0
        
        y_star = optimizerParams["a"] * x + optimizerParams["b"]
        loss = linear_loss(optimizerParams["a"],optimizerParams["b"],x,y)

        optimizerGrads["a"] = da(y, y_star, x)
        optimizerGrads["b"] = db(y, y_star)

        ax.scatter(optimizerParams["a"], optimizerParams["b"], loss, color='black')
        plt.subplot(2,2,2)
        plt.scatter(optimizerParams["a"], optimizerParams["b"],s=5,color='blue')
        plt.plot([last_a,optimizerParams["a"]],[last_b,optimizerParams["b"]],color='aqua')

        plt.subplot(2, 2, 3)
        plt.cla()
        plt.plot(x, y, 'ro')
        virtual_x = np.linspace(min_x,max_x,2)
        virtual_y = optimizerParams["a"] * virtual_x + optimizerParams["b"]
        plt.plot(virtual_x, virtual_y)
            
        all_loss.append(loss)
        all_step.append(step)
        plt.subplot(2,2,4)
        plt.plot(all_step,all_loss,color='orange')
        plt.xlabel("step")
        plt.ylabel("loss")

        last_a = optimizerParams["a"]
        last_b = optimizerParams["b"]

        momentumOptimizer.update(params=optimizerParams, grads=optimizerGrads)

        print("step: ", step, " loss: ", loss)
        plt.show()
        plt.pause(0.01)
        
    plt.show()
    plt.pause(99)