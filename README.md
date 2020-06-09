# Momentum-SGD


## 1. 모멘텀의 개요 및 동작 원리

`모멘텀`은 물리에서 운동량, 관성, 탄성, 가속도 등을 의미하는 단어입니다. 따라서 모멘텀 SGD는 물리와 관련이 있음을 알 수 있으며, 경사하강법(SGD)에 관성을 더해 주는 알고리즘입니다.

경사하강법과 마찬가지로, 매번 기울기를 구한 뒤 바로 가중치에 반영하기 전에 이전에 수정한 방향을 참고하여, 일정 비율만 수정하게 됩니다. 이에 따라, 수정하는 방향이 왔다갔다하는 지그재그 현상이 줄어들고, 이전 이동값을 고려해 일정 비율을 반영한 다음값으로 이동하는 관성의 효과를 낼 수 있습니다.

`모멘텀 SGD`에서 갱신하려는 가중치 매개변수는 다음 수식과 같이 나타낼 수 있습니다. 

<img src = "./img/momentum.JPG">

여기서, W는 갱신하려는 가중치 매개변수이며, V는 물리에서 말하는 속도를 의미합니다.

속도가 추가되는 것으로 하여금, 해를 찾아가는 과정에서 기울기 방향으로 물체가 가속되는 물리법칙을 모방하고 있습니다.

예를 들어 만약 가중치 매개변수가 2개 있는 최적해를 찾아가는 과정이라면, 어떤 곡면에서 공이 굴러가는 듯한 모습으로 최적해를 찾아가게 됩니다.

## 2. 모멘텀의 동작 코드와 단위 테스트

### 2-1. TensorFlow를 활용한 모멘텀 동작 코드

#### TensorFlow

**2차원 정규 분포 무작위 데이터 클래스**
```python
import numpy as np
import matplotlib.pyplot as plt

class twoDData:
    def __init__(self, num_points, x_mid, x_range, y_mid, y_range):
        self.num_points = num_points
        self.x_mid = x_mid
        self.x_range = x_range
        self.y_mid = y_mid
        self.y_range = y_range
        self.x_data = []
        self.y_data = []

    def Data_Genearion(self):
        vectors_set = []
        for i in np.arange(self.num_points):
            x = np.random.normal(self.x_mid, self.x_range)
            y = np.random.normal(self.y_mid, self.y_range)
            vectors_set.append([x, y])

        x_data = [v[0] for v in vectors_set]
        y_data = [v[1] for v in vectors_set]

        self.x_data = x_data
        self.y_data = y_data
        
        return  x_data, y_data

    def Data_Draw(self):
        plt.plot(self.x_data, self.y_data,'ro')
        plt.ylim([min(self.y_data)-10,max(self.y_data) +10])
        plt.xlim([min(self.x_data)-10,max(self.x_data) +10])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
```

**TensorFlow 핵심 Code**
```python
optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
```


**TensorFlow 전체 코드**
```python
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
```

### 2-2. 모멘텀 동작 코드 단위 테스트

#### 2-2-1. 모멘텀 동작 코드 단위 테스트 코드

```python
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
```

#### 2-2-2. 모멘텀 동작 코드 단위 테스트 결과

```
----------------------------------------------------------------------
Ran 2 tests in 0.004s

OK
```


## 3. 모멘텀의 구체화

### 3-1. numpy 를 활용한 SGD 구체화
```python
class SGD:

    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```

### 3-1. numpy 를 활용한 모멘텀 SGD 구체화

```python
from SGD import SGD

class Momentum(SGD):
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
          
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] -  self.lr * grads[key]
            params[key] +=  self.v[key]
```

## 4. 구체화한 모듈의 단위테스트

### 4-1. 단위 테스트 코드  
```python
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

if __name__ == '__main__':
    unittest.main()
```

### 4-2. 단위 테스트 코드 결과
```
----------------------------------------------------------------------
Ran 1 test in 0.000s

OK
```

## 5. 모멘텀 최적화 알고리즘 검증
> 성능 검증은 로젠브룩 함수를 통해 진행하였습니다.
