from random import randint
import matplotlib.pyplot as plt
from math import pow
from numpy import subtract
from sklearn import linear_model
import numpy as np

# create random dataset for linear coefficients
def random_dataset(k, b):
    x = []
    y = []
    for i in range(100):
        local_x = randint(0, 100)
        local_y = k * (local_x + randint(0, 5)) + b
        x.append(local_x)
        y.append(local_y)
    print('==============')
    print(x)
    print(y)
    print('==============')
    return x, y


def plot_random_dataset():
    x, y = random_dataset(1, 2)
    plt.plot(x, y, 'ro')
    plt.show()


# cost function (mean squared error)
def mse(expected, actual):
    if len(expected) != len(actual):
        raise AssertionError("input arrays must have equal size")
    cost = 0
    size = len(expected)
    for i in range(size):
        cost += pow(actual[i] - expected[i], 2)
    return cost / size


def func(yi, new_yi, xi):
    return xi * (yi - new_yi)


def linear_regression(y, x, k_current=0, b_current=0, learning_rate=0.0001, epochs=10000):
    n = float(len(y))
    eps = 0.01
    cost = 10000000
    for i in range(epochs):
    # while cost >= eps:
        new_y = list(map(lambda xi: k_current * xi + b_current, x))
        # print(new_y)
        cost = mse(y, new_y)
        # print('cost = ', cost)
        k_grad = - (2 / n) * sum(list(map(func, y, new_y, x)))
        b_grad = - (2 / n) * sum(subtract(y, new_y))
        k_current -= learning_rate * k_grad
        b_current -= learning_rate * b_grad
        # print('k=', k_current)
        # print('b=', b_current)
    return k_current, b_current, cost


if __name__ == '__main__':
    # print(sum(list(map(func, [3,4], [2,3], [1,2]))))
    # k_current = 2
    # b_current = 3
    # print(list(map(lambda xi: k_current * xi + b_current, [1,2])))
    x, y = random_dataset(-10, 400)
    # x = [41, 56, 87, 46, 96]
    # y = [44, 59, 91, 48, 103]
    # k, b, cost = linear_regression(y, x)
    # y_predict = []
    # for i in range(len(x)):
    #     y_predict.append(b + k * x[i])
    #
    # plt.plot(x, y, 'o')
    # plt.plot(x, y_predict, 'k-')
    # plt.show()

    regr = linear_model.LinearRegression()

    regr.fit(
        np.ndarray((len(x),1), buffer=np.array(x), dtype=int),
         np.ndarray((len(y),1), buffer=np.array(y), dtype=int)
    )
    # arr = np.ndarray((4,1), buffer=np.array([1, 2, 3, 4]), dtype=int)
    # print(arr)
    print(regr.coef_)

    predict = regr.predict(np.asarray([[41], [56]]))
    print(predict)

    k, b, cost = linear_regression(y, x)
    print(k)
    y_predict = [b + k * 41, b + k * 56]
    print(y_predict)
