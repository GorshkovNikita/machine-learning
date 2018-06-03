from plot_binary import plot_binary_on_plt
import datasets.epl as epl
import numpy as np
import matplotlib.pyplot as plt


def dot_product(weights, inputs):
    if len(weights) != len(inputs):
        raise ArithmeticError('length must be equals')
    size = len(inputs)
    result = 0
    for i in range(size):
        result += weights[i] * inputs[i]
    return result


class Perceptron:
    bias_weight = 0
    weights = []
    epochs = 100
    learning_rate = 1

    # train_data - array of arrays if inputs
    # one sample of input is array of places
    # results - array of actual results for each input true/false
    def train(self, train_data, results):
        train_data_size = len(train_data)
        input_size = len(train_data[0])
        self.weights = [0] * input_size
        while True:
        # for _ in range(self.epochs):
            predicted_results = []
            for j in range(train_data_size):
                predicted_result = self.predict(train_data[j])
                predicted_results.append(predicted_result)
                sum = 0
                for k in range(input_size):
                    self.weights[k] += self.learning_rate * (results[j] - predicted_result) * train_data[j][k] # (1 if train_data[j][k] < 4 else 0)
                    sum += train_data[j][k]
                    self.bias_weight += self.learning_rate * (results[j] - predicted_result)  # * (sum / input_size)
                # print(self.weights)
                print(self.bias_weight)
            if np.array_equal(results, predicted_results):
                break

    def predict(self, input_list):
        product = dot_product(self.weights, input_list) + self.bias_weight
        print('product = ', product)
        if product <= 0.0:
            return -1
        else:
            return 1


if __name__ == '__main__':
    perceptron = Perceptron()
    perceptron.train(epl.train_data, epl.train_data_results)
    res = []
    for x in epl.test_data:
        res.append(perceptron.predict(x))
    print(np.array_equal(res, epl.test_data_results))
    print('weights = ', perceptron.weights)
    print('bias = ', perceptron.bias_weight)

    plt.plot(
        [0, - perceptron.bias_weight / perceptron.weights[1]],
        [perceptron.weights[1], - perceptron.weights[0] - perceptron.bias_weight / perceptron.weights[1]
    ])
    plot_binary_on_plt(plt, epl.test_data, epl.test_data_results)
    plt.show()
