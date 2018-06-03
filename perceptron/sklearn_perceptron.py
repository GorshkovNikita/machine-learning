from sklearn import linear_model
import numpy as np
import datasets.epl as epl
import matplotlib.pyplot as plt

import plot_binary

if __name__ == '__main__':
    p = linear_model.Perceptron(max_iter=10)
    p.fit(epl.train_data, np.asarray(epl.train_data_results))
    print(p.classes_)
    print(p.score(epl.train_data, epl.train_data_results))
    res = p.predict(epl.test_data)
    print(res)
    print(np.array_equal(res, epl.test_data_results))
    print(p.coef_)
    print(p.intercept_)
    # plt.plot([0, 8/3], [3, 11/3], 'r')
    # plot_binary.plot_binary(epl.train_data, epl.train_data_results)
    plt.show()
