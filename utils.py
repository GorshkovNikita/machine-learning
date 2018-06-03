from sklearn import preprocessing
from datasets import epl

if __name__ == '__main__':
    x = preprocessing.normalize(epl.test_data)
    print(x)
    y = preprocessing.scale(epl.test_data)
    print(y)