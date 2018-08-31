from pprint import pprint
import numpy as np


if __name__ == '__main__':
    data = np.genfromtxt('test.csv', delimiter=',')
    print('type(data): ', type(data))
    pprint(data)

    splitted = np.hsplit(data, [1])
    pprint(splitted)
