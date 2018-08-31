import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionModel:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return x * self.W + self.b

    # Uses Mean Squared Error, although instead of mean, sum is used.
    def loss(self, x, y):
        return np.sum(np.square(self.f(x) - y))


def main():
    fig, ax = plt.subplots()

    data = np.genfromtxt('length_weight.csv', delimiter=',')
    x_train, y_train = np.hsplit(data, 2)
    # x_train = np.mat([[1], [1.5], [2], [3], [4], [5], [6]])
    # y_train = np.mat([[5], [3.5], [3], [4], [3], [1.5], [2]])

    ax.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    model = LinearRegressionModel(W=np.mat([[0.17947166]]), b=np.mat([[-3.5857582]]))

    x = np.mat([[np.min(x_train)], [np.max(x_train)]])
    ax.plot(x, model.f(x), label='$y = f(x) = xW+b$')

    print('loss:', model.loss(x_train, y_train))

    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
