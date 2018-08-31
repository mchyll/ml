import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        f = tf.sigmoid(tf.matmul(self.x, self.W) + self.b)

        # Uses Mean Squared Error
        self.loss = tf.losses.mean_squared_error(self.y, f)


class LinearPlot:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return 1 / (1 + np.exp(-(x @ self.W + self.b)))

    # Uses Mean Squared Error
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))


def train(x_train, y_train):
    model = LinearRegressionModel()

    # Training: adjust the model so that its loss is minimized
    minimize_operation = tf.train.GradientDescentOptimizer(10).minimize(model.loss)

    # Create session object for running TensorFlow operations
    session = tf.Session()

    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    for epoch in range(10000):
        session.run(minimize_operation, {model.x: x_train, model.y: y_train})

    # Evaluate training accuracy
    W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
    print("W = %s, b = %s, loss = %s" % (W, b, loss))

    session.close()
    return model, W, b, loss


def plot(W, b, loss, x_data, y_data):
    fig, ax = plt.subplots()

    ax.plot(x_data, y_data, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    model_plot = LinearPlot(W, b)

    # x = np.mat([[np.min(x_data)], [np.max(x_data)]])
    x = np.linspace(0, np.max(x_data), 100)
    x = np.reshape(x, (len(x), 1))
    ax.plot(x, model_plot.f(x), label='$y = f(x) = xW+b$')

    print('loss:', model_plot.loss(x_data, y_data))

    ax.legend()
    plt.show()


if __name__ == '__main__':
    x_data = np.array([[0], [1]])
    y_data = np.array([[1], [0]])
    model, W, b, loss = train(x_data, y_data)
    plot(W, b, loss, x_data, y_data)
