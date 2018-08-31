import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class SigmoidRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        # f = tf.constant(20.0) * tf.sigmoid(tf.matmul(self.x, self.W) + self.b) + tf.constant(31.0)
        self.f = 20 * tf.sigmoid(tf.matmul(self.x, self.W) + self.b) + 31
        # f = 20 / (1 + np.exp(-(tf.matmul(self.x, self.W) + self.b))) + 31

        # Uses Mean Squared Error
        self.loss = tf.losses.mean_squared_error(self.y, self.f)
        # self.loss = tf.reduce_sum(tf.square(self.f - self.y))


class SigmoidPlot:
    def __init__(self, W, b):
        # print('SigmoidPlot(W={}, b={})'.format(W, b))
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return 20 / (1 + np.exp(-(x @ self.W + self.b))) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))


def train(x_train, y_train):
    model = SigmoidRegressionModel()

    # Training: adjust the model so that its loss is minimized
    # minimize_operation = tf.train.GradientDescentOptimizer(0.000000001).minimize(model.loss)
    minimize_operation = tf.train.AdamOptimizer(0.01).minimize(model.loss)

    # Create session object for running TensorFlow operations
    session = tf.Session()

    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    for epoch in range(1000):
        session.run(minimize_operation, {model.x: x_train, model.y: y_train})
        if epoch % 100 == 0:
            W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
            print('Epoch {}: W={}, b={}, loss={}'.format(epoch, W, b, loss))

    # Evaluate training accuracy
    W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
    print("W = %s, b = %s, loss = %s" % (W, b, loss))

    f_test = session.run([model.f], {model.x: np.array([[0]])})
    print('f_test model.f([[0]]):', f_test)

    session.close()
    return model, W, b, loss


def plot(W, b, loss, x_data, y_data):
    fig, ax = plt.subplots()

    ax.plot(x_data, y_data, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    model_plot = SigmoidPlot(W, b)

    x = np.linspace(0, np.max(x_data), 100)
    x = np.reshape(x, (len(x), 1))
    print('f(0):', model_plot.f(np.array([[0]])))
    ax.plot(x, model_plot.f(x), label='$y = f(x) = 20\\sigma(xW+b)+31$')

    print('loss:', model_plot.loss(x_data, y_data))

    ax.legend()
    plt.show()


if __name__ == '__main__':
    data = np.genfromtxt('day_head_circumference.csv', delimiter=',')
    x_data, y_data = np.hsplit(data, 2)
    model, W, b, loss = train(x_data, y_data)
    plot(W, b, loss, x_data, y_data)
