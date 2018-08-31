import numpy as np
import tensorflow as tf


class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        f = tf.matmul(self.x, self.W) + self.b

        # Uses Mean Squared Error, although instead of mean, sum is used.
        # self.loss = tf.reduce_sum(tf.square(f - self.y))
        # print('typeof loss: ', type(self.loss))
        self.loss = tf.losses.mean_squared_error(self.y, f)
        # print('typeof loss: ', type(self.loss))


def main():
    # Observed/training input and output
    data = np.genfromtxt('length_weight.csv', delimiter=',')
    x_train, y_train = np.hsplit(data, 2)
    # x_train = np.mat([[1], [1.5], [2], [3], [4], [5], [6]])
    # y_train = np.mat([[5], [3.5], [3], [4], [3], [1.5], [2]])

    model = LinearRegressionModel()

    # Training: adjust the model so that its loss is minimized
    minimize_operation = tf.train.GradientDescentOptimizer(0.0001).minimize(model.loss)

    # Create session object for running TensorFlow operations
    session = tf.Session()

    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    for epoch in range(50000):
        session.run(minimize_operation, {model.x: x_train, model.y: y_train})

    # Evaluate training accuracy
    W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
    print("W = %s, b = %s, loss = %s" % (W, b, loss))

    session.close()


if __name__ == '__main__':
    main()
