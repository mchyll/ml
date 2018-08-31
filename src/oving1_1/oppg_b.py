import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, art3d


class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32, name='x')
        self.y = tf.placeholder(tf.float32, name='y')

        # Model variables
        self.W = tf.Variable([[0.0], [0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        self.f = tf.matmul(self.x, self.W) + self.b
        # print(self.f.op)

        # Uses Mean Squared Error
        self.loss = tf.losses.mean_squared_error(self.y, self.f)


class LinearPlot:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # HUSK ALLTID: '@' er matrisemultiplikasjon, '*' er elementmultiplikasjon!

    # Uses Mean Squared Error
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))


def train(x_train, y_train):
    model = LinearRegressionModel()

    # Training: adjust the model so that its loss is minimized
    minimize_operation = tf.train.GradientDescentOptimizer(0.0001).minimize(model.loss)

    # Create session object for running TensorFlow operations
    session = tf.Session()

    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    for epoch in range(100000):
        session.run(minimize_operation, {model.x: x_train, model.y: y_train})

    # Evaluate training accuracy
    W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
    print("W = %s, b = %s, loss = %s" % (W, b, loss))

    f_test = session.run([model.f], {model.x: np.array([[1, 1]])})
    print('f_test model.f([1, 1]):', f_test)

    session.close()
    return model, W, b, loss


def plot(W, b, loss, x_data, y_data):
    model = LinearPlot(W, b)

    fig = plt.figure('Linear regression: 3D')

    plot1 = fig.add_subplot(111, projection='3d')

    plot1.plot(
        x_data[:, 0].squeeze(),
        x_data[:, 1].squeeze(),
        y_data[:, 0].squeeze(),
        'o',
        label='$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$',
        color='blue')

    plot1_f = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color='green',
                                   label='$y = f(x) = xW+b$')

    plot1_info = fig.text(0.01, 0.02, '')

    plot1_loss = []
    for i in range(0, x_data.shape[0]):
        line, = plot1.plot([0, 0], [0, 0], [0, 0], color='red')
        plot1_loss.append(line)
        if i == 0:
            line.set_label('$|f(\\hat x^{(i)})-\\hat y^{(i)}|$')

    plot1.set_xlabel('$x_1$')
    plot1.set_ylabel('$x_2$')
    plot1.set_zlabel('$y$')
    plot1.legend(loc='upper left')
    plot1.set_xticks([])
    plot1.set_yticks([])
    plot1.set_zticks([])
    plot1.w_xaxis.line.set_lw(0)
    plot1.w_yaxis.line.set_lw(0)
    plot1.w_zaxis.line.set_lw(0)
    plot1.quiver([0], [0], [0], [np.max(x_data[:, 0] + 1)], [0], [0], arrow_length_ratio=0.05, color='black')
    plot1.quiver([0], [0], [0], [0], [np.max(x_data[:, 1] + 1)], [0], arrow_length_ratio=0.05, color='black')
    plot1.quiver([0], [0], [0], [0], [0], [np.max(y_data[:, 0] + 1)], arrow_length_ratio=0.05, color='black')

    def update_figure(event=None):
        if event is not None:
            if event.key == 'W':
                model.W[0, 0] += 0.01
            elif event.key == 'w':
                model.W[0, 0] -= 0.01
            elif event.key == 'E':
                model.W[1, 0] += 0.01
            elif event.key == 'e':
                model.W[1, 0] -= 0.01

            elif event.key == 'B':
                model.b[0, 0] += 0.05
            elif event.key == 'b':
                model.b[0, 0] -= 0.05

            elif event.key == 'c':
                model.W = W.copy()
                model.b = b.copy()

        nonlocal plot1_f
        plot1_f.remove()
        x1_grid, x2_grid = np.meshgrid(np.linspace(1, np.max(x_data[:, 0]), 10),
                                       np.linspace(1, np.max(x_data[:, 1]), 10))
        y_grid = np.empty([10, 10])
        for i in range(0, x1_grid.shape[0]):
            for j in range(0, x1_grid.shape[1]):
                x = np.array([[x1_grid[i, j], x2_grid[i, j]]])
                y_grid[i, j] = model.f(x)
                # y_grid[i, j] = model.f([[x1_grid[i, j], x2_grid[i, j]]])

        plot1_f = plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color='green')

        for i in range(0, x_data.shape[0]):
            plot1_loss[i].set_data([x_data[i, 0], x_data[i, 0]], [x_data[i, 1], x_data[i, 1]])
            plot1_loss[i].set_3d_properties([y_data[i, 0], model.f(x_data[i, :])])

        plot1_info.set_text(
            '$W=\\left[\\stackrel{%.2f}{%.2f}\\right]$\n$b=[%.2f]$\n$loss = \\sum_i(f(\\hat x^{(i)}) - \\hat y^{(i)})^2 = %.2f$' %
            (model.W[0, 0], model.W[1, 0], model.b[0, 0], model.loss(x_data, y_data)))

        fig.canvas.draw()

    update_figure()
    fig.canvas.mpl_connect('key_press_event', update_figure)

    plt.show()


if __name__ == '__main__':
    data = np.genfromtxt('day_length_weight.csv', delimiter=',')
    y_data, x_data = np.hsplit(data, [1])
    model, W, b, loss = train(x_data, y_data)
    plot(W, b, loss, x_data, y_data)
