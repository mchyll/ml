import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, art3d


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class XORSigmoidModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32, name='x')
        self.y = tf.placeholder(tf.float32, name='y')

        # Model variables
        self.W1 = tf.Variable([[1.0, -1.0],
                               [-1.0, 1.0]])
        self.b1 = tf.Variable([[0.0, 0.0]])

        self.W2 = tf.Variable([[-1.0],
                               [1.0]])
        self.b2 = tf.Variable([[0.0]])

        # Hidden layer
        self.logits_1 = tf.matmul(self.x, self.W1) + self.b1
        self.f1 = tf.sigmoid(self.logits_1)

        # Output layer
        self.logits_2 = tf.matmul(self.f1, self.W2) + self.b2
        self.f2 = tf.sigmoid(self.logits_2)

        # Uses Sigmoid cross entropy
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, self.logits_2)


class XORSigmoigPlot:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    # Predictor, first layer
    def f1(self, x):
        return sigmoid(x @ self.W1 + self.b1)  # HUSK ALLTID: '@' er matrisemultiplikasjon, ikke '*'!

    # Predictor, second layer
    def f2(self, h):
        return sigmoid(h @ self.W2 + self.b2)

    def f(self, x):
        return self.f2(self.f1(x))

    def loss(self, x, y):
        return -np.mean(np.multiply(y, np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))


def train(x_train, y_train):
    model = XORSigmoidModel()

    # Training: adjust the model so that its loss is minimized
    minimize_operation = tf.train.GradientDescentOptimizer(1).minimize(model.loss)

    # Create session object for running TensorFlow operations
    session = tf.Session()

    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    for epoch in range(10000):
        session.run(minimize_operation, {model.x: x_train, model.y: y_train})

    # Evaluate training accuracy
    W1, b1, W2, b2, loss = session.run([model.W1, model.b1, model.W2, model.b2, model.loss],
                                       {model.x: x_train, model.y: y_train})
    print("W1 = %s\nb1 = %s\nW2 = %s\nb2 = %s\nloss = %s" % (W1, b1, W2, b2, loss))

    f2_test, logits2_test = session.run([model.f2, model.logits_2], {model.x: np.array([[0, 1]])})
    print('f2_test model.f2([0, 1]):', f2_test)
    print('logits2_test model.logits_2([0, 1]):', logits2_test)

    session.close()
    return W1, b1, W2, b2, loss


def plot(W1, b1, W2, b2, x_data, y_data):
    model = XORSigmoigPlot(W1, b1, W2, b2)

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
        x1_grid, x2_grid = np.meshgrid(np.linspace(0, np.max(x_data[:, 0]), 10),
                                       np.linspace(0, np.max(x_data[:, 1]), 10))
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


def plot2(W1, b1, W2, b2, x_train, y_train):
    model = XORSigmoigPlot(W1, b1, W2, b2)

    fig = plt.figure("Logistic regression: the logical XOR operator")

    plot1 = fig.add_subplot(121, projection='3d')

    plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green",
                         label="$h=$f1$(x)=\\sigma(x$W1$+$b1$)$")
    plot1_h1 = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]))
    plot1_h2 = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]))

    plot1.plot(
        x_train[:, 0].squeeze(),
        x_train[:, 1].squeeze(),
        y_train[:, 0].squeeze(),
        'o',
        label="$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$",
        color="blue")

    plot1_info = fig.text(0.01, 0.02, "")

    plot1.set_xlabel("$x_1$")
    plot1.set_ylabel("$x_2$")
    plot1.set_zlabel("$h_1,h_2$")
    plot1.legend(loc="upper left")
    plot1.set_xticks([0, 1])
    plot1.set_yticks([0, 1])
    plot1.set_zticks([0, 1])
    plot1.set_xlim(-0.25, 1.25)
    plot1.set_ylim(-0.25, 1.25)
    plot1.set_zlim(-0.25, 1.25)

    plot2 = fig.add_subplot(222, projection='3d')

    plot2_f2 = plot2.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green",
                                    label="$y=$f2$(h)=\\sigma(h $W2$+$b2$)$")

    plot2_info = fig.text(0.8, 0.9, "")

    plot2.set_xlabel("$h_1$")
    plot2.set_ylabel("$h_2$")
    plot2.set_zlabel("$y$")
    plot2.legend(loc="upper left")
    plot2.set_xticks([0, 1])
    plot2.set_yticks([0, 1])
    plot2.set_zticks([0, 1])
    plot2.set_xlim(-0.25, 1.25)
    plot2.set_ylim(-0.25, 1.25)
    plot2.set_zlim(-0.25, 1.25)

    plot3 = fig.add_subplot(224, projection='3d')

    plot3_f = plot3.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green",
                                   label="$y=f(x)=$f2$($f1$(x))$")

    plot3_info = fig.text(0.3, 0.03, "")

    plot3.set_xlabel("$x_1$")
    plot3.set_ylabel("$x_2$")
    plot3.set_zlabel("$y$")
    plot3.legend(loc="upper left")
    plot3.set_xticks([0, 1])
    plot3.set_yticks([0, 1])
    plot3.set_zticks([0, 1])
    plot3.set_xlim(-0.25, 1.25)
    plot3.set_ylim(-0.25, 1.25)
    plot3.set_zlim(-0.25, 1.25)

    table = plt.table(
        cellText=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], colWidths=[0.1] * 3,
        colLabels=["$x_1$", "$x_2$", "$f(x)$"], cellLoc="center", loc="lower right")

    def update_figure(event=None):
        if event is not None:
            if event.key == "W":
                model.W1[0, 0] += 0.2
            elif event.key == "w":
                model.W1[0, 0] -= 0.2
            elif event.key == "E":
                model.W1[0, 1] += 0.2
            elif event.key == "e":
                model.W1[0, 1] -= 0.2
            elif event.key == "R":
                model.W1[1, 0] += 0.2
            elif event.key == "r":
                model.W1[1, 0] -= 0.2
            elif event.key == "T":
                model.W1[1, 1] += 0.2
            elif event.key == "t":
                model.W1[1, 1] -= 0.2

            elif event.key == "Y":
                model.W2[0, 0] += 0.2
            elif event.key == "y":
                model.W2[0, 0] -= 0.2
            elif event.key == "U":
                model.W2[1, 0] += 0.2
            elif event.key == "u":
                model.W2[1, 0] -= 0.2

            elif event.key == "B":
                model.b1[0, 0] += 0.2
            elif event.key == "b":
                model.b1[0, 0] -= 0.2
            elif event.key == "N":
                model.b1[0, 1] += 0.2
            elif event.key == "n":
                model.b1[0, 1] -= 0.2

            elif event.key == "M":
                model.b2[0, 0] += 0.2
            elif event.key == "m":
                model.b2[0, 0] -= 0.2

            elif event.key == "c":
                model.W1 = W1.copy()
                model.W2 = W2.copy()
                model.b1 = b1.copy()
                model.b2 = b2.copy()

        nonlocal plot1_h1, plot1_h2, plot2_f2, plot3_f
        plot1_h1.remove()
        plot1_h2.remove()
        plot2_f2.remove()
        plot3_f.remove()
        x1_grid, x2_grid = np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
        h1_grid = np.empty([10, 10])
        h2_grid = np.empty([10, 10])
        f2_grid = np.empty([10, 10])
        f_grid = np.empty([10, 10])
        for i in range(0, x1_grid.shape[0]):
            for j in range(0, x1_grid.shape[1]):
                h = model.f1(np.array([[x1_grid[i, j], x2_grid[i, j]]]))
                h1_grid[i, j] = h[0, 0]
                h2_grid[i, j] = h[0, 1]
                f2_grid[i, j] = model.f2(np.array([[x1_grid[i, j], x2_grid[i, j]]]))
                f_grid[i, j] = model.f(np.array([[x1_grid[i, j], x2_grid[i, j]]]))

        plot1_h1 = plot1.plot_wireframe(x1_grid, x2_grid, h1_grid, color="lightgreen")
        plot1_h2 = plot1.plot_wireframe(x1_grid, x2_grid, h2_grid, color="darkgreen")

        plot1_info.set_text(
            "W1$=\\left[\\stackrel{%.2f}{%.2f}\\/\\stackrel{%.2f}{%.2f}\\right]$\nb1$=[{%.2f}, {%.2f}]$" %
            (model.W1[0, 0], model.W1[1, 0], model.W1[0, 1], model.W1[1, 1], model.b1[0, 0], model.b1[0, 1]))

        plot2_f2 = plot2.plot_wireframe(x1_grid, x2_grid, f2_grid, color="green")

        plot2_info.set_text("W2$=\\left[\\stackrel{%.2f}{%.2f}\\right]$\nb2$=[{%.2f}]$" % (
        model.W2[0, 0], model.W2[1, 0], model.b2[0, 0]))

        plot3_f = plot3.plot_wireframe(x1_grid, x2_grid, f_grid, color="green")

        plot3_info.set_text(
            "$loss = -\\frac{1}{n}\\sum_{i=1}^{n}\\left [ \\hat y^{(i)} \\log\\/f(\\hat x^{(i)}) + (1-\\hat y^{(i)}) \\log (1-f(\\hat x^{(i)})) \\right ] = %.2f$" %
            model.loss(x_train, y_train))

        table._cells[(1, 2)]._text.set_text("${%.1f}$" % model.f(np.array([[0, 0]])))
        table._cells[(2, 2)]._text.set_text("${%.1f}$" % model.f(np.array([[0, 1]])))
        table._cells[(3, 2)]._text.set_text("${%.1f}$" % model.f(np.array([[1, 0]])))
        table._cells[(4, 2)]._text.set_text("${%.1f}$" % model.f(np.array([[1, 1]])))

        fig.canvas.draw()

    update_figure()
    fig.canvas.mpl_connect('key_press_event', update_figure)

    plt.show()


if __name__ == '__main__':
    x_data = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
    y_data = np.array([[0],
                       [1],
                       [1],
                       [0]])
    W1, b1, W2, b2, loss = train(x_data, y_data)
    plot2(W1, b1, W2, b2, x_data, y_data)
