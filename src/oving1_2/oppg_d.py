import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, art3d
from tqdm import tqdm


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class MNISTModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32, name='x', shape=[None, 784])
        self.y = tf.placeholder(tf.float32, name='y', shape=[None, 10])

        # Model variables
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))

        # Single layer
        self.logits = tf.matmul(self.x, self.W) + self.b
        self.f = tf.nn.softmax(self.logits)

        # Uses cross entropy
        self.loss = tf.losses.softmax_cross_entropy(self.y, self.logits)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.f, 1), tf.argmax(self.y, 1)), tf.float32))


def train(x_train, y_train, x_test, y_test):
    model = MNISTModel()

    # Training: adjust the model so that its loss is minimized
    minimize_operation = tf.train.GradientDescentOptimizer(2).minimize(model.loss)

    # Create session object for running TensorFlow operations
    session = tf.Session()

    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    for epoch in range(300):
        session.run(minimize_operation, {model.x: x_train, model.y: y_train})
        if epoch % 100 == 0:
            accuracy = session.run([model.accuracy], {model.x: x_test, model.y: y_test})
            print('Epoch: {}, accuracy: {}'.format(epoch, accuracy))

    # Evaluate training accuracy
    W, b, loss = session.run([model.W, model.b, model.loss],
                             {model.x: x_train, model.y: y_train})
    accuracy = session.run([model.accuracy], {model.x: x_test, model.y: y_test})

    print("W = %s\nb = %s\nloss = %s\naccuracy = %s" % (W.shape, b.shape, loss, accuracy))

    session.close()
    return W, b, loss


def flatten(data):
    flat = np.empty((data.shape[0], 784))
    for i in range(data.shape[0]):
        flat[i, :] = data[i, :].reshape([1, 784])
    return flat


def onehot_encode(labels):
    result = np.empty((labels.shape[0], 10))
    for i in range(labels.shape[0]):
        result[i, :] = np.array([[1 if j == labels[i] else 0 for j in range(10)]])
    return result


if __name__ == '__main__':
    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = flatten(x_train[:6000, :])
    y_train = onehot_encode(y_train[:6000])
    x_test = flatten(x_test)
    y_test = onehot_encode(y_test)
    '''

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x_train = mnist.train.images[:5500, :]
    y_train = mnist.train.labels[:5500, :]
    x_test = mnist.test.images[:10000, :]
    y_test = mnist.test.labels[:10000, :]

    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)

    W, b, loss = train(x_train, y_train, x_test, y_test)
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.title(i)
        plt.imshow(W[:, i].reshape([28, 28]), cmap=plt.get_cmap('seismic'))
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)

    # plt.imsave('x_train_1.png', x_train[0, :].reshape([28, 28]))
    plt.show()
