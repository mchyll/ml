import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time


class ConvolutionalNeuralNetworkModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)

        # Model variables
        W1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))  # 5x5 filters, 1 in-channel, 32 out-channels
        b1 = tf.Variable(tf.random_normal([32]))
        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))  # 3x3 filters, 32 in-channel, 64 out-channels
        b2 = tf.Variable(tf.random_normal([64]))
        W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 1024]))  # (width / 4) * (height / 4) * 64. Divided by 4 due to pooling twice
        b3 = tf.Variable(tf.random_normal([1024]))
        W4 = tf.Variable(tf.random_normal([1024, 10]))
        b4 = tf.Variable(tf.random_normal([10]))

        # Model operations
        conv1 = tf.nn.bias_add(tf.nn.conv2d(self.x, W1, strides=[1, 1, 1, 1], padding='SAME'), b1)  # Using builtin function for adding bias
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='SAME'), b2)
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        dense = tf.nn.bias_add(tf.matmul(tf.reshape(pool2, [-1, 7 * 7 * 64]), W3, name='Dense_1'), b3)
        dense = tf.nn.relu(dense)
        dense = tf.nn.dropout(dense, keep_prob=self.keep_prob)
        logits = tf.nn.bias_add(tf.matmul(dense, W4, name='Dense_2_logits'), b4)

        # Predictor
        f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)

        # Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f, 1), tf.argmax(self.y, 1)), tf.float32))


def train(x_train_batches, y_train_batches, x_test, y_test):
    model = ConvolutionalNeuralNetworkModel()

    # Training: adjust the model so that its loss is minimized
    minimize_operation = tf.train.AdamOptimizer(0.001).minimize(model.loss)

    # Create session object for running TensorFlow operations
    session = tf.Session()

    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    accuracies = []
    start = time.time()
    for epoch in range(20):
        for batch in range(len(x_train_batches)):
            session.run(minimize_operation, {model.x: x_train_batches[batch],
                                             model.y: y_train_batches[batch],
                                             model.keep_prob: 0.8})

        accuracy = session.run(model.accuracy, {model.x: x_test, model.y: y_test, model.keep_prob: 1.0})
        accuracies.append(accuracy)
        print("epoch", epoch)
        print("accuracy", accuracy)

    delta = time.time() - start
    print('Elapsed time:', delta)

    plt.plot(np.arange(20), accuracies)
    plt.show()

    session.close()


if __name__ == '__main__':
    (x_train_, y_train_), (x_test_, y_test_) = tf.keras.datasets.mnist.load_data()
    x_train_ = x_train_ / 255
    x_test_ = x_test_ / 255
    x_train = np.reshape(x_train_, (-1, 28, 28, 1))  # tf.nn.conv2d takes 4D arguments
    y_train = np.zeros((y_train_.size, 10))
    y_train[np.arange(y_train_.size), y_train_] = 1  # One-hot encodes the y-values

    batches = 600  # Divide training data into batches to speed up optimization
    x_train_batches = np.split(x_train, batches)
    y_train_batches = np.split(y_train, batches)

    x_test = np.reshape(x_test_, (-1, 28, 28, 1))
    y_test = np.zeros((y_test_.size, 10))
    y_test[np.arange(y_test_.size), y_test_] = 1

    train(x_train_batches, y_train_batches, x_test, y_test)
