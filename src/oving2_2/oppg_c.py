import numpy as np
import tensorflow as tf
from oving2_2.coding import create_coding, encode_string, encode_feature


class LongShortTermMemoryModel:
    def __init__(self, encodings_size_in, encodings_size_out):
        # Model constants
        cell_state_size = 128
        cell_state_size = encodings_size_in

        # Cells
        cell = tf.contrib.rnn.BasicLSTMCell(cell_state_size)

        # Model input
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')  # Needed by cell.zero_state call, and can be dependent on usage (training or generation)
        self.x = tf.placeholder(tf.float32, [3, None, encodings_size_in], name='x')  # Shape: [batch_size, max_time, encodings_size]
        self.y = tf.placeholder(tf.float32, [None, encodings_size_out], name='y')  # Shape: [batch_size, encodings_size]
        self.in_state = cell.zero_state(self.batch_size, tf.float32)  # Can be used as either an input or a way to get the zero state

        # Model variables
        W = tf.Variable(tf.random_normal([cell_state_size, encodings_size_out]))
        b = tf.Variable(tf.random_normal([encodings_size_out]))

        # Model operations
        lstm, self.out_state = tf.nn.dynamic_rnn(cell, self.x, initial_state=self.in_state)  # lstm has shape: [batch_size, max_time, cell_state_size]

        # Logits, where tf.einsum multiplies a batch of txs matrices (lstm) with W
        logits = tf.nn.bias_add(tf.matmul(lstm[:, -1, :], W), b)  # b: batch, t: time, s: state, e: encoding

        # Predictor
        self.f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)


def train(x_train, y_train, char_encodings, index_to_char, batch_size):
    model = LongShortTermMemoryModel(len(index_to_char), len(y_train))

    # Training: adjust the model so that its loss is minimized
    minimize_operation = tf.train.RMSPropOptimizer(0.05).minimize(model.loss)

    # Create session object for running TensorFlow operations
    session = tf.Session()

    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    # Initialize model.in_state
    zero_state = session.run(model.in_state, {model.batch_size: batch_size})

    for epoch in range(500):
        session.run(minimize_operation, {model.batch_size: batch_size, model.x: x_train, model.y: y_train, model.in_state: zero_state})

        if epoch % 10 == 9:
            print("epoch", epoch)
            print("loss", session.run(model.loss, {model.batch_size: batch_size, model.x: x_train, model.y: y_train, model.in_state: zero_state}))

            # Generate characters from the initial characters ' h'
            state = session.run(model.in_state, {model.batch_size: batch_size})
            y, state = session.run([model.f, model.out_state], {model.batch_size: batch_size, model.x: [[char_encodings['c']]], model.in_state: state})  # ' '
            y, state = session.run([model.f, model.out_state], {model.batch_size: batch_size, model.x: [[char_encodings['a']]], model.in_state: state})  # 'h'
            y, state = session.run([model.f, model.out_state], {model.batch_size: batch_size, model.x: [[char_encodings['t']]], model.in_state: state})  # 'h'
            y, state = session.run([model.f, model.out_state], {model.batch_size: batch_size, model.x: [[char_encodings[' ']]], model.in_state: state})  # 'h'
            text = index_to_char[y.argmax()]
            # for c in range(50):
            #     y, state = session.run([model.f, model.out_state], {
            #         model.batch_size: batch_size,
            #         model.x: [[char_encodings[index_to_char[y.argmax()]]]],
            #         model.in_state: state
            #     })
            #     text += index_to_char[y[0].argmax()]
            print(text)

    session.close()


if __name__ == '__main__':
    alphabet = ' hatcm'  # Strings are iterable, no need for the list index_to_chars
    char_encodings, _ = create_coding(alphabet)

    emojis = [
        '\U0001f3a9',
        '\U0001f408',
        '\U0001f468'
    ]
    emoji_encodings, _ = create_coding(emojis)

    x_train = [
        encode_string('hat ', char_encodings),
        encode_string('cat ', char_encodings),
        encode_string('matt', char_encodings)
    ]
    y_train = [
        encode_feature('\U0001f3a9', emoji_encodings),
        encode_feature('\U0001f408', emoji_encodings),
        encode_feature('\U0001f468', emoji_encodings)
    ]

    train(x_train, y_train, char_encodings, alphabet, batch_size=3)
