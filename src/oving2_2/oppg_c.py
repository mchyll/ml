import tensorflow as tf
from oving2_2.coding import create_coding, encode_string, encode_feature


class LongShortTermMemoryModel:
    def __init__(self, encodings_size_in, encodings_size_out):
        # Model constants
        cell_state_size = 128

        # Cells
        cell = tf.contrib.rnn.BasicLSTMCell(cell_state_size)

        # Model input
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')  # Needed by cell.zero_state call, and can be dependent on usage (training or generation)
        self.x = tf.placeholder(tf.float32, [None, None, encodings_size_in], name='x')  # Shape: [batch_size, max_time, encodings_size]
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


def train(x_train, y_train, x_encoding_len, y_encoding_len, out_labels, word_to_test):
    model = LongShortTermMemoryModel(x_encoding_len, y_encoding_len)

    # Training: adjust the model so that its loss is minimized
    minimize_operation = tf.train.RMSPropOptimizer(0.05).minimize(model.loss)

    # Create session object for running TensorFlow operations
    session = tf.Session()

    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    # Initialize model.in_state
    zero_state = session.run(model.in_state, {model.batch_size: len(x_train)})

    for epoch in range(500):
        session.run(minimize_operation, {model.batch_size: 1, model.x: x_train, model.y: y_train, model.in_state: zero_state})

        if epoch % 10 == 9:
            print("epoch", epoch)
            print("loss", session.run(model.loss, {model.batch_size: 1, model.x: x_train, model.y: y_train, model.in_state: zero_state}))

            # Generate emoji from word_to_test
            state = session.run(model.in_state, {model.batch_size: 1})
            y, state = session.run([model.f, model.out_state], {
                model.batch_size: 1,
                model.x: [word_to_test],
                model.in_state: state
            })

            print(out_labels[y.argmax()])

    session.close()


if __name__ == '__main__':
    train_data = [
        ('hat ', '\U0001f3a9'),
        ('cat ', '\U0001f408'),
        ('matt', '\U0001f468'),
        ('flat', '\U0001f3d8'),
        ('cap ', '\U0001f9e2'),
        ('cup ', '\U0001f964'),
        ('dog ', '\U0001f415'),
        ('many', '\U0000274c')
    ]

    alphabet = ''.join(set(''.join([data[0] for data in train_data])))
    emojis = ''.join(set([data[1] for data in train_data]))

    char_encodings, _ = create_coding(alphabet)
    emoji_encodings, _ = create_coding(emojis)

    x_train = [encode_string(data[0], char_encodings) for data in train_data]
    y_train = [encode_feature(data[1], emoji_encodings) for data in train_data]

    train(x_train, y_train, len(alphabet), len(emojis), emojis, encode_string('many cat', char_encodings))
