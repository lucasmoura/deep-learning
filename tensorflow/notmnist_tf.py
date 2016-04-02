from __future__ import print_function
import numpy as np
import tensorflow as tf

from six.moves import cPickle as pickle
from six.moves import range

image_size = 28
num_steps = 801
num_labels = 10
num_hidden_neurons = 1024
pickle_file = 'notMNIST.pickle'
batch_size = 128
beta = 0.007


def forward_propagation(data, weights_01, weights_02, biases_01, biases_02,
                        keep_probability):
    n1 = tf.nn.relu(tf.matmul(data, weights_01) + biases_01)
    n1 = tf.nn.dropout(n1, keep_probability)
    return tf.matmul(n1, weights_02) + biases_02


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Transforms labels into one hot enconding format
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def main():
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    graph = tf.Graph()
    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be
        # fed at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32,
                                         shape=(batch_size, num_labels))
        keep_probability = tf.placeholder(tf.float32)
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random valued following a
        # (truncated) normal distribution. The biases get initialized to zero.
        weights_01 = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_hidden_neurons]))
        biases_01 = tf.Variable(tf.zeros([num_hidden_neurons]))

        weights_02 = tf.Variable(
            tf.truncated_normal([num_hidden_neurons, num_labels]))
        biases_02 = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases.
        # We compute the softmax and cross-entropy (it's one operation in
        # TensorFlow, because it's very common, and it can be optimized).
        # We take the average of this cross-entropy across all
        # training examples: that's our loss.
        logits = forward_propagation(tf_train_dataset, weights_01, weights_02,
                                     biases_01, biases_02, keep_probability)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Applying L2 regularization
        regularizers = tf.nn.l2_loss(weights_01) + tf.nn.l2_loss(weights_02)
        loss += beta * regularizers

        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(logits)

        valid_prediction = tf.nn.softmax(
            forward_propagation(tf_valid_dataset, weights_01, weights_02,
                                biases_01, biases_02, keep_probability))
        test_prediction = tf.nn.softmax(
            forward_propagation(tf_test_dataset, weights_01, weights_02,
                                biases_01, biases_02, keep_probability))

    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get
        # initialized as we described in the graph: random weights for the
        # matrix, zeros for the biases.
        tf.initialize_all_variables().run()
        print('Initialized')

        for step in range(num_steps):
            # Pick an offset within the training data, which has been
            # randomized. Note: we could use better randomization
            # across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the
            # minibatch. The key of the dictionary is the placeholder node
            # of the graph to be fed,  and the value is the numpy array to
            # feed to it.
            feed_dict = {tf_train_dataset: batch_data,
                         tf_train_labels: batch_labels,
                         keep_probability: 0.5}

            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)

            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions,
                                                              batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(feed_dict={keep_probability: 1.0}),
                    valid_labels))

        print('Test accuracy: %.1f%%' % accuracy(
            test_prediction.eval(feed_dict={keep_probability: 1.0}),
            test_labels))


if __name__ == '__main__':
    main()
