import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def get_data(mnist, folder_name):
    return eval('mnist.{0}.images'.format(folder_name))


def get_label(mnist, folder_name):
    return eval('mnist.{0}.labels'.format(folder_name))


def main():
    mnist_data = input_data.read_data_sets('MNIST_data/', one_hot=True)

    test_data = get_data(mnist_data, 'test')
    test_labels = get_label(mnist_data, 'test')

    learning_rate = 0.01

    x = tf.placeholder(tf.float32, [None, 784])
    real_values = tf.placeholder(tf.float32, [None, 10])

    weights = tf.Variable(tf.zeros([784, 10]))
    bias = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, weights) + bias)

    cross_entropy = -tf.reduce_sum(real_values * tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        cross_entropy)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_x, batch_y = mnist_data.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_x, real_values: batch_y})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(real_values, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(
        sess.run(accuracy, feed_dict={x: test_data, real_values: test_labels}))


if __name__ == '__main__':
    main()
