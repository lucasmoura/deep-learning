from tensorflow.examples.tutorials.mnist import input_data


def get_data(mnist, folder_name):
    return eval('mnist.{0}.images'.format(folder_name))


def get_label(mnist, folder_name):
    return eval('mnist.{0}.labels'.format(folder_name))


def main():
    mnist_data = input_data.read_data_sets('MNIST_data/', one_hot=True)

    train_data = get_data(mnist_data, 'train')
    train_labels = get_label(mnist_data, 'train')

    validation_data = get_data(mnist_data, 'validation')
    validation_labels = get_label(mnist_data, 'validation')

    test_data = get_data(mnist_data, 'test')
    test_labels = get_label(mnist_data, 'test')

    print('\n')
    print('Train data shape: {0}'.format(train_data.shape))
    print('Validation data shape: {0}'.format(validation_data.shape))
    print('Test data shape: {0}'.format(test_data.shape))


if __name__ == '__main__':
    main()
