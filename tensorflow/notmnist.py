import os
import sys
import random
import tarfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve

image_size = 28  # Pixel width and height.
images_list = ['notMNIST_large/A/a2F6b28udHRm.png',
               'notMNIST_large/A/a2FkZW4udHRm.png']
num_classes = 10
pixel_depth = 255.0  # Number of levels per pixel.
train_size = 200000
test_size = 10000
valid_size = 10000
url = 'http://yaroslavvb.com/upload/notMNIST/'

np.random.seed(133)


def check_overlap_data(dataset1, dataset2):
    dataset1.flags.writeable = False
    dataset2.flags.writeable = False

    hash1 = set([hash(image.data) for image in dataset1])
    hash2 = set([hash(image.data) for image in dataset2])

    return len(set.intersection(hash1, hash2))


def load(data_folders, min_num_images_per_class):
    dataset_names = []

    for folder in data_folders:
        set_filename = folder + '.pickle'

        if os.path.isfile(set_filename):
            dataset_names.append(set_filename)
            continue

        dataset = load_letter(folder, min_num_images_per_class)

        try:
            with open(set_filename, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            dataset_names.append(set_filename)
        except Exception as e:
            print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def load_letter(folder, min_num_images):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    image_index = 0
    print folder
    for image in os.listdir(folder):
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s'
                                % str(image_data.shape))
            dataset[image_index, :, :] = image_data
            image_index += 1
        except IOError as e:
            print('Could not read:', image_file,
                  ':', e, '- it\'s ok, skipping.')

    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


# Download a file if not present, and make sure it's the right size.
def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)

    statinfo = os.stat(filename)

    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
        return filename
    else:
        raise Exception('Failed to verify' + filename +
                        '. Can you get to it with a browser?')


def get_folders_name(root):
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if d != '.DS_Store' and os.path.isdir(os.path.join(root, d))]

    return data_folders


def extract(filename):
    tar = tarfile.open(filename)
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz

    print('Extracting data for %s. This may take a while. Please wait.' % root)

    if os.path.isdir(root):
        folders = [d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))]

        if len(folders) == num_classes:
            return get_folders_name(root)

    sys.stdout.flush()
    tar.extractall()
    tar.close()

    data_folders = get_folders_name(root)

    if len(data_folders) != num_classes:
        raise Exception('Expected %d folders, one per class. Found %d instead.'
                        % (num_classes, len(data_folders)))

    print(data_folders)
    return data_folders


def show_images(images_list):
    for image in images_list:
        img = mpimg.imread(image)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()


def show_sample_image_from_dataset(dataset_list):
    dataset_id = random.randint(0, len(dataset_list)-1)

    with open(dataset_list[dataset_id], 'r') as dataset:
        images = pickle.load(dataset)

    image_id = random.randint(0, len(images)-1)
    image_array = images[image_id]
    plt.imshow(image_array, cmap=plt.cm.gray)
    plt.show()


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def verify_dataset_data_balance(dataset_list):
    num_data_per_class = []

    for dataset in dataset_list:
        with open(dataset, 'r') as d:
            images = pickle.load(d)
            num_data_per_class.append(images.shape[0])

    num_data_array = np.array(num_data_per_class)

    return True if num_data_array.std() < 1.2 else False


def main():
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

    train_folders = extract(train_filename)
    test_folders = extract(test_filename)

    train_datasets = load(train_folders, 45000)
    test_datasets = load(test_folders, 1800)

    # show_images(images_list)
    # show_sample_image_from_dataset(train_datasets)
    if verify_dataset_data_balance(train_datasets):
        print 'Train dataset is well balanced!'

    if verify_dataset_data_balance(test_datasets):
        print 'Test dataset is well balanced!'

    [valid_dataset, valid_labels,
     train_dataset, train_labels] = merge_datasets(train_datasets,
                                                   train_size, valid_size)
    __, __, test_dataset, test_labels = merge_datasets(test_datasets,
                                                       test_size)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)

    print ('Overlap between training and validation data: %d'
           % check_overlap_data(train_dataset, valid_dataset))
    print ('Overlap between training and test data: %d'
           % check_overlap_data(train_dataset, test_dataset))
    print ('Overlap between validation and test data: %d'
           % check_overlap_data(valid_dataset, test_dataset))


if __name__ == "__main__":
    main()
