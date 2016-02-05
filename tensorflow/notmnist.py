import os
import sys
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
url = 'http://yaroslavvb.com/upload/notMNIST/'


def load(data_folders, min_num_images_per_class):
    dataset_names = []
    for folder in data_folders:
        dataset = load_letter(folder, min_num_images_per_class)
        set_filename = folder + '.pickle'
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


def extract(filename):
    tar = tarfile.open(filename)
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz

    print('Extracting data for %s. This may take a while. Please wait.' % root)

    if os.path.isdir(root) and len(os.listdir(root)) == num_classes:
        data_folders = sorted(os.listdir(root))
        return data_folders

    sys.stdout.flush()
    tar.extractall()
    tar.close()

    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if d != '.DS_Store']

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


def main():
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

    train_folders = extract(train_filename)
    test_folders = extract(test_filename)
    
    show_images(images_list)


if __name__ == "__main__":
    main()
