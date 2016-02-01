import os
import sys
import tarfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from six.moves.urllib.request import urlretrieve


images_list = ['notMNIST_large/A/a2F6b28udHRm.png',
               'notMNIST_large/A/a2FkZW4udHRm.png']
num_classes = 10
url = 'http://yaroslavvb.com/upload/notMNIST/'


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
