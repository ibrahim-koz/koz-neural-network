import pickle
from os import listdir
from os.path import join

import cv2
import numpy as np


def shuffle_together(a, b):
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)

    a = a[indices]
    b = b[indices]
    return a, b


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def process_data(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    resize_weight = 30
    resize_height = 30
    img = image_resize(img, resize_weight, resize_height)
    return np.array([img.flatten()])


def process_directory(dir_path):
    images = []
    for f in listdir(dir_path):
        file_path = join(dir_path + f)
        image = process_data(file_path)
        if image.shape[1] == 900:
            images.append(image)
    return images


# We've extracted our intended features from our dataset.
# training_set = dict()
# training_set_path = "data/seg_train/seg_train/"
# training_set_categories = listdir(training_set_path)
# for category in training_set_categories:
#     training_set[category] = process_directory(join(training_set_path + category + "/"))

# training_set_out = open("feature_extracted_data/training_set.pickle", "wb")
# pickle.dump(training_set, training_set_out)
# training_set_out.close()

# validation_set = dict()
# validation_set_path = "data/seg_dev/seg_dev/"
# validation_set_categories = listdir(validation_set_path)
# for category in validation_set_categories:
#     validation_set[category] = process_directory(join(validation_set_path + category + "/"))

# validation_set_out = open("feature_extracted_data/validation_set.pickle", "wb")
# pickle.dump(validation_set, validation_set_out)
# validation_set_out.close()


# training_set_in = open("feature_extracted_data/training_set.pickle", "rb")
# training_set = pickle.load(training_set_in)
# del training_set_in
#
# validation_set_in = open("feature_extracted_data/validation_set.pickle", "rb")
# validation_set = pickle.load(validation_set_in)
# del validation_set_in

# x_train = np.concatenate((list(training_set.values())))
# y_train = []
# for num, category in enumerate(training_set.keys()):
#     y_train.extend([num] * len(training_set[category]))
# y_train = np.array(y_train)
# x_train, y_train = shuffle_together(x_train, y_train)
#
# x_train_out = open("feature_extracted_data/x_train.pickle", "wb")
# pickle.dump(x_train, x_train_out)
# x_train_out.close()
#
# y_train_out = open("feature_extracted_data/y_train.pickle", "wb")
# pickle.dump(y_train, y_train_out)
# y_train_out.close()
#
# x_validation = np.concatenate((list(validation_set.values())))
# y_validation = []
# for num, category in enumerate(validation_set.keys()):
#     y_validation.extend([num] * len(validation_set[category]))
# y_validation = np.array(y_validation)
# x_validation, y_validation = shuffle_together(x_validation, y_validation)
#
# x_validation_out = open("feature_extracted_data/x_validation.pickle", "wb")
# pickle.dump(x_validation, x_validation_out)
# x_validation_out.close()
#
# y_validation_out = open("feature_extracted_data/y_validation.pickle", "wb")
# pickle.dump(y_validation, y_validation_out)
# y_validation_out.close()