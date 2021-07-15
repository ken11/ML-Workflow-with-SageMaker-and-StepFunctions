import os
import pickle
import zipfile
import gzip
import numpy as np


if __name__ == "__main__":
    # NOTE: Performing MNIST tasks in TensorFlow is inherently easy,
    #       but I dare to do it with the input and pre-processing steps as an example.
    with zipfile.ZipFile('/opt/ml/processing/input/input.zip') as zf:
        zf.extractall('/opt/ml/processing/input')

    with gzip.open('/opt/ml/processing/input/train-images-idx3-ubyte.gz', 'rb') as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16)
    with gzip.open('/opt/ml/processing/input/t10k-images-idx3-ubyte.gz', 'rb') as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16)
    with gzip.open('/opt/ml/processing/input/train-labels-idx1-ubyte.gz', 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open('/opt/ml/processing/input/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)

    x_train = x_train.reshape(-1, 28, 28) / 255.0
    x_test = x_test.reshape(-1, 28, 28) / 255.0

    train_features_output_path = os.path.join(
        "/opt/ml/processing/train", "train_features.pkl")
    train_labels_output_path = os.path.join(
        "/opt/ml/processing/train", "train_labels.pkl")
    test_features_output_path = os.path.join(
        "/opt/ml/processing/test", "test_features.pkl")
    test_labels_output_path = os.path.join(
        "/opt/ml/processing/test", "test_labels.pkl")

    with open(train_features_output_path, 'wb') as f:
        pickle.dump(x_train, f, -1)

    with open(train_labels_output_path, 'wb') as f:
        pickle.dump(y_train, f, -1)

    with open(test_features_output_path, 'wb') as f:
        pickle.dump(x_test, f, -1)

    with open(test_labels_output_path, 'wb') as f:
        pickle.dump(y_test, f, -1)
