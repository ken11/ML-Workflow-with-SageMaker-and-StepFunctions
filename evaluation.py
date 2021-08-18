import os
import pickle
import tarfile
import json

import tensorflow as tf

from smexperiments.tracker import Tracker

if __name__ == "__main__":
    model_path = os.path.join("/opt/ml/processing/model", "model.tar.gz")
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    print("Loading model")
    model = tf.keras.models.load_model('./')

    test_features_data = os.path.join("/opt/ml/processing/test", "test_features.pkl")
    test_labels_data = os.path.join("/opt/ml/processing/test", "test_labels.pkl")
    with open(test_features_data, 'rb') as f:
        x_test = pickle.load(f)
    with open(test_labels_data, 'rb') as f:
        y_test = pickle.load(f)

    accuracy_score = model.evaluate(x_test, y_test, verbose=0)[1]

    with Tracker.load() as processing_tracker:
        processing_tracker.log_parameters({"evaluate:accuracy": accuracy_score})

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps({"accuracy": accuracy_score}))
