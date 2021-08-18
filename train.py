import os
import argparse
import pickle
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=str, default="3e-5")
    parser.add_argument("--epochs", type=str, default="1")
    args, _ = parser.parse_known_args()

    print(f"learning_rate: {args.learning_rate}")
    print(f"epochs: {args.epochs}")

    training_data_directory = "/opt/ml/input/data/train_data"
    train_features_data = os.path.join(training_data_directory, "train_features.pkl")
    train_labels_data = os.path.join(training_data_directory, "train_labels.pkl")
    with open(train_features_data, 'rb') as f:
        x_train = pickle.load(f)
    with open(train_labels_data, 'rb') as f:
        y_train = pickle.load(f)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=float(args.learning_rate)
    )
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'])

    # TODO: Make the number of epochs a hyperparameter
    model.fit(x_train, y_train, epochs=int(args.epochs))
    model_output_directory = "/opt/ml/model"
    model.save(model_output_directory)
