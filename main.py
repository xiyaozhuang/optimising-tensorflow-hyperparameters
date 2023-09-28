# %%
# Import libraries
import optuna
import tensorflow as tf


# %%
# Define objective
def objective(trial):
    # Load data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Preprocess data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Optimise layers
    n_layers = trial.suggest_int("n_layers", 1, 3)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        model.add(tf.keras.layers.Dense(num_hidden, activation="relu"))

    model.add(tf.keras.layers.Dense(10))

    # Compile model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Train model
    model.fit(train_images, train_labels, epochs=10)

    # Evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    return test_acc


# %%
# Run trials
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_trial)
