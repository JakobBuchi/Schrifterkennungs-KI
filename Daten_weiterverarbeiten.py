import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

print("Lade Trainingsdaten...")
X = np.load("train_images.npy")          # (N,32,32) bereits 0-1 skaliert
y = np.load("train_labels.npy").flatten()

print("Shape:", X.shape, y.shape)
print("Klassenverteilung:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

X_train = X_train.reshape(-1, 32, 32, 1).astype("float32")
X_test  = X_test.reshape(-1, 32, 32, 1).astype("float32")

y_train_cat = to_categorical(y_train, 26)
y_test_cat  = to_categorical(y_test, 26)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(26, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

print("\nStarte Training...\n")
history = model.fit(
    X_train, y_train_cat,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stop],
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print("\nTest Accuracy:", acc)

model.save("buchstaben_ki_modell.h5")
print("✅ Modell gespeichert!")
