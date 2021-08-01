import os
import numpy as np

import tensorflow as tf
from pathlib import Path
import data_prep

EPOCHS = 30

base_model = tf.keras.models.load_model("/home/pi/SpeakerRecognitionOnRPi/models/model_10_speakers_dropout.h5")


train_ds, valid_ds, test_ds = data_prep.create_datasets(data_prep.DATASET_AUDIO_PATH)

print("DATA LOADED!!!")
print(train_ds)
print(valid_ds)
print(test_ds)


def residual_block(x, filters, conv_num=3, activation='relu'):
    s = tf.keras.layers.Conv1D(filters, 1, padding='same')(x)
    for i in range(conv_num - 1):
        x = tf.keras.layers.Conv1D(filters, 3, padding='same')(x)
        x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv1D(filters, 3, padding='same')(x)
    x = tf.keras.layers.Add()([x, s])
    x = tf.keras.layers.Activation(activation)(x)
    return tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)

def build_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape, name="input")
    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = tf.keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

class_names = os.listdir(data_prep.DATASET_AUDIO_PATH)
print("CLASS NAMES: ", class_names)
new_model = build_model((data_prep.SAMPLING_RATE // 2, 1), len(set(class_names)))
new_model.summary()

for i, layer in enumerate(new_model.layers[:-3]):
    new_model.layers[i].set_weights(base_model.layers[i].get_weights())
    new_model.layers[i].trainable = False

new_model.summary()

new_model.compile(
    optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model_save_filename = "test_model.h5"
earlystopping_cb = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
mdlcheckpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_save_filename, monitor="val_accuracy", save_best_only=True)
lr_tuner = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.000001)

history = new_model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    callbacks=[earlystopping_cb, lr_tuner, mdlcheckpoint_cb]
)

#Evaluation
print("Validation accuracy")
print(new_model.evaluate(valid_ds))
print("Test accuracy")
print(new_model.evaluate(test_ds))
