"""
model_training.py
Example training script to create a simple CNN for face classification.
Directory expected:
  dataset/
    train/
      known/   -> images of known person(s)
      unknown/ -> images of other faces/background
    val/
      known/
      unknown/

Run: python model_training.py
"""
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Config
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "face_cnn_model.h5")
LABEL_MAP_FILE = os.path.join(MODEL_DIR, "label_map.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# Data generators with basic augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.05,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   fill_mode="nearest")

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[1], IMG_SIZE[0], 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# Callbacks
checkpoint = ModelCheckpoint(MODEL_FILE, monitor="val_accuracy", save_best_only=True, verbose=1)
earlystop = EarlyS
