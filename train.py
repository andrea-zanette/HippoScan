import random
import os, shutil
from glob import glob
from data_generator import DataGenerator
from resunet import ResUNet
from metrics import dice_coef, dice_loss
from data_generator import parse_mask
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.metrics import Precision, Recall
from keras.optimizers import Nadam
import numpy as np

random.seed(0)
# Limit TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Paths to training/validation sets
train_images_path = "dataset/images/train"
train_masks_path = "dataset/masks/train"
valid_images_path = "dataset/images/valid"
valid_masks_path = "dataset/masks/valid"

# Where to save the model
model_path = "model"

# Where TensorBoard saves logs
log_path = "logs"

# Parameters
image_size = 256
batch_size = 8
lr = 1e-3
min_lr = 1e-7
epochs = 120

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

if __name__ == "__main__":
    
    # Check if dataset is present
    if not os.path.exists("dataset"):
        print("Can't find the dataset")
        exit()

    # Create folders for the model and the logs
    if os.path.exists(model_path):
        # Delete old logs
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    
    if os.path.exists(log_path):
        # Delete old logs
        shutil.rmtree(log_path)
    os.mkdir(log_path)
    
    # Find all images and masks
    train_images = glob(os.path.join(train_images_path, "*"))
    train_masks = glob(os.path.join(train_masks_path, "*"))
    valid_images = glob(os.path.join(valid_images_path, "*"))
    valid_masks = glob(os.path.join(valid_masks_path, "*"))

    # Sorting
    train_images.sort()
    train_masks.sort()
    valid_images.sort()
    valid_masks.sort()

    if len(train_images) != len(train_masks) or len(valid_images) != len(valid_masks):
        print("Some data are missing")
        exit()
  
    # Generators for the training process
    gen_train = DataGenerator(image_size, train_images, train_masks, batch_size)
    gen_valid = DataGenerator(image_size, valid_images, valid_masks, batch_size)
    
    # Build and compile the model
    arch = ResUNet(input_size=(image_size, image_size, 1), n_classes=1)
    model = arch.build_model()
    
    optimizer = Nadam(lr)
    metrics = [dice_coef, Recall(), Precision()]
    
    model.compile(loss=dice_loss, optimizer=optimizer, metrics=metrics)

    # Callbacks
    checkpoint = ModelCheckpoint(os.path.join(model_path, '{val_loss:.4f}'+".h5"), verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=min_lr, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False, verbose=1)
    tensorboard = TensorBoard(log_dir=log_path,  histogram_freq=1)
    callbacks = [checkpoint, reduce_lr, early_stopping, tensorboard]

    train_steps = len(train_images)//batch_size
    valid_steps = len(valid_images)//batch_size

    # Train the model
    model.fit(
        gen_train,
        validation_data=gen_valid,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        epochs=epochs,
        callbacks=callbacks
    )
