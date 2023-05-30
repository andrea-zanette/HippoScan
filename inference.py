import numpy as np
import os, argparse, csv, random, time
from glob import glob
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.callbacks import Callback
from data_generator import DataGenerator
from metrics import dice_loss, dice_coef
import cv2
from keras.metrics import Precision, Recall, MeanIoU
from keras.optimizers import Nadam

# Limit TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

test_images_path = "dataset/images/test"
test_masks_path = "dataset/masks/test"
model_path = "model"
log_path = "logs"
folder_name = "evaluation"
image_size = 256
batch_size = 8
lr = 1e-5

class CSVLoggerEvaluation(Callback):
    """A custom callback for writing model evaluation results at the end of each batch

        Args:
            log_path (String): The path where to save the results
            model_name (String): The name of the model
    """
    def __init__(self, log_path, model_name):
        self.log_path = log_path
        self.model_name = model_name
        self.first_batch = True
        self.dest_file = None
        self.writer = None

    def on_test_begin(self, logs=None):
        # Check if destination folder exists
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        # Create the .csv file (or reset if already exists)
        self.dest_file = open(os.path.join(self.log_path, self.model_name+".csv"), mode='w')
        # Create the writer
        self.writer = csv.writer(self.dest_file)

    def on_test_batch_end(self, batch, logs=None):
        if self.first_batch:
            # Save the headers on the first batch
            self.writer.writerow(list(logs.keys()))
            self.first_batch = False
        # Save batch results
        self.writer.writerow(list(logs.values()))
   
    def on_test_end(self, logs=None):
        # Close file stream
        self.dest_file.close()

def evaluate_model(model_name, is_quantized):
    """This function evaluates the model on the test set

        Args:
            model_name (String): The name of the model
            is_quantized (Boolean): If the given model has been quantized
        Returns: 
            None
    """

    if not os.path.exists(os.path.join(model_path, model_name+".h5")):
        print("Model name not valid")

    test_images = glob(os.path.join(test_images_path, "*"))
    test_masks = glob(os.path.join(test_masks_path, "*"))
    test_images.sort()
    test_masks.sort()

    if len(test_images) != len(test_masks):
        print("Some data are missing")
        exit()

    gen_test = DataGenerator(image_size, test_images, test_masks, batch_size)

    if is_quantized:
        from tensorflow_model_optimization.quantization.keras import vitis_quantize
        with vitis_quantize.quantize_scope():
            model = load_model(os.path.join(model_path, model_name+".h5"))
        optimizer = Nadam(lr)
        metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2)]
        model.compile(loss=dice_loss, optimizer=optimizer, metrics=metrics)
    else:
        with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
            model = load_model(os.path.join(model_path, model_name+".h5"))

    csv_cb = CSVLoggerEvaluation(os.path.join(log_path, folder_name), model_name)
    callbacks=[csv_cb]
    
    start_time = time.perf_counter()
    model.evaluate(
        gen_test, 
        verbose=1, 
        steps=(len(test_images)//batch_size),
        callbacks=callbacks
    )
    end_time = time.perf_counter()
    print('Evaluation took %.3f seconds.' % (end_time-start_time))

def show_predictions(model_name, is_quantized, how_many=10):
    """This function prints images, their ground truth mask, and model prediction.
        Press any key to skip to the following sample.
        Images are taken randomly from the test set.

        Args:
            model_name (String): The name of the model
            is_quantized (Boolean): If the given model has been quantized
            how_many (Integer): How many samples to display
        Returns: 
            None
    """

    if not os.path.exists(os.path.join(model_path, model_name+".h5")):
        print("Model name not valid")

    test_images = glob(os.path.join(test_images_path, "*"))
    test_masks = glob(os.path.join(test_masks_path, "*"))
    test_images.sort()
    test_masks.sort()

    # Shuffle randomly and extract how_many samples
    zip_paths = list(zip(test_images, test_masks))
    random.shuffle(zip_paths)
    test_images, test_masks = zip(*zip_paths)

    test_images = test_images[:how_many]
    test_masks = test_masks[:how_many]

    gen_test = DataGenerator(image_size, test_images, test_masks, 1)

    if is_quantized:
        from tensorflow_model_optimization.quantization.keras import vitis_quantize
        with vitis_quantize.quantize_scope():
            model = load_model(os.path.join(model_path, model_name+".h5"))
        optimizer = Nadam(lr)
        metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2)]
        model.compile(loss=dice_loss, optimizer=optimizer, metrics=metrics)
    else:
        with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
            model = load_model(os.path.join(model_path, model_name+".h5"))

    for i in range(how_many):
        img, mask = gen_test.__getitem__(i)
        
        pred = model.predict(np.expand_dims(img[0], axis=0))

        cv2.imshow("Data ("+str(i+1)+"/"+str(how_many)+")", img[0])
        cv2.imshow("Mask ("+str(i+1)+"/"+str(how_many)+")", mask[0])
        cv2.imshow("Prediction ("+str(i+1)+"/"+str(how_many)+")", pred[0])

        cv2.waitKey()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_name', type=str, help='Model name', required=True)
    ap.add_argument('-s', '--show_predictions', help="Set this flag to true to see some predictions", action='store_true', default=False, required=False)
    ap.add_argument('-n', '--how_many', type=int, help="Only if show_predictions is True, set how many samples to display", default=10, required=False)
    ap.add_argument('-q', '--is_quantized', help="Set to True if the model is quantized", action='store_true', default=False)
    args = ap.parse_args()

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
    
    if args.show_predictions:
        show_predictions(args.model_name, args.is_quantized, args.how_many)
    else:
        evaluate_model(args.model_name, args.is_quantized)
