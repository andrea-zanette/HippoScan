import numpy as np
import os, argparse
from glob import glob
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from data_generator import DataGenerator, parse_mask
from inference import evaluate_model
from metrics import dice_loss, dice_coef

# Limit TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

calib_images_path = "dataset/images/calib"
calib_masks_path = "dataset/masks/calib"
model_path = "model"
image_size = 256
batch_size = 8

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

def quantize_model(model_name, quant_model_name, calib_size, fft, fft_epochs):
    
    if not os.path.exists(os.path.join(model_path, model_name+".h5")):
        print("Model name not valid")
    
    calib_images = glob(os.path.join(calib_images_path, "*"))
    calib_masks = glob(os.path.join(calib_masks_path, "*"))
    calib_images.sort()
    calib_masks.sort()

    if len(calib_images) < calib_size:
        print("Not enough samples! Reducing calibration set dimension to "+str(len(calib_images)))
        calib_size = len(calib_images)

    calib_images = calib_images[:calib_size]
    calib_masks = calib_images[:calib_size]

    gen_calib = DataGenerator(image_size, calib_images, calib_masks, batch_size)
    
    with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
        float_model = load_model(os.path.join(model_path, model_name+".h5"))

    from tensorflow_model_optimization.quantization.keras import vitis_quantize
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    if fft:
        quant_model_name += "_fft"
        quantized_model = quantizer.quantize_model(calib_dataset=gen_calib, verbose=1, include_fast_ft=True, fast_ft_epochs = fft_epochs)
    else:
        quantized_model = quantizer.quantize_model(calib_dataset=gen_calib, verbose=1)

    quantized_model.save(os.path.join(model_path, quant_model_name+".h5"))


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_name', type=str, help='Name of the float model (without the .h5 extension)', required=True)
    ap.add_argument('-e', '--evaluate_post', help="Set this flag to automatically evaluate the quantized model", default=False, required=False, action='store_true')
    ap.add_argument('-s', '--calib_size', type=int, default=540, help='Dimension of the calibration dataset. Default is 540', required=False)
    ap.add_argument('-fft', '--fastfinetuning', help="Perform Fast Fine Tuning instead of Post Training Quantization. Default is False", default=False, required=False, action='store_true')
    ap.add_argument('-fft_e', '--fastfinetuning_epochs', type=int, help="Set how many epochs are performed with Fast Fine Tuning. Default is 10", default=10, required=False)
    args = ap.parse_args()

    quant_model_name = args.model_name+"_quant"
    quantize_model(
        args.model_name, 
        quant_model_name, 
        calib_size = args.calib_size, 
        fft = args.fastfinetuning, 
        fft_epochs = args.fastfinetuning_epochs
    )

    if args.evaluate_post:
        evaluate_model(quant_model_name, True)
