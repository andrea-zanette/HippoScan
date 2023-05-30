import numpy as np
import cv2
from tensorflow import keras
from keras.utils import Sequence

def parse_image(img_path, image_size):
    image = cv2.imread(img_path, 0)
    h, w = image.shape
    if (h == image_size) and (w == image_size):
        pass
    else:
        image = cv2.resize(image, (image_size, image_size))
    # Enlarge shape to (image_size, image_size, 1)
    image = np.expand_dims(image, -1)
    # Normalize pixels between [0.,1.]
    image = image.astype('float32') / 255.0
    # Remove background noise
    image = np.where(image > 0.05, image, 0.0).astype('float32')
    return image

def parse_mask(mask_path, image_size):
    mask = cv2.imread(mask_path, 0)
    h, w = mask.shape
    if (h == image_size) and (w == image_size):
        pass
    else:
        mask = cv2.resize(mask, (image_size, image_size))
    mask = np.expand_dims(mask, -1)
    mask = mask.astype('float32') / 255.0
    # Mask's pixels must be 0's or 1's
    mask = np.where(mask > 0.5, 1.0, 0.0).astype('float32')
    return mask

class DataGenerator(Sequence):
    def __init__(self, image_size, images_paths, masks_paths, batch_size=8):
        self.image_size = image_size
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.batch_size = batch_size

    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.images_paths):
            self.batch_size = len(self.images_paths) - index*self.batch_size

        batch_images_paths = self.images_paths[index*self.batch_size : (index+1)*self.batch_size]
        batch_masks_batch = self.masks_paths[index*self.batch_size : (index+1)*self.batch_size]

        images_batch = []
        masks_batch = []

        for i in range(self.batch_size):
            # Parse and return the image and its mask
            image = parse_image(batch_images_paths[i], self.image_size)
            mask = parse_mask(batch_masks_batch[i], self.image_size)
            images_batch.append(image)
            masks_batch.append(mask)
        return np.array(images_batch), np.array(masks_batch)
    
    def __len__(self):
        return int(np.ceil(len(self.images_paths)/float(self.batch_size)))
