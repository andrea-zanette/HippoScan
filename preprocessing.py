import os, random, shutil, cv2, re
import numpy as np
from glob import glob
from tqdm import tqdm

random.seed(0)

IMG_SIZE = 256
PATTERN = r'[^/]+/[^/]+/[^/]+/([^_]+_[^_]+_[^_]+_[^_]+_[^_]+)[^/]+_(\w+\.jpg)'

old_train_images_path = "archive/original/100"
old_test_images_path = "archive/original/35"
old_train_masks_path = "archive/label/100label"
old_test_masks_path = "archive/label/35label"

dataset_name = "dataset"
image_folder = "images"
mask_folder = "masks"
sets = [["train", "valid"], ["test", "calib"]]
splits = [[0.7, 0.3], [0.5, 0.5]]


def parse_image(img_path):
    # Read image in grayscale
    image = cv2.imread(img_path, 0)
    h, w = image.shape
    if (h == IMG_SIZE) and (w == IMG_SIZE):
        pass
    else:
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = np.expand_dims(image, -1)
    return image

def parse_mask(mask_paths):
    mask_1 = cv2.imread(mask_paths[0], 0)
    mask_2 = cv2.imread(mask_paths[1], 0)
    mask = cv2.addWeighted(mask_1, 1, mask_2, 1, 0)
    h, w = mask.shape
    if (h == IMG_SIZE) and (w == IMG_SIZE):
        pass
    else:
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = np.expand_dims(mask, -1)
    # Remove any gray pixel
    mask = np.where(mask > 20, 255, 0)
    return mask

def parse_and_save(images_paths, masks_paths, dest_images, dest_masks):
    corrupted = id = 0
    # Create set destination folders
    os.mkdir(dest_images)
    os.mkdir(dest_masks)
    with tqdm(total=len(images_paths)) as pbar:
        for img_path in images_paths:
            # Extract the folder name and the sample number in the archive
            match = re.search(PATTERN, img_path)
            folder = match.group(1)
            num = match.group(2)
            # Find the two corresponding masks
            mask_paths = [path for path in masks_paths if folder in path and num in path]
            if len(mask_paths) != 2:
                print(folder)
                corrupted += 1
            else:
                # Parse and save results
                mask = parse_mask(mask_paths)
                pos_pixels = np.sum(mask.flatten())
                if pos_pixels > 20:
                    image = parse_image(img_path)
                    cv2.imwrite(os.path.join(dest_images, str(id)+".jpg"), image)
                    cv2.imwrite(os.path.join(dest_masks, str(id)+".jpg"), mask)
                    id += 1
            pbar.update(1)
    return corrupted


if __name__ == "__main__":

    # Check if archive is present
    if not os.path.exists("archive"):
        print("Original archive missing")
        exit()

    # Remove dataset if already present
    if os.path.exists("dataset"):
        shutil.rmtree("dataset")

    os.makedirs(os.path.join(dataset_name, image_folder))
    os.makedirs(os.path.join(dataset_name, mask_folder))

    # Find all the images and masks of the archive
    old_train_images = glob(os.path.join(old_train_images_path, "*", "*"))
    old_train_masks = glob(os.path.join(old_train_masks_path, "*", "*", "*"))
    old_test_images = glob(os.path.join(old_test_images_path, "*", "*"))
    old_test_masks = glob(os.path.join(old_test_masks_path, "*", "*", "*"))

    # Random shuffling
    random.shuffle(old_train_images)
    random.shuffle(old_test_images)

    print("There are "+str(len(old_train_images))+" old train images")
    print("There are "+str(len(old_test_images))+" old test images")

    old_images = [old_train_images, old_test_images]
    old_masks = [old_train_masks, old_test_masks]

    for idx, images in enumerate(old_images):

        split_index = int(len(images) * splits[idx][0])
        set1 = images[:split_index]
        set2 = images[split_index:]

        print("Start making the new "+sets[idx][0]+" set...")

        dest_images = os.path.join(dataset_name, image_folder, sets[idx][0])
        dest_masks = os.path.join(dataset_name, mask_folder, sets[idx][0])

        corrupted = parse_and_save(
            set1, old_masks[idx], dest_images, dest_masks)

        if corrupted:
            print(str(corrupted)+" were corrupted")

        print("Start making the new "+sets[idx][1]+" set...")

        dest_images = os.path.join(dataset_name, image_folder, sets[idx][1])
        dest_masks = os.path.join(dataset_name, mask_folder, sets[idx][1])

        corrupted = parse_and_save(
            set2, old_masks[idx], dest_images, dest_masks)

        if corrupted:
            print(str(corrupted)+" were corrupted")

    print("The new dataset is ready!")
