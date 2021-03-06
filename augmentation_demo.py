# import necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,
    help="path to input image")
ap.add_argument("-o","--output", required=True,
    help="path to output directory to store augmented images")
ap.add_argument("-p","--prefix", type=str, default="image",
    help="output filename prefix")
args = vars(ap.parse_args())

# load the input image, convert it to numpy array, and then reshape it to have an extra dimension
print("[INFO] loading example image...")


image = load_img(args["image"])  # a PIL image
image = img_to_array(image)      # convert to a numpy array
image = np.expand_dims(image, axis=0) # add one additional dimension at index 0


# construct the image generator for data augmentation then
# initialise the total number of images generated thus far
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")
total = 0

# construct the actual python generator
# aug.flow will return (x) x.shape=(1,h,w,3)
print("[INFO] generating images...")
imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
    save_prefix=args["prefix"], save_format="jpg")

# loop over example from out image data generator
for image in imageGen:
    # increment our counter
    total +=1

    # if we have reached 10 examples, break from the loop
    if total == 10:
        break;
