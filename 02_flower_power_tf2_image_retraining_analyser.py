import os
import glob
## tells Tensorflow not to use any CUDA devices !
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt
import numpy as np
from golib import utils as u

# set the levels of messages we want from Tensorflow backend
#  0| DEBUG, 1| INFO,2| WARNING,3| ERROR
# tf.get_logger().setLevel('WARNING')
# tf.autograph.set_verbosity(2)

u.display_versions()
u.is_gpu_available(True)

module_selection = ("mobilenet_v2_100_224", 224)
handle_base, pixels = module_selection
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

print("### List of possible categories :")
for class_index, class_string in enumerate(classes, start=0):
    print("[{i}]\t{n}".format(i=class_index, n=class_string))

saved_model_path = "model/saved_flowers_model_retraining"
print("### Now will load model from {m}".format(m=saved_model_path))
model = load_model(saved_model_path)

print("### Now trying model.predict with a brand new images !!")
test_images = []
for i in glob.glob('test/*.jpg'):
    test_images.append(i)

test_images.sort()
for f in test_images:
    u.get_prediction(f, model, classes, IMAGE_SIZE)
