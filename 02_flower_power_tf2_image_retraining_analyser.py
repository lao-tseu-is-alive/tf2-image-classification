import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import itertools
import os
import matplotlib.pylab as plt
import numpy as np
from golib import utils as u

u.display_versions()
u.is_gpu_available(True)

module_selection = ("mobilenet_v2_100_224", 224)
handle_base, pixels = module_selection
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

BATCH_SIZE = 32

data_dir = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

datagen_kwargs = dict(rescale=1. / 255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                       interpolation="bilinear")

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)


def get_class_string_from_index(index):
    for class_string, class_index in valid_generator.class_indices.items():
        if class_index == index:
            return class_string

saved_model_path = "model/saved_flowers_model"
print("### Now will load model from {m}".format(m=saved_model_path))
model = load_model(saved_model_path)

print("### Now trying model.predict with a brand new image !!")
test_image = tf.keras.preprocessing.image.load_img('test/tulip/tulip01.jpg',
                                                   target_size=IMAGE_SIZE,
                                                   interpolation='bilinear')
print("### check test image shape : {s}".format(s=np.shape(test_image)))
plt.imshow(test_image)
plt.show()
test_prediction_scores = model.predict(np.expand_dims(test_image, axis=0))
test_predicted_index = np.argmax(test_prediction_scores)
print("### Predicted label for tulip01 is ...: " + get_class_string_from_index(test_predicted_index))

print("### Now trying model.predict with a brand new image !!")
test_image02 = tf.keras.preprocessing.image.load_img('test/tulip/tulip02.jpg',
                                                     target_size=IMAGE_SIZE,
                                                     interpolation='bilinear')
print("### check test image shape : {s}".format(s=np.shape(test_image02)))
plt.imshow(test_image02)
plt.show()
test_prediction_scores02 = model.predict(np.expand_dims(test_image02, axis=0))
test_predicted_index02 = np.argmax(test_prediction_scores02)
print("### Predicted label for tulip02 is ...: " + get_class_string_from_index(test_predicted_index02))
