import tensorflow as tf
import tensorflow_hub as hub
import itertools
import os
import matplotlib.pylab as plt
import numpy as np
from golib import utils as u

u.display_versions()
u.is_gpu_available(True)

print("### based on Google Tutorial at : https://www.tensorflow.org/hub/tutorials/tf2_image_retraining")
module_selection = ("mobilenet_v2_100_224", 224)
handle_base, pixels = module_selection
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("### Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

BATCH_SIZE = 32
print("### will load images flower_photos from tensorflow web (or from your cache in .keras/datasets/flower_photos")
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

do_data_augmentation = False
if do_data_augmentation:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        horizontal_flip=True,
        width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2,
        **datagen_kwargs)
else:
    train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True, **dataflow_kwargs)

do_fine_tuning = False

print("### Building model with", MODULE_HANDLE)
model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_generator.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,) + IMAGE_SIZE + (3,))
model.summary()
print("### Training the model")
model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
    metrics=['accuracy'])

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size
hist = model.fit(
    train_generator,
    epochs=5, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps).history
print("### Training the model")
plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0, 2])
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0, 1])
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])


def get_class_string_from_index(index):
    for class_string, class_index in valid_generator.class_indices.items():
        if class_index == index:
            return class_string


x, y = next(valid_generator)
image = x[0, :, :, :]
true_index = np.argmax(y[0])
plt.imshow(image)
plt.axis('off')
plt.show()

# Expand the validation image to (1, 224, 224, 3) before predicting the label
prediction_scores = model.predict(np.expand_dims(image, axis=0))
predicted_index = np.argmax(prediction_scores)
print("### True label: " + get_class_string_from_index(true_index))
print("### Predicted label: " + get_class_string_from_index(predicted_index))

saved_model_path = "model/saved_flowers_model"
tf.saved_model.save(model, saved_model_path)

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