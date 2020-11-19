import tensorflow as tf
import tensorflow_hub as hub
import numpy as np



def display_versions():
    """display_versions print the version of installed packages like tensorflow or numpy"""
    print("### TF  version:  {v}".format(v=tf.__version__))
    print("### Hub version: {v}".format(v=hub.__version__))
    print("### numpy  version:  {v}".format(v=np.__version__))


def is_gpu_available(verbose: bool = False):
    """is_gpu_available checks if a GPU is available

    :param verbose: allows to display additional debug information
    :type verbose: bool
    :return: True if GPU is present, False in all other cases
    :rtype: bool
    """
    gpuDevices = tf.config.list_physical_devices('GPU')
    if len(gpuDevices) > 0:
        gpu01 = gpuDevices[0]
        if gpu01.device_type == 'GPU':
            if verbose:
                print("### GPU is available at : {n}".format(n=gpu01.name))
            return True
        else:
            print("### GPU is NOT AVAILABLE")
            return False


def get_prediction(image_path, model, classes, image_size):
    test_image = tf.keras.preprocessing.image.load_img(image_path,
                                                       target_size=image_size,
                                                       interpolation='bilinear')
    # print("### check test image shape : {s}".format(s=np.shape(test_image)))
    # plt.imshow(test_image)
    # plt.show()
    test_prediction_scores = model.predict(np.expand_dims(test_image, axis=0))
    prediction = tf.nn.softmax(test_prediction_scores).numpy()[0]
    test_predicted_index = np.argmax(test_prediction_scores)
    iterator = np.nditer(prediction,  flags=['f_index'])
    print("\n### %%% Predictions for {} ###".format(image_path))
    for i in iterator:
        print("{:10} \twith a {:>.2f} percent confidence".format(classes[iterator.index], i * 100))

    if image_path.lower().find(classes[test_predicted_index]) > -1:
        print("### ✔ ✔  Predicted label for {:10} is CORRECT : {:10} {:2.2f} percent confidence".format(
            image_path, classes[test_predicted_index], (100 * prediction[test_predicted_index])))
    else:
        print("### ⚠ ⚠  Predicted label for {:10} is WRONG : {:10} {:2.2f} percent confidence".format(
            image_path, classes[test_predicted_index], (100 * prediction[test_predicted_index])))
    return classes[test_predicted_index]
