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


def get_prediction(image_path, model, classes, image_size, rescale=1. / 255):
    test_image = tf.keras.preprocessing.image.load_img(image_path,
                                                       target_size=image_size,
                                                       interpolation='bilinear')
    #print("### check test image shape : {s}".format(s=np.shape(test_image)))
    #print('Min: %.3f, Max: %.3f' % (np.expand_dims(test_image, axis=0).min(), np.expand_dims(test_image, axis=0).max()))
    #imgArray = np.expand_dims(test_image, axis=0) # bad idea to forget normalize
    imgArray = np.expand_dims(test_image, axis=0) * rescale
    #print('Min: %.3f, Max: %.3f' % (imgArray.min(), imgArray.max()))
    # plt.imshow(test_image)
    # plt.show()
    test_prediction_scores = model.predict(imgArray)
    #print(test_prediction_scores)
    prediction = tf.nn.softmax(test_prediction_scores).numpy()[0]
    test_predicted_index = np.argmax(test_prediction_scores)
    iterator = np.nditer(prediction,  flags=['f_index'])
    #print("\n### %%% Predictions for {} ###".format(image_path))
    res = []
    for i in iterator:
        res.append(" {} {:2.2f}%".format(classes[iterator.index], i * 100))
    print('#[{}]#'.format(','.join(res)))
    if image_path.lower().find(classes[test_predicted_index]) > -1:
        print("### ✔ ✔  Predicted label for {:10} is CORRECT : {:10} {:2.2f} percent confidence".format(
            image_path, classes[test_predicted_index], (100 * prediction[test_predicted_index])))
    else:
        print("### ⚠ ⚠  Predicted label for {:10} is WRONG : {:10} {:2.2f} percent confidence".format(
            image_path, classes[test_predicted_index], (100 * prediction[test_predicted_index])))
    return classes[test_predicted_index]

### RESULTS WITHOUT NORMALIZATION : 19 erros in 30 samples .... BAD idea
### ✔ ✔  Predicted label for test/daisy01.jpg is CORRECT : daisy      35.35 percent confidence
### ✔ ✔  Predicted label for test/daisy02.jpg is CORRECT : daisy      70.25 percent confidence
### ✔ ✔  Predicted label for test/daisy03.jpg is CORRECT : daisy      84.88 percent confidence
### ✔ ✔  Predicted label for test/daisy04.jpg is CORRECT : daisy      70.34 percent confidence
### ✔ ✔  Predicted label for test/daisy05.jpg is CORRECT : daisy      74.72 percent confidence
### ✔ ✔  Predicted label for test/daisy06.jpg is CORRECT : daisy      49.87 percent confidence
### ⚠ ⚠  Predicted label for test/dandelion01.jpg is WRONG : rose       43.05 percent confidence
### ⚠ ⚠  Predicted label for test/dandelion02.jpg is WRONG : daisy      48.46 percent confidence
### ⚠ ⚠  Predicted label for test/dandelion03.jpg is WRONG : daisy      63.70 percent confidence
### ✔ ✔  Predicted label for test/dandelion04.jpg is CORRECT : dandelion  42.64 percent confidence
### ✔ ✔  Predicted label for test/dandelion05.jpg is CORRECT : dandelion  31.92 percent confidence
### ✔ ✔  Predicted label for test/dandelion06.jpg is CORRECT : dandelion  59.03 percent confidence
### ⚠ ⚠  Predicted label for test/rose01.jpg is WRONG : daisy      37.06 percent confidence
### ⚠ ⚠  Predicted label for test/rose02.jpg is WRONG : daisy      41.48 percent confidence
### ⚠ ⚠  Predicted label for test/rose03.jpg is WRONG : dandelion  50.20 percent confidence
### ✔ ✔  Predicted label for test/rose04.jpg is CORRECT : rose       48.03 percent confidence
### ⚠ ⚠  Predicted label for test/rose05.jpg is WRONG : daisy      42.43 percent confidence
### ⚠ ⚠  Predicted label for test/rose06.jpg is WRONG : tulip      42.57 percent confidence
### ⚠ ⚠  Predicted label for test/sunflowers01.jpg is WRONG : dandelion  76.93 percent confidence
### ⚠ ⚠  Predicted label for test/sunflowers02.jpg is WRONG : rose       52.88 percent confidence
### ⚠ ⚠  Predicted label for test/sunflowers03.jpg is WRONG : dandelion  69.36 percent confidence
### ⚠ ⚠  Predicted label for test/sunflowers04.jpg is WRONG : daisy      41.50 percent confidence
### ⚠ ⚠  Predicted label for test/sunflowers05.jpg is WRONG : daisy      44.89 percent confidence
### ⚠ ⚠  Predicted label for test/sunflowers06.jpg is WRONG : dandelion  32.57 percent confidence
### ⚠ ⚠  Predicted label for test/tulips01.jpg is WRONG : dandelion  61.37 percent confidence
### ⚠ ⚠  Predicted label for test/tulips02.jpg is WRONG : rose       49.48 percent confidence
### ⚠ ⚠  Predicted label for test/tulips03.jpg is WRONG : daisy      50.48 percent confidence
### ⚠ ⚠  Predicted label for test/tulips04.jpg is WRONG : dandelion  39.44 percent confidence
### ⚠ ⚠  Predicted label for test/tulips05.jpg is WRONG : dandelion  34.97 percent confidence


### RESULTS WITH NORMALIZATION : 100% GOOD with data never seen before AND NORMALIZATION
### ✔ ✔  Predicted label for test/daisy01.jpg is CORRECT : daisy      91.57 percent confidence
### ✔ ✔  Predicted label for test/daisy02.jpg is CORRECT : daisy      96.15 percent confidence
### ✔ ✔  Predicted label for test/daisy03.jpg is CORRECT : daisy      96.76 percent confidence
### ✔ ✔  Predicted label for test/daisy04.jpg is CORRECT : daisy      94.02 percent confidence
### ✔ ✔  Predicted label for test/daisy05.jpg is CORRECT : daisy      94.82 percent confidence
### ✔ ✔  Predicted label for test/daisy06.jpg is CORRECT : daisy      93.88 percent confidence
### ✔ ✔  Predicted label for test/dandelion01.jpg is CORRECT : dandelion  97.70 percent confidence
### ✔ ✔  Predicted label for test/dandelion02.jpg is CORRECT : dandelion  75.43 percent confidence
### ✔ ✔  Predicted label for test/dandelion03.jpg is CORRECT : dandelion  96.14 percent confidence
### ✔ ✔  Predicted label for test/dandelion04.jpg is CORRECT : dandelion  97.40 percent confidence
### ✔ ✔  Predicted label for test/dandelion05.jpg is CORRECT : dandelion  99.05 percent confidence
### ✔ ✔  Predicted label for test/dandelion06.jpg is CORRECT : dandelion  90.55 percent confidence
### ✔ ✔  Predicted label for test/rose01.jpg is CORRECT : rose       94.74 percent confidence
### ✔ ✔  Predicted label for test/rose02.jpg is CORRECT : rose       97.16 percent confidence
### ✔ ✔  Predicted label for test/rose03.jpg is CORRECT : rose       79.87 percent confidence
### ✔ ✔  Predicted label for test/rose04.jpg is CORRECT : rose       99.06 percent confidence
### ✔ ✔  Predicted label for test/rose05.jpg is CORRECT : rose       90.98 percent confidence
### ✔ ✔  Predicted label for test/rose06.jpg is CORRECT : rose       94.56 percent confidence
### ✔ ✔  Predicted label for test/sunflowers01.jpg is CORRECT : sunflowers 96.70 percent confidence
### ✔ ✔  Predicted label for test/sunflowers02.jpg is CORRECT : sunflowers 97.20 percent confidence
### ✔ ✔  Predicted label for test/sunflowers03.jpg is CORRECT : sunflowers 97.77 percent confidence
### ✔ ✔  Predicted label for test/sunflowers04.jpg is CORRECT : sunflowers 98.05 percent confidence
### ✔ ✔  Predicted label for test/sunflowers05.jpg is CORRECT : sunflowers 90.58 percent confidence
### ✔ ✔  Predicted label for test/sunflowers06.jpg is CORRECT : sunflowers 92.64 percent confidence
### ✔ ✔  Predicted label for test/tulips01.jpg is CORRECT : tulip      91.98 percent confidence
### ✔ ✔  Predicted label for test/tulips02.jpg is CORRECT : tulip      62.89 percent confidence
### ✔ ✔  Predicted label for test/tulips03.jpg is CORRECT : tulip      95.30 percent confidence
### ✔ ✔  Predicted label for test/tulips04.jpg is CORRECT : tulip      98.42 percent confidence
### ✔ ✔  Predicted label for test/tulips05.jpg is CORRECT : tulip      84.23 percent confidence
### ✔ ✔  Predicted label for test/tulips06.jpg is CORRECT : tulip      91.51 percent confidence



