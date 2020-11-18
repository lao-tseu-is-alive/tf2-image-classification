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