"""
Script for util functions
"""
import os
from keras import models
from complexnn import *

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)


def load_model(problem):
    if problem.endswith('.h5'):
        filename = problem
    else:
        filename = os.path.join('%s_model.h5' % problem)
    try:
        model = models.load_model(filename, {"QuaternionConv2D": conv.QuaternionConv2D,
                                             "QuaternionGRU": recurrent.QuaternionGRU,
                                             "QuaternionDense": dense.QuaternionDense})
        print("\nModel loaded successfully from file %s\n" % filename)
    except OSError:
        print("\nModel file %s not found!!!\n" % filename)
        model = None
    return model