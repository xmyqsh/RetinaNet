# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from .RetinaNet_train_test import RetinaNet_train_test

def get_network(name):
    """Get a network by name."""
    if name.split('_')[0] == 'RetinaNet':
        if name.split('_')[1] == 'train':
           return RetinaNet_train_test()
        else:
           raise KeyError('Unknown dataset: {}'.format(name))
    else:
        raise KeyError('Unknown dataset: {}'.format(name))

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
