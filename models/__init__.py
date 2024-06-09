from .lenet import *


def get_model(config):
    return globals()[config.architecture](config.num_classes)
