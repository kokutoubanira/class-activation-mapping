import argparse
import vgg_GN as M 
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import horce2zebra_dataset as  hzd
from chainer import cuda


class BaseBackprop(object):

    def __init__(self, model):
        self.model = model
        self.size = 256
        self.xp = model.xp

    def get_l(self, x):
        y = self.model.predictor(x)
        l = self.model.predictor.feats
        w = self.model.predictor.fc16.W
        return w, l


class class_activate_mapping(BaseBackprop):

    def __init__(self, model):
        super(class_activate_mapping, self).__init__(model)

    def generate(self, x, label):
        weight,layer = self.get_l(x)

        gcam = self.xp.tensordot(weight.data[label], layer.data[0], axes=(0, 0))
        gcam = self.xp.maximum(gcam, 0)

        return chainer.cuda.to_cpu(gcam)