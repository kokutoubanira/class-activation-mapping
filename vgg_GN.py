#!/usr/bin/env python
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import functions
import numpy as np
from chainer.functions.array.reshape import reshape


# Network definition
class MLP(chainer.Chain):

    def __init__(self, num_class, train=True):
        super(MLP, self).__init__()
        self.size = 256
        self.fc = None
        self.feats = None
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.conv1 = L.Convolution2D(None, 64, 3, stride= 1, pad=1)
            self.gn1=L.GroupNormalization(32)
            self.conv2 = L.Convolution2D(None, 64, 3, stride= 1, pad=1)
            self.gn2=L.GroupNormalization(32)

            self.conv3 = L.Convolution2D(None, 128, 3, stride= 1, pad=1)
            self.gn3=L.GroupNormalization(32)
            self.conv4 = L.Convolution2D(None, 128, 3, stride= 1, pad=1)
            self.gn4=L.GroupNormalization(32)

            self.conv5 = L.Convolution2D(None, 256, 3, stride= 1, pad=1)
            self.gn5=L.GroupNormalization(32)
            self.conv6 = L.Convolution2D(None, 256, 3, stride= 1, pad=1)
            self.gn6=L.GroupNormalization(32)
            self.conv7 = L.Convolution2D(None, 256, 3, stride= 1, pad=1)
            self.gn7=L.GroupNormalization(32)

            self.conv8 = L.Convolution2D(None, 512, 3, stride= 1, pad=1)
            self.gn8=L.GroupNormalization(32)
            self.conv9 = L.Convolution2D(None, 512, 3, stride= 1, pad=1)
            self.gn9=L.GroupNormalization(32)
            self.conv10 = L.Convolution2D(None, 512, 3, stride= 1, pad=1)
            self.gn10=L.GroupNormalization(32)

            self.conv11 = L.Convolution2D(None, 512, 3, stride= 1, pad=1)
            self.gn11=L.GroupNormalization(32)
            self.conv12 = L.Convolution2D(None, 512, 3, stride= 1, pad=1)
            self.gn12=L.GroupNormalization(32)
            self.conv13 = L.Convolution2D(None, 512, 3, stride= 1, pad=1)
            self.gn13=L.GroupNormalization(32)

            self.fc16 = L.Linear(None, num_class)  # n_units -> n_out


    def forward(self, x):
        h = F.relu(self.gn1(self.conv1(x)))
        h = F.max_pooling_2d(F.relu(self.gn2(self.conv2(h))), 2, stride=2)
        h = F.relu(self.gn3(self.conv3(h)))
        h = F.max_pooling_2d(F.relu(self.gn4(self.conv4(h))), 2, stride=2)
        
        h = F.relu(self.gn5(self.conv5(h)))
        h = F.relu(self.gn6(self.conv6(h)))
        h = F.max_pooling_2d(F.relu(self.gn7(self.conv7(h))), 2, stride=2)

        h = F.relu(self.gn8(self.conv8(h)))
        h = F.relu(self.gn9(self.conv9(h)))
        h = F.max_pooling_2d(F.relu(self.gn10(self.conv10(h))), 2, stride=2)

        h = F.relu(self.gn11(self.conv11(h)))
        h = F.relu(self.gn12(self.conv12(h)))

        h = F.relu(self.gn13(self.conv13(h)))

        if not chainer.config.train:
            self.feats = h

        h = self._global_average_pooling_2d(h)

        return self.fc16(h)



    def _global_average_pooling_2d(self, x):
        n, channel, rows, cols = x.shape
        h = F.average_pooling_2d(x, (rows, cols), stride=1)
        h = reshape(h, (n, channel))
        return h


    


