from __future__ import print_function 

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import matplotlib.pyplot as plt

import argparse
import vgg_GN as M 
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import horce2zebra_dataset as  hzd
import vgg_cam
import numpy as np
import cv2


def main():
    
    #モデルの読み込み
    model = L.Classifier(M.MLP(2))
    
    test = hzd.h2zdataset("train_data")
    test_filepath = hzd.h2zdataset("train_data")._filename

    chainer.serializers.load_npz("vgg_horse2zebra.npz", model)
    

    x, t = test[14]
    x = x[None, ...]
    with chainer.using_config('train', False):


        y = F.softmax(model.predictor(x))
        y = y.data
        print(y)
        print('label:', t)
        print('predicted_label:', y.argmax(axis=1)[0])

        cam = vgg_cam.class_activate_mapping(model)
        cam_gen = cam.generate(x, t)
        cam_gen = np.uint8(cam_gen * 255 / cam_gen.max())
        cam_gen = cv2.resize(cam_gen, (model.predictor.size, model.predictor.size))
       
        
        
        #src = cv2.imread(test_filepath[1505], 1)
        #src = cv2.resize(src, (model.predictor.size, model.predictor.size))
        x = x.reshape((x.shape[1] , x.shape[2], x.shape[3]))
        x = x.transpose(1,2,0)
        x = x * 255

        heatmap = cv2.applyColorMap(cam_gen, cv2.COLORMAP_JET)
        cam_gen = np.float32(x) + np.float32(heatmap)
        cam_gen = 255 * cam_gen /cam_gen.max()

    
        cv2.imwrite('gcam.png', cam_gen)
        I = cv2.imread('gcam.png')
        cv2.imshow('window', I)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()    
