#!/usr/bin/env python
import argparse
import os
import sys
import numpy
import six.moves.cPickle as pickle
from skimage import io

if len(sys.argv)!=2:
    print "usage : python compute_mean.py num (num is 0 or 1. 0 is pre_tain,1 is main_train)"
    exit()
    
target_dir = "train_data" if sys.argv[1]=="1" else "pre_train"
sum_image = None
count = 0
for i in os.listdir(target_dir):
    if os.path.isdir(os.path.join(target_dir,i)):
        dirpath=os.path.join(target_dir,i)
        for j in os.listdir(dirpath):
            if ".png" in j:
                filepath = os.path.join(dirpath,j)
                image = numpy.asarray(io.imread(filepath)).transpose(2, 0, 1)
                if sum_image is None:
                    sum_image = numpy.ndarray(image.shape, dtype=numpy.float32)
                    sum_image[:] = image
                else:
                    sum_image += image
                count += 1
                sys.stderr.write('\r{}'.format(count))
                sys.stderr.flush()
sys.stderr.write('\n')
mean = sum_image / count
pickle.dump(mean, open(os.path.join(target_dir,"mean.npy"), 'wb'), -1)
