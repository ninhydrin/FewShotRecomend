#!/usr/bin/env python
#from __future__ import print_function
import pickle
import numpy as np
import six
import random
import chainer
from skimage import io
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
import argparse
import os

parser = argparse.ArgumentParser(description='train spectrogram')
parser.add_argument('--train_type', '-t',default="main",choices=('pre','main'),
                    help='train type(pre_train or main_train)')
parser.add_argument('--mean', '-m', default='mean.npy',
                    help='mean image file path')
parser.add_argument('--epoch', '-e', default=5,type=int,
                    help='epoch num')
parser.add_argument('--b_size', '-b', default=20,type=int,
                    help='batchsize')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--learning', '-l', default="adam",choices=("adam","sgd"),
                    help='learning algorithm(adam ot sgd)')
args = parser.parse_args()

batchsize = args.b_size

import models
is_pre = True if args.train_type == "pre" else False
for_dic = False


if is_pre:
    print "start pre train:spec"
    save_name = "pre_train/pre_spec.model"
    model=models.Spectro(pre=True)
else:
    print "start main train:spec"
    model=models.Spectro()
    save_name = "train_data/my_spec.model"
    serializers.load_hdf5(save_name, model)

print "algorithm is ",args.learning
target_dir = "train_data" if args.train_type == "main" else "pre_train"
mean_image = pickle.load(open(os.path.join(target_dir,"mean.npy"), 'rb'))[:3,:,:]

def read_image(path):
    # Data loading routine
    image = np.asarray(io.imread(path)).transpose(2, 0, 1)
    image = image.astype(np.float32)
    if image.shape[2]!=3:
        image=image[:3,:,:]
    image -= mean_image
    image /= 255
    return image

def load_image_list(path):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append((pair[0], np.int32(pair[1])))
    return tuples

if is_pre:
    train_list = load_image_list("pre_train/train_cross.txt")
    val_list = load_image_list("pre_train/val_cross.txt")
    train_list = [(os.path.join("pre_train/pre_spec",i[0]),i[1]) for i in train_list]
    val_list = [(os.path.join("pre_train/pre_spec",i[0]),i[1]) for i in val_list]
    N_test = len(val_list)
else:
    data_dir = "train_data"
    spec_dir=os.path.join(data_dir,"posi_spectro")
    posi = [(os.path.join(spec_dir,i),0) for i in os.listdir(spec_dir) if not i[0]=="."]
    spec_dir=os.path.join(data_dir,"nega_spectro")
    nega = [(os.path.join(spec_dir,i),1) for i in os.listdir(spec_dir) if not i[0]=="."]
    p_num = len(posi)
    n_num = len(nega)
    train_list = posi+nega
    print 'positive num = ',p_num,'negative num = ',n_num
    del posi,nega

N=len(train_list)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

optimizer = optimizers.Adam() if args.learning == "adam" else optimizers.MomentumSGD(lr=0.0001, momentum=0.9)

optimizer.setup(model)

for epoch in six.moves.range(1, args.epoch + 1):
    print "epoch ", epoch,"/",args.epoch
    if args.learning == "sgd":
        print "learning rate : ", optimizer.lr
    x_batch = np.ndarray((batchsize,3,model.insize,model.insize), dtype=np.float32)
    y_batch = np.ndarray((batchsize,), dtype=np.int32)
    sum_accuracy = 0
    sum_loss = 0
    batch_range=range(N)
    random.shuffle(train_list)

    for i in six.moves.range(0, N, batchsize):
        if i+batchsize>N:
            x_batch = np.ndarray((N-i, 3,model.insize,model.insize), dtype=np.float32)
            y_batch = np.ndarray((N-i,), dtype=np.int32)

        for num,j in enumerate(batch_range[i:i+batchsize]):
            x_batch[num]=read_image(train_list[j][0])
            y_batch[num]=train_list[j][1]
        
        x = chainer.Variable(xp.asarray(x_batch))
        t = chainer.Variable(xp.asarray(y_batch))
        
        optimizer.update(model, x, t)

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print 'train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N)
    if epoch == args.epoch and not is_pre:
        for_dic = True
        N_test = N
        val_list = train_list
        posi_pre = np.array([],dtype=np.float32)
        nega_pre = np.array([],dtype=np.float32)

    if is_pre or for_dic:
        # evaluation
        model.train = False
        x_batch = np.ndarray((batchsize,3,model.insize,model.insize), dtype=np.float32)
        y_batch = np.ndarray((batchsize,), dtype=np.int32)
        sum_accuracy = 0
        sum_loss = 0
        batch_range=range(N_test)
        for i in six.moves.range(0, N_test, batchsize):
            if i+batchsize>N_test:
                x_batch = np.ndarray((N_test-i, 3,model.insize,model.insize), dtype=np.float32)
                y_batch = np.ndarray((N_test-i,), dtype=np.int32)

            for num,j in enumerate(batch_range[i:i+batchsize]):
                x_batch[num]=read_image(val_list[j][0])
                y_batch[num]=val_list[j][1]

            x = chainer.Variable(xp.asarray(x_batch),volatile="on")
            t = chainer.Variable(xp.asarray(y_batch),volatile="on")

            loss = model(x, t)
            if for_dic:
                for label,prob in zip(t.data,model.pre.data):
                    if label == 0:
                        posi_pre = np.append(posi_pre,prob[0])
                    else:
                        nega_pre = np.append(nega_pre,prob[0])


            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)

        print 'test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test)
    print 'save the model'
    serializers.save_hdf5(save_name, model)
    if args.learning == "sgd":
        optimizer.lr *= 0.97
    model.train = True
# Save the model and the optimizer
print "last save"
serializers.save_hdf5(save_name, model)
if for_dic:
    if not os.path.exists("train_data/threshold.dic"):
        dic={}
    else:
        dic = pickle.load(open("train_data/threshold.dic"))
    dic["spec"]=[posi_pre.mean(),nega_pre.mean()]
    pickle.dump(dic,open("train_data/threshold.dic","w"),-1)

#print('save the optimizer')
#serializers.save_hdf5('pre_spec.state', optimizer)
