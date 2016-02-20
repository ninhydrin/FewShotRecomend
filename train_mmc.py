#!/usr/bin/env python
#from __future__ import print_function
import sys
import numpy as np
import six
import random
import pickle
import chainer
import os
#from chainer import computational_graph
from chainer import cuda
#import chainer.links as L
from chainer import optimizers
from chainer import serializers
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='train favorite music')

parser.add_argument('--feature', '-f',default="chroma",choices=('mel','mfcc',"chroma"),
                    help='feature type (mel or mfcc or chroma)')
parser.add_argument('--train_type', '-t',default="main",choices=('pre','main'),
                    help='train type(pre_train or main_train)')
parser.add_argument('--epoch', '-e', default='40',type=int,
                    help='epoch num')
parser.add_argument('--b_size', '-b', default='20',type=int,
                    help='batchsize')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--learning', '-l', default="adam",choices=("adam","sgd"),
                    help='learning algorithm(adam ot sgd)')

args = parser.parse_args()

batchsize = args.b_size
is_pre = True if args.train_type == "pre" else False
for_dic = False
print "start ",args.train_type," train:",args.feature

data_dir= "pre_train" if is_pre else "train_data"


if is_pre:
    model_name="pre_"+args.feature
    val_list = []
    train_list = pickle.load(open(os.path.join(data_dir,model_name)))
    for i in reversed(range(0,1000,50)):
        val_list.append(train_list.pop(i))
    N_test = len(val_list)
    N = len(train_list)
    print 'train num = ',N,'val num = ',N_test

else:
    model_name="my_"+args.feature
    posi = pickle.load(open(os.path.join(data_dir,"posi_"+args.feature)))
    nega = pickle.load(open(os.path.join(data_dir,"nega_"+args.feature)))
    p_num=len(posi)
    n_num=len(nega)
    train_list = posi+nega
    del posi,nega
    N = p_num+n_num
    print 'positive num = ',p_num,'negative num = ',n_num
save_name = os.path.join(data_dir,model_name)    

import models

if args.feature == "chroma": 
    model = models.Chroma(pre = is_pre)
elif args.feature == "mel":
    model = models.Mel(pre = is_pre)
elif args.feature == "mfcc":
    model = models.Mfcc(pre = is_pre)

if not is_pre:
    print 'Load model from', save_name+".model"
    serializers.load_hdf5(save_name+".model", model)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

optimizer = optimizers.Adam() if args.learning == "adam" else optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)
print "algorithm is ",args.learning
for epoch in tqdm(xrange(1, args.epoch + 1)):
    #print 'epoch ', epoch,"/",args.epoch
    #if args.learning == "sgd":
    #    print "learning rate : ",optimizer.lr
    x_batch = np.ndarray((batchsize, model.insize), dtype=np.float32)
    y_batch = np.ndarray((batchsize,), dtype=np.int32)
    random.shuffle(train_list)

    sum_accuracy = 0
    sum_loss = 0
    count = 0

    batch_range=range(N)
    for i in six.moves.range(0, N, batchsize):
        if i+batchsize>N:
            x_batch = np.ndarray((N-i, model.insize), dtype=np.float32)
            y_batch = np.ndarray((N-i,), dtype=np.int32)
        for num,j in enumerate(batch_range[i:i+batchsize]):
            x_batch[num]=train_list[j][1]
            y_batch[num]=train_list[j][0]
        

        x = chainer.Variable(xp.asarray(x_batch))
        t = chainer.Variable(xp.asarray(y_batch))

        optimizer.update(model, x, t)

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    #print 'train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N)

    if epoch == args.epoch and not is_pre:
        for_dic = True
        N_test = N
        val_list = train_list
        posi_pre = np.array([],dtype=np.float32)
        nega_pre = np.array([],dtype=np.float32)
        
    if is_pre or for_dic:
        model.train = False
        sum_accuracy = 0
        sum_loss = 0
        x_batch = np.ndarray((batchsize, model.insize), dtype=np.float32)
        y_batch = np.ndarray((batchsize,), dtype=np.int32)
        batch_range=range(N_test)
        for i in six.moves.range(0, N_test, batchsize):

            if i+batchsize>N_test:
                x_batch = np.ndarray((N_test-i, model.insize), dtype=np.float32)
                y_batch = np.ndarray((N_test-i,), dtype=np.int32)

            for num,j in enumerate(batch_range[i:i+batchsize]):
                x_batch[num]=val_list[j][1]
                y_batch[num]=val_list[j][0]

            x = chainer.Variable(xp.asarray(x_batch),volatile='on')
            t = chainer.Variable(xp.asarray(y_batch),volatile='on')
            loss = model(x, t)
            if for_dic:
                for label,prob in zip(t.data,model.pre.data):
                    if label == 0:
                        posi_pre = np.append(posi_pre,prob[0])
                    else:
                        nega_pre = np.append(nega_pre,prob[0])


            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)

        #print 'test  mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test)
    if epoch % 10 == 0:
        #print 'save the model'
        serializers.save_hdf5(save_name+".model", model)
        #print('save the optimizer')
        #serializers.save_hdf5(save_name+'.state', optimizer)
    if args.learning == "sgd":
        optimizer.lr *= 0.97
    model.train = True
# Save the model and the optimizer
print "last save"
serializers.save_hdf5(save_name+".model", model)
if for_dic:
    if not os.path.exists("train_data/threshold.dic"):
        dic={}
    else:
        dic = pickle.load(open("train_data/threshold.dic"))
    dic[args.feature]=[posi_pre.mean(),nega_pre.mean()]
    pickle.dump(dic,open("train_data/threshold.dic","w"),-1)

#print('save the optimizer')
#serializers.save_hdf5(save_name+'.state', optimizer)

