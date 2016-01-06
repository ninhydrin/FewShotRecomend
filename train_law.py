#!/usr/bin/env python
#from __future__ import print_function
import sys
import numpy as np
import six
import random
import pickle
import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
import scipy.io.wavfile as siw
import os
import argparse
import librosa

parser = argparse.ArgumentParser(description='pre train')
parser.add_argument('--train_type', '-t',default="main",choices=('main','pre'),
                    help='train type (main or pre)')
parser.add_argument('--epoch', '-e', default='5',type=int,
                    help='epoch num')
parser.add_argument('--b_size', '-b', default='20',type=int,
                    help='batchsize')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--learning', '-l', default="adam",choices=("adam","sgd"),
                    help='learning algorithm(adam ot sgd)')
parser.add_argument('--resume', '-r', default=1, type=int,choices=(0,1),
                    help='use init model when main train(0 or 1,default 1 use pre train model)')
args = parser.parse_args()

batchsize = args.b_size

is_pre = True if args.train_type == "pre" else False
for_dic = False

import models

if is_pre:
    print "start pre train:law"
    save_name = "pre_train/pre_law.model"
    model = models.Law(pre=True)
else:
    model = models.Law()
    save_name = "train_data/my_law.model"
    print "start main train:law"
    if args.resume:
        serializers.load_hdf5(save_name, model)
print "algorithm is",args.learning
    
train_list=[]

window_num=model.insize

if is_pre:
    pre_dir = "pre_train/preTrain"
    counter = 0
    val_data=[]
    train_data=[]
    val_list=[]
    data_dir = [i for i in os.listdir(pre_dir) if i[0] !="." and "wav" in i]
    assert len(data_dir)==1000
    for c,i in enumerate(data_dir):
        if c % 100 == 1 or c%100 == 2:
            y,sr = librosa.load(os.path.join(pre_dir,i),sr=22050,mono=False)
            five_p = y.shape[1]/20
            val_data.append((counter,y.mean(0)[five_p::10]))
        else:
            y,sr = librosa.load(os.path.join(pre_dir,i),sr=22050,mono=False)
            five_p = y.shape[1]/20
            train_data.append((counter,y.mean(0)[five_p::10]))
        if c % 100 == 99:
            counter+=1

    for cate,i in train_data:
        for j in range(i.size/window_num):
            train_list.append((cate,i[j*window_num:(j+1)*window_num]))
    for cate,i in val_data:
        for j in range(i.size/window_num):
            val_list.append((cate,i[j*window_num:(j+1)*window_num]))
    del train_data,val_data
    N_test = len(val_list)

else:
    dataset=[]
    for i in os.listdir("positive"):
        if ".wav" in i:            
            y,sr = librosa.load(os.path.join("positive",i),sr=44100,mono=False)
            five_p = y.shape[1]/20
            dataset.append((0,y.mean(0)[five_p:-five_p:100]))
    p_num=len(dataset)
    for i in os.listdir("negative"):
        if ".wav" in i:            
            y,sr = librosa.load(os.path.join("negative",i),sr=44100,mono=False)
            five_p = y.shape[1]/20
            dataset.append((1,y.mean(0)[five_p:-five_p:100]))
    n_num=len(dataset)-p_num
    for cate,i in dataset:
        for j in range(i.size/window_num):
            train_list.append((cate,i[j*window_num:(j+1)*window_num]))
    del dataset

N = len(train_list)

print 'train num = ',N
if is_pre:
    print "val num = ",N_test
else:
    print 'positive num = ',p_num,'negative num = ',n_num
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

xp = np if args.gpu < 0 else cuda.cupy


optimizer = optimizers.Adam() if args.learning == "adam" else optimizers.MomentumSGD(lr=0.0001, momentum=0.9)
optimizer.setup(model)

for epoch in six.moves.range(1, args.epoch + 1):
    print 'epoch ', epoch,"/",args.epoch #,":learning rate ",optimizer.lr
    if args.learning == "sgd":
        print "learning rate : ",optimizer.lr

    x_batch = np.ndarray((batchsize,1,1,model.insize), dtype=np.float32)
    y_batch = np.ndarray((batchsize,), dtype=np.int32)
    random.shuffle(train_list)

    sum_accuracy = 0
    sum_loss = 0

    batch_range=range(N)
    for i in six.moves.range(0, N, batchsize):
        if i+batchsize>N:
            x_batch = np.ndarray((N-i, 1,1,model.insize), dtype=np.float32)
            y_batch = np.ndarray((N-i,), dtype=np.int32)

        for num,j in enumerate(batch_range[i:i+batchsize]):
            x_batch[num][0][0]=train_list[j][1]
            y_batch[num]=train_list[j][0]

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
        sum_accuracy = 0
        sum_loss = 0
        model.train=False
        x_batch = np.ndarray((batchsize,1,1,model.insize), dtype=np.float32)
        y_batch = np.ndarray((batchsize,), dtype=np.int32)
        batch_range=range(N_test)
        for i in six.moves.range(0, N_test, batchsize):
            
            if i+batchsize>N_test:
                x_batch = np.ndarray((N_test-i,1,1,model.insize), dtype=np.float32)
                y_batch = np.ndarray((N_test-i,), dtype=np.int32)
                
            for num,j in enumerate(batch_range[i:i+batchsize]):
                x_batch[num][0][0]=val_list[j][1]
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
            
        print 'test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test)
    # Save the model and the optimizer
    #if epoch % 10 == 0:
    if epoch % 10 == 0:
        print 'save the model'
        serializers.save_hdf5(save_name, model)
        #print('save the optimizer')
        #serializers.save_hdf5(save_name, optimizer)
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
    dic["law"]=[posi_pre.mean(),nega_pre.mean()]
    pickle.dump(dic,open("train_data/threshold.dic","w"),-1)
#print('save the optimizer')
#serializers.save_hdf5(feature_name+'.state', optimizer)
