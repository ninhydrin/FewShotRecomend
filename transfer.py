import argparse
import os
import chainer
from chainer import serializers
import pickle
parser = argparse.ArgumentParser(description='model transfer')
parser.add_argument('model',default='mel',choices=('mel','mfcc','chroma','spec','law','construct','divide'),
                    help='choice model transfer (mel,mfcc,chroma,spec,law,construct or divide)')
args = parser.parse_args()
model_name = args.model

import models
if model_name == "chroma":
    src_model = models.Chroma(pre=True)
    dst_model = models.Chroma()
elif model_name == "mel":
    src_model = models.Mel(pre=True)
    dst_model = models.Mel()
elif model_name == "mfcc":
    src_model = models.Mfcc(pre=True)
    dst_model = models.Mfcc()
elif model_name == "spec":
    src_model = models.Spectro(pre=True)
    dst_model = models.Spectro()
elif model_name == "law":
    src_model = models.Law(pre=True)
    dst_model = models.Law()

if model_name == "construct":
    if os.path.exists("train_data/my_spec.model"):
        exit()
    print 'Construct Spectro model'
    src_conv = models.DivideConv()
    dst_model = models.Spectro(pre=True)
    serializers.load_hdf5("train_data/divide_conv.param", src_conv)
    dst_dict = {i[0]:i[1] for i in dst_model.namedparams()}
    for i in src_conv.namedparams():
        dst_dict[i[0]].data = i[1].data
    dst_model.fc6.W.data[:2048] = pickle.load(open("train_data/fc6_W_one.param"))
    dst_model.fc6.W.data[2048:] = pickle.load(open("train_data/fc6_W_two.param"))
    dst_model.fc6.b.data=pickle.load(open("train_data/fc6_b.param"))
    dst_model.fc7.W.data=pickle.load(open("train_data/fc7_W.param"))
    dst_model.fc7.b.data=pickle.load(open("train_data/fc7_b.param"))
    dst_model.fc_last.W.data=pickle.load(open("train_data/fc_last_W.param"))
    dst_model.fc_last.b.data=pickle.load(open("train_data/fc_last_b.param"))
    serializers.save_hdf5('train_data/my_spec.model',dst_model)
    os.remove("train_data/fc6_W_one.param")
    os.remove("train_data/fc6_W_two.param")
    os.remove("train_data/fc6_b.param")
    os.remove("train_data/fc7_W.param")
    os.remove("train_data/fc7_b.param")
    os.remove("train_data/fc_last_W.param")
    os.remove("train_data/fc_last_b.param")
    os.remove("train_data/divide_conv.param")
    print 'Construct complete!!'

elif model_name == "divide":
    print 'Divide Spectro model for git'
    src_model = models.Spectro()
    dst_conv = models.DivideConv()
    serializers.load_hdf5("train_data/my_spec.model", src_model)
    conv_dict = {i[0]:i[1] for i in dst_conv.namedparams()}
    for i in src_model.namedparams():
        if conv_dict.has_key(i[0]):
            conv_dict[i[0]].data = i[1].data
    serializers.save_hdf5('train_data/divide_conv.param',dst_conv)
    pickle.dump(src_model.fc6.W.data[:2048],open("train_data/fc6_W_one.param","w"),-1)
    pickle.dump(src_model.fc6.W.data[2048:],open("train_data/fc6_W_two.param","w"),-1)
    pickle.dump(src_model.fc6.b.data,open("train_data/fc6_b.param","w"),-1)
    pickle.dump(src_model.fc7.W.data,open("train_data/fc7_W.param","w"),-1)
    pickle.dump(src_model.fc7.b.data,open("train_data/fc7_b.param","w"),-1)
    pickle.dump(src_model.fc_last.W.data,open("train_data/fc_last_W.param","w"),-1)
    pickle.dump(src_model.fc_last.b.data,open("train_data/fc_last_b.param","w"),-1)
    os.remove('train_data/my_spec.model')
    print 'Divide complete!!'
else:
    print 'Load model from', args.model+".model"
    serializers.load_hdf5("pre_train/pre_"+model_name+".model",src_model)
    src_dict = {i[0]:i[1] for i in src_model.namedparams()}
    for i in dst_model.namedparams():
        if "fc_last" in i[0]:
            continue
        i[1].data=src_dict[i[0]].data
    print 'save ',"train_data/my_"+model_name+".model"
    serializers.save_hdf5("train_data/my_"+model_name+'.model',dst_model)


