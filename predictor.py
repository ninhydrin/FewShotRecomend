#coding:utf-8
#from __future__ import print_function
#import json
#import multiprocessing
import os
import sys
model_dir = "train_data"
root_path = os.path.abspath(os.path.dirname(__file__))
if not os.path.exists(os.path.join(model_dir,"threshold.dic")):
    print "you need to execute train before predict!!"
    sys.exit()

from cv2 import resize
import numpy as np
#import copy
import chainer
import six.moves.cPickle as pickle
#from six.moves import queue
from tqdm import tqdm
from skimage import io
from chainer import cuda
from chainer import optimizers
#from chainer.functions import caffe
from chainer import serializers
#from scipy import ndimage
import audioop
import argparse
import models
import tools
import warnings

result_dir = "result"
parser = argparse.ArgumentParser(description='this is predictor. if you input music directory,this extract your favorite likely music.')
parser.add_argument('target',nargs ="*",help='path to directory that contains musics you want to predict')
#parser.add_argument('--gpu', '-g',default=-1,type=int,help="gpu ID")
args = parser.parse_args()

#if args.gpu >= 0:
#    cuda.get_device(args.gpu).use()
#    model.to_gpu()
xp = np#cuda.cupy if args.gpu >= 0 else np

mean_image = pickle.load(open("train_data/mean.npy", 'rb'))
mean_image =mean_image[:3,:,:]

def get_result(x_batch,y_batch,model):
    x = chainer.Variable(xp.asarray(x_batch))
    y = chainer.Variable(xp.asarray(y_batch))
    a=model(x,y)
    return model.pre.data

def read_image(path,insize=256):
    image = np.asarray(io.imread(path))
    if image.shape[0]==2:
        image=image[0]
    image = resize(image,(insize,insize)).transpose(2, 0, 1)
    image=image[:3,:,:]
    image=image.astype(np.float32)
    image -= mean_image
    image = image/255.0
    return image

def predict_mmc(model,in_list,b_size=5):
    x_batch = np.ndarray((b_size, model.insize), dtype=np.float32)
    y_batch = np.zeros((b_size,), dtype=np.int32)
    for j in range(5):
        x_batch[j]=in_list[j]
    x = chainer.Variable(xp.asarray(x_batch))
    y = chainer.Variable(xp.asarray(y_batch))
    a=model(x,y)
    return model.pre.data


def predict_law(model,path): #1曲あたり20個取り出す（重複あり）。無理なときは10個取り出す
    law_list = tools.slice_law(path)
    x_batch = np.ndarray((len(law_list),1,1,model.insize), dtype=np.float32)
    for i in range(len(law_list)):
        x_batch[i][0][0] = law_list[i]
    y_batch = np.zeros((len(law_list),), dtype=np.int32)
    return get_result(x_batch,y_batch,model)

def predict_spec(model,path):
    temp_spec = os.path.join(result_dir,"temp.png")
    tools.make_spec(path,temp_spec)
    x_batch = np.ndarray((1,3,model.insize,model.insize), dtype=np.float32)
    y_batch = np.zeros((1,), dtype=np.int32)
    x_batch[0] = read_image(temp_spec)
    return get_result(x_batch,y_batch,model)[0]

def dir_sort(path):
    #n_path=""
    #for i in path:
    #    n_path +=i+" "
    #path = n_path[:-1]
    m_list = os.listdir(path)
    m_list = [os.path.join(path,i) for i in m_list if ".wav" in i and i[0] != "."]
    return m_list

def main(m_list):

    mel_model = models.Mel()
    mfcc_model = models.Mfcc()
    chroma_model = models.Chroma()
    spec_model = models.Spectro()
    law_model = models.Law()
    
    serializers.load_hdf5(os.path.join(model_dir,"my_mel.model"), mel_model)
    serializers.load_hdf5(os.path.join(model_dir,"my_mfcc.model"), mfcc_model)
    serializers.load_hdf5(os.path.join(model_dir,"my_chroma.model"), chroma_model)
    serializers.load_hdf5(os.path.join(model_dir,"my_law.model"), law_model)
    serializers.load_hdf5(os.path.join(model_dir,"my_spec.model"), spec_model)

    candidate = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in tqdm(m_list):
            try:
                p_n=[]
                #print  "begin",i,"({0}/{1})".format(num+1,len(m_list))
                mel,mfcc,chroma = tools.get_feature(i)
                p_n.append(predict_mmc(mel_model,mel).mean(0))
                #tools.fprint("mel done   ")
                p_n.append(predict_mmc(mfcc_model,mfcc).mean(0))
                #tools.fprint("mfcc done  ")
                p_n.append(predict_mmc(chroma_model,chroma).mean(0))
                #tools.fprint("chroma done")
                p_n.append(predict_law(law_model,i).mean(0))
                #tools.fprint("law done   ")        
                p_n.append(predict_spec(spec_model,i))
                #tools.fprint("spec done  ")
                candidate.append((i,p_n))
                #tools.fprint("end        \n")
            except audioop.error:
                pass
        return candidate

if __name__ == "__main__":
    dir_path = " ".join(args.target)
    m_list = dir_sort(dir_path)
    #m_list = args.target
    candidate=main(m_list)
    os.remove(os.path.join(result_dir,"temp.png"))
    threshold=pickle.load(open("train_data/threshold.dic"))
    th_list=[threshold["mel"],threshold["mfcc"],threshold["chroma"],threshold["law"],threshold["spec"]]
    th_list=[sum(i)/2 for i in th_list]
    result_txt = open(os.path.join(result_dir,"result.txt"),"w")        
    for i in candidate:
        result_txt.write(i[0]+"\n")
        count=0.
        for th,j in zip(th_list,i[1]):
            count += j[0]-th
            result_txt.write("{0:.6f}\n".format(j[0]))
        comment = "Neither"
        if count > 0.25:
            comment = "Like"
        elif count < 0:
            comment = "Hate"
        result_txt.write(comment+"\n\n")
    result_txt.close()
