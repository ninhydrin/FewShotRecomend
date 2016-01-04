import os
import sys
import librosa
import tools
#from sklearn.cluster import KMeans
import pickle
import numpy as np
#import multiprocessing

posi_dir="positive"
nega_dir="negative"

data_dir="train_data"

positive_sample = [i for i in os.listdir(posi_dir) if ".wav" in i]
negative_sample = [i for i in os.listdir(nega_dir) if ".wav" in i]

def p_dump(a,b):
        pickle.dump(a,open(os.path.join(data_dir,b),"w"),-1)
        
#cluster_num=14
#rand_num=2

#pool=multiprocessing.Pool(8)
#a=pool.map(make_vec,positive_sample)

def make_train():
    #posi_tempo=[]
    print "start positive"
    posi_mel=[]
    posi_chroma=[]
    posi_mfcc=[]
    #posi_tempo=[]
    #nega_tempo=[]
    for i in positive_sample:
        tools.make_vec(os.path.join(posi_dir,i),0,(posi_mel,posi_mfcc,posi_chroma))
        tools.get_feature(os.path.join(posi_dir,i),0,(posi_mel,posi_mfcc,posi_chroma))
        tools.make_spec(os.path.join(posi_dir,i),os.path.join(data_dir,"posi_spectro",i[:-3]+"png"))

    p_dump(posi_mel,"posi_mel")
    p_dump(posi_mfcc,"posi_mfcc")
    p_dump(posi_chroma,"posi_chroma")

    del posi_mel
    del posi_mfcc
    del posi_chroma
    
    print "start negative"
    nega_mel=[]
    nega_chroma=[]
    nega_mfcc=[]

    for i in negative_sample:
        tools.make_vec(os.path.join(nega_dir,i),0,(nega_mel,nega_mfcc,nega_chroma))
        tools.get_feature(os.path.join(nega_dir,i),1,(nega_mel,nega_mfcc,nega_chroma))
        tools.make_spec(os.path.join(nega_dir,i),os.path.join(data_dir,"nega_spectro",i[:-3]+"png"))
    p_dump(nega_mel,"nega_mel")
    p_dump(nega_mfcc,"nega_mfcc")
    p_dump(nega_chroma,"nega_chroma")

if __name__ == "__main__":    
    make_train()
    

