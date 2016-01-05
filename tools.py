#coding:utf-8
import os
import cv2 
import sys
from pylab import *
import pickle
#import scipy
#from scipy.io import wavfile
import numpy as np
import librosa

pre_train_dir="pre_train_music/"

mel_divide_num = 98
mfcc_divide_num = 98
chroma_divide_num = 14

def fprint(st):
    sys.stdout.write("\r{}".format(st))
    sys.stdout.flush()

def make_spec(path,save_name,N=4096,mono=False):#スペクトログラム
    y,sr = librosa.load(path,sr=44100,mono=mono)
    if mono:
        y = np.vstack([y,y])
    y=y.transpose(1,0)
    
    #figure(num=None, figsize=(6.61, 6.4), dpi=10, facecolor='w', edgecolor='k')#512*512
    figure(num=None, figsize=(3.31, 3.2), dpi=10, facecolor='w', edgecolor='k')#512*512
    pxx, freqs, bins, im = specgram(y.flatten(), NFFT=N, Fs=sr, noverlap=0, window=np.hamming(N))
    pxx = pxx.transpose(1,0)
    tick_params(labelbottom="off")
    tick_params(labelleft="off")
    tick_params(length=0)
    tick_params(width=0)
    num=1
    for i in range(len(y)/4):
        if pxx[-num].max() > 0:
            break
        num+=1
    xlim(0,(bins.max()-ceil(num*2048/44100.)))
    ylim(0,freqs.max())
    savefig(save_name, bbox_inches="tight",pad_inches=0.0)
    close()
    spec_image = cv2.imread(save_name)
    cv2.imwrite(save_name,cv2.resize(spec_image,(256,256)))

def slice_law(path,cate=-1,window_num=5000,num=20,stride_num = 0):
    y,sr = librosa.load(path,sr=44100,mono=False)
    five_p = y.shape[1]/num
    y=y.mean(0)[five_p:-five_p:100]
    if not stride_num:
        stride_num = (y.size-window_num)/(num-1)
    train_list = []
    if stride_num < window_num/4:
        num = 10
        stride_num = (y.size-window_num)/(num-1)
    if cate < 0:
        for i in range(num):
            train_list.append(y[i*stride_num:i*stride_num+window_num])
    else:
        for i in range(num):
            train_list.append((cate,y[i*stride_num:i*stride_num+window_num]))            
    return train_list


def get_mfcc(y,sr,n_mfcc=40): #メル周波数ケプストラム係数(2階微分まで使用)
    mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_d = librosa.feature.delta(mfcc_delta)
    return np.vstack([mfcc,mfcc_delta,mfcc_delta_d])

def get_chroma(y,sr,cqt=True,harmonic=False):#クロマベクトル
    if harmonic:
        return librosa.feature.chroma_cqt(divide_ham_per(y)[0],sr=sr)
    elif cqt:
        return librosa.feature.chroma_cqt(y=y[::2], sr=sr)
    else:
        S = np.abs(librosa.stft(y, n_fft=4096))
        return librosa.feature.chroma_stft(S=S, sr=sr)
   
def get_mel(y,sr):#メルスペクトログラム
    return librosa.feature.melspectrogram(y=y, sr=sr)
"""    

def get_pre_train():#事前学習用学習セットの作成
    positive_train=os.listdir(pre_train_dir)
    count=0
    pre_mel=[]
    pre_mfcc=[]
    pre_chroma=[]
    for i in positive_train:
        if "." in i: continue
        a=os.listdir(pre_train_dir+i)
        for j in a:
            if j[0]=="." or not ".au" in j: continue
            path =  os.path.join(pre_train_dir,i,j)
            make_spec(path,os.path.join("pre_train/pre_spec/",j[:-2]+"png"),mono=True)
            a,b,c = pre_vec(path)
            pre_mel.append((count,a))
            pre_mfcc.append((count,b))
            pre_chroma.append((count,c))            
            make_vec(path,count,(pre_mel,pre_mfcc,pre_chroma))            
        count+=1
    pickle.dump(pre_mel,open("pre_train/pre_mel","w"),-1)
    pickle.dump(pre_mfcc,open("pre_train/pre_mfcc","w"),-1)
    pickle.dump(pre_chroma,open("pre_train/pre_chroma","w"),-1)



def get_tempo(y,sr,only_tempo=1):
    if only_tempo:
        return librosa.beat.beat_track(y=y, sr=sr)[0]        
    else:
        return librosa.beat.beat_track(y=y, sr=sr)

def make_vec_botsu(path,c_num,t_dir):#特徴の確率分布を抽出(KL-ダイバージェンス用)
    print "convert ",path
    y,sr = librosa.load(path,sr=44100)
    
    #posi_tempo = (0,spec.get_tempo(y,sr))
    mel = get_mel(y,sr).transpose(1,0)
    mel = standardize(mel,mel_divide_num)
    #km = KMeans(n_clusters=cluster_num).fit(mel)    
    #mel = km.cluster_centers_.flatten()
    t_dir[0].append((c_num,mel))
    mfcc = get_mfcc(y,sr).transpose(1,0)
    mfcc = standardize(mfcc,mfcc_divide_num)
    #km = KMeans(n_clusters=cluster_num).fit(mfcc)    
    #mfcc = km.cluster_centers_.flatten()
    t_dir[1].append((c_num,mfcc))
    chroma = get_chroma(y[::4],sr).transpose(1,0)
    chroma = standardize(chroma,chroma_divide_num)
    #km = KMeans(n_clusters=cluster_num).fit(chroma)    
   #chroma = km.cluster_centers_.flatten()
    t_dir[2].append((c_num,chroma))
    print "done ",path
    #return path,mel,mfcc,chroma

def pre_vec(path):#事前学習用の特徴抽出
    print "convert ",path
    y,sr = librosa.load(path,sr=44100)
    y=y[-1300000:]
    mel = get_mel(y,sr).mean(1)
    mfcc = get_mfcc(y,sr).mean(1)
    chroma = get_chroma(y[::4],sr).mean(1)
    print "done ",path
    return mel,mfcc,chroma

"""
def divide_ham_per(y):
    return librosa.effects.hpss(y)


def get_feature(path):#入力されたpathの音楽から特徴抽出 return mel,mfcc,chroma
    div_num=5
    N=2048
    y,sr = librosa.load(path,sr=44100)
    y=y[len(y)/20:-len(y)/20]
    pxx, freqs, bins, im = specgram(y, NFFT=N, Fs=sr, noverlap=0, window=np.hamming(N))
    pxx = pxx.transpose(1,0)
    pxx_max=[i.argmax() for i in pxx]
    f_unit=pxx.shape[0]/div_num
    pxx_max = [argmax(pxx_max[i*f_unit:(i+1)*f_unit])+i*f_unit for i in range(div_num)]
    candidate = []
    count = 0
    cut_range=sr*15 #15秒
    mel=[]
    mfcc=[]
    chroma=[]
    for i in pxx_max:        
        left = i*N-cut_range if i*N-cut_range >= 0 else 0
        right = i*N+cut_range if i*N+cut_range <= y.shape[0] else y.shape[0]
        candidate.append(y[left:right])

    for i in candidate:
        mel.append(get_mel(i,sr).mean(1))
        #t_dir[0].append((c_num,mel))
        mfcc.append(get_mfcc(i,sr).mean(1))
        #t_dir[1].append((c_num,mfcc))
        c_len = len(i)/5000*5000 #5000刻みぐらいじゃないと応答が消える
        chroma.append(get_chroma(i[:c_len],sr).mean(1))
        #t_dir[2].append((c_num,chroma))
    return mel,mfcc,chroma

def standardize(feature_mat,d_num):#標準化
    av=feature_mat.shape[0]/d_num
    feature_mat = np.vstack([feature_mat[av*i:av*(i+1)].mean(0) for i in range(d_num)])
    return (feature_mat-feature_mat.mean(0))/feature_mat.std(0)
    
if __name__ == "__main__":
    pass

