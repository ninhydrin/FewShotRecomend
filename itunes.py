import plistlib
import os
from tqdm import tqdm
import numpy as np
import sys
import argparse
import urllib

parser = argparse.ArgumentParser(description='This is simple learning sample creater. This create symlink from itunes to posidir and negadir. Default play count threshold is Median')
parser.add_argument('--itunes',default=os.path.join(os.environ['HOME'],"Music/iTunes"),help='path to iTunes directory')
parser.add_argument('--threshold', '-t',default=-1,type=int,help="Threshold. defalt : Median")
args = parser.parse_args()

try:
    xml = plistlib.readPlist(os.path.join(args.itunes,"iTunes Music Library.xml"))
except IOError:
    print "I cant find {}.\nUsage : ./weakRecommender.sh itunes path/to/iTunes".format(args.itunes)
    exit(1)
    
count_list=[]
med_list=[]
for i in tqdm(xml["Tracks"]):
    track_info=xml["Tracks"][i]
    if track_info.has_key("Play Count"):
        med_list.append(track_info["Play Count"])
        count_list.append((urllib.unquote(track_info["Location"][7:]),track_info["Play Count"]))

med = np.median(np.array(med_list))

posi_dir = "positive"
nega_dir = "negative"
for path in os.listdir(posi_dir):
    if os.path.islink(os.path.join(posi_dir,path)):
        os.remove(os.path.join(posi_dir,path))
for path in os.listdir(nega_dir):
    if os.path.islink(os.path.join(nega_dir,path)):
        os.remove(os.path.join(nega_dir,path))
        
for track in tqdm(count_list):
    if track[1] >= med:
        try:
            root,ext = os.path.splitext(track[0])
            if ext == ".wav":
                root,ext = os.path.split(track[0])
                os.symlink(track[0],os.path.join(posi_dir,ext))
        except OSError:
            pass
    else:
        try:
            root,ext = os.path.splitext(track[0])
            if ext == ".wav":
                root,ext = os.path.split(track[0])
                os.symlink(track[0],os.path.join(nega_dir,ext))
        except OSError:
            pass
