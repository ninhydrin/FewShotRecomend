#! /bin/bash

if [ $1 == "train" ] ; then
    gpu=-1
    if [ $# == 2 ]; then
	gpu=$2
    fi	
    echo "start train your favorite music"
    python trainMaker.py
    python compute_mean.py 1
    python transfer.py construct
    python train_law.py -t main -g $gpu
    python train_spec.py -t main -g $gpu
    python train_mmc.py -t main -f mel -g $gpu
    python train_mmc.py -t main -f mfcc -g $gpu
    python train_mmc.py -t main -f chroma -g $gpu
    echo "end train"
elif [ $1 == "predict" ] ; then
    if [ $# < 2 ] ; then
	echo "usage : ./tools.sh predict path/to/music_directory (save_text 0 or 1)"
    else
	echo "start predict   target:"$2
	python predictor.py $2 -o $3
    fi
elif [ $1 == "pre" ]; then
    echo "this is pre train mode.you don't need to use. but execute ?[Y/n]"
    read ans
    case `echo $ans |tr y Y` in 
	Y* )
	    gpu=-1
	    if [ $# == 2 ]; then
		gpu=$2
	    fi	
	    echo "start pre training"
	    python train_law.py -t pre -g $gpu
	    #python train_spec.py -t pre -g $gpu
	    python train_mmc.py -t pre -f mel -g $gpu
	    python train_mmc.py -t pre -f mfcc -g $gpu
	    python train_mmc.py -t pre -f chroma -g $gpu
	    echo "compute spectrogram mean"
	    python compute_mean.py 0
	    echo "tranfer params"
	    python transfer.py -m law
	    python transfer.py -m spec
	    python transfer.py -m mel
	    python transfer.py -m mfcc
	    python transfer.py -m chroma
	    echo "pre train complete!!";;
	 *) echo "stop pre training";;
     esac
else
    echo -e "usage : ./tools.sh mode gpu_num or music_directory\n"
    echo -e "you use this for the first time\n\n./tools.sh train gpu_num\n"
    echo -e "once the training is completed\n\n./tools.sh predict path/to/music_directory gpu_num\n\n"
    echo -e "default gpu_num is -1 (not use)"
fi


