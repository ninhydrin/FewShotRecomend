#! /bin/bash
function return_value {
    RET=$?
    if [ ${RET} -eq 1 ]; then
	echo "Stop All"
	exit 1
    fi
}
if [ $# -gt 0 ] ; then
    if [ $1 = "train" ] ; then
	gpu=-1
	if [ $# -eq 2 ]; then
	    gpu=$2
	fi	
	echo "start train your favorite music"
	#echo "tranfer params"
	#python transfer.py construct
	python trainMaker.py
	return_value
	python compute_mean.py 1
	python train_law.py -t main -g $gpu
	python train_spec.py -t main -g $gpu
	python train_mmc.py -t main -f mel -g $gpu
	python train_mmc.py -t main -f mfcc -g $gpu
	python train_mmc.py -t main -f chroma -g $gpu
	echo "end train"
    elif [ $1 = "predict" ] ; then
	if [ $# -lt 2 ] ; then
	    echo "usage : ./weakRecommender.sh predict path/to/music_directory"
	else 
	    echo "start predict target: "$2
	    python predictor.py $2
	fi
    elif [ $1 = "itunes" ] ; then
	if [ $# -ne 3 ] ; then
	    echo "start import iTunes"
	    python itunes.py
	else
	    echo "start import iTunes :"$2
	    python itunes.py --itunes $2
	fi
    elif [ $1 = "pre" ]; then
	echo "this is pre train mode.you don't need to use. but execute ?[Y/n]"
	read ans
	case `echo $ans |tr y Y` in 
	    Y* )
		gpu=-1
		if [ $# -eq 2 ]; then
		    gpu=$2
		fi	
		echo "start pre training"
		python compute_mean.py 0
		python train_law.py -t pre -g $gpu
		python train_spec.py -t pre -g $gpu
		python train_mmc.py -t pre -f mel -g $gpu
		python train_mmc.py -t pre -f mfcc -g $gpu
		python train_mmc.py -t pre -f chroma -g $gpu
		
		python transfer.py law
		python transfer.py spec
		python transfer.py mel
		python transfer.py mfcc
		python transfer.py chroma
		#python transfer.py divide	    
		echo "pre train complete!!";;
	    *) echo "stop pre training";;
	esac
    else
	echo -e "usage : ./weakRecommender.sh mode gpu_num or music_directory\n"
	echo "mode : train, predict or itunes"
    fi
else
    echo -e "usage : ./weakRecommender.sh mode gpu_num or music_directory\n"
    echo -e "you use this for the first time\n\n./weakRecommender.sh train gpu_num\n"
    echo -e "once the training is completed\n\n./tools.sh predict path/to/music_directory gpu_num\n\n"
    echo -e "default gpu_num is -1 (not use)"
fi
