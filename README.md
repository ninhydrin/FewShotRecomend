# FewShotRecommend
    
## Requirements
- Python 2.7+
- NumPy
- Pickle
- Argparse
- OpenCV
- Librosa
- Skimage
- Pylab
- Chainer (1.5.1): https://github.com/pfnet/chainer

 Anaconda-2.3.0 の使用を推奨

## How to use
positiveにお気に入りの音楽を、negativeに嫌い（というか趣向に合わない）音楽を入れる（.wavにのみ対応）。
その後FewShotRecommendディレクトリ内で

./weakRecommender.sh train

を実行し好みを学習（PCのスペックによっては時間がかかる）。
学習終了後は音楽ファイルの入ったディレクトリを指定して

./weakRecommender.sh predict path/to/music_dir

resultディレクトリに結果が保存される。


